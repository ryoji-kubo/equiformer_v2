# EquiformerV2 Training Pipeline (OC20)

This document outlines the step-by-step execution flow of the EquiformerV2 training pipeline when executing the `main_oc20.py` entry point. 

The repository is built on top of the **Open Catalyst Project (OCP)** framework, which uses a dynamic `registry` to load Python classes directly from strings defined in a YAML configuration file.

---

## 1. Initialization and Setup (`main_oc20.py`)

When you launch a training script (e.g., `sh scripts/.../equiformer_v2_..._small_valid.sh`), the execution begins at the bottom of `main_oc20.py`:

1. **Config Parsing:** Command line arguments and the specified YAML file (`--config-yml`) are merged into a single `config` dictionary via `build_config()`.
2. **Runner Instantiation:** The `Runner()(config)` is invoked.
3. **Distributed Setup:** The runner calls `oc20.trainer.dist_setup.setup(config)` to initialize the multi-GPU environment (using PyTorch DDP or MPI depending on the cluster).
4. **Task & Trainer Assignment:** 
   * The script looks at `config["mode"]` (which is `train`) and fetches `TrainTask` from the registry.
   * The script looks at `config["trainer"]` (which is `forces_v2`) and fetches `ForcesTrainerV2` from the registry.
   * It sets up the task by passing the trainer to it (`self.task.setup(self.trainer)`) and begins the run loop via `self.task.run()`.

---

## 2. Trainer Setup (`oc20/trainer/base_trainer_v2.py`)

Before training begins, the `ForcesTrainerV2` class must initialize the model, data, and optimizers. It inherits its initialization logic from `BaseTrainerV2`.

During `self.load()`:
1. **Datasets:** PyTorch `DataLoader`s are created using OCP's LMDB dataset classes (`trajectory_lmdb_v2`).
2. **Model:** The `equiformer_v2` model is dynamically instantiated from `nets/equiformer_v2/equiformer_v2_oc20.py` and wrapped in `DistributedDataParallel`.
3. **Loss Functions:** Loss criteria are loaded (typically `mae` for energy and `l2mae` for forces).
4. **Optimizer (`optim_factory.py`):** A custom AdamW optimizer is created. Because Equiformer uses 1D tensor product weights, weight decay is selectively applied. Biases, layer norms, and 1D embedding frequencies are explicitly excluded from decay.
5. **Scheduler:** A learning rate scheduler (e.g., Cosine with Warmup) is initialized.

---

## 3. The Training Loop (`oc20/trainer/forces_trainer_v2.py`)

The `TrainTask` executes the `train()` method inside `ForcesTrainerV2`. Unlike standard PyTorch scripts that use external engines, OCP trainers contain monolithic training loops.

For every epoch, and for every batch in the `train_loader`:

### A. Forward Pass
* `out = self._forward(batch)` calls the model's forward pass.
* This is wrapped in an `amp.autocast()` context, enabling Mixed Precision (FP16/BF16) to save VRAM and speed up operations.

### B. Loss Calculation
* `loss = self._compute_loss(out, batch)` is executed.
* The predicted energy is compared against the DFT ground-truth energy (`batch.y_relaxed`).
* The predicted forces are compared against the DFT ground-truth forces (`batch.force`).
* Both losses are combined using an `energy_coefficient` and `force_coefficient` (e.g., forces are often weighted 100x higher than energy).

### C. Backward Pass and Optimization
* `loss.backward()` calculates the gradients. (Handled safely via the AMP `scaler`).
* `self._backward(loss)` is invoked:
  * Gradients are unscaled and clipped using `torch.nn.utils.clip_grad_norm_` (crucial to prevent explosive gradients in higher-degree spherical harmonics).
  * `optimizer.step()` updates the model weights.
  * An Exponential Moving Average (EMA) of the model weights is updated.

---

## 4. Inside the Model (`nets/equiformer_v2/equiformer_v2_oc20.py`)

When `self._forward(batch)` is called, the data flows through the `EquiformerV2_OC20` architecture:

1. **Graph Construction:** `self.generate_graph(data)` computes pairwise distances and edges between atoms within a `max_radius`.
2. **Rotation Matrices:** `self._init_edge_rot_mat(...)` builds 3x3 rotation matrices for every edge. This is required to align the spherical harmonic representations to the edge direction.
3. **Initial Embeddings:** 
   * Atoms are embedded into `SO3_Embedding`s using their atomic numbers.
   * Edge distances are expanded using a Gaussian Radial Basis Function (`GaussianSmearing`).
4. **Transformer Blocks:** The data passes through `self.num_layers` of `TransBlockV2` (located in `transformer_block.py`). Each block consists of:
   * **SO(2) Graph Attention:** Embeddings are rotated to align with edges, convolved via `SO2_Convolution`, processed through attention, and rotated back.
   * **FeedForward Network (FFN):** Point-wise transformations using `S2Activation` or `GateActivation`.
5. **Energy & Force Heads:** 
   * The output node embeddings are passed through an `energy_block` and pooled to predict a single energy value per graph.
   * If `regress_forces` is True, the embeddings are also passed through a separate `force_block` to predict a 3D vector (x, y, z) for every atom.

---

## 5. Logging and Evaluation

Throughout the training loop:
* **Logging (`logger.py`):** Every `--print-every` steps (e.g., 200), metrics (loss, MAE, learning rate, time) are written to the console and saved to `debug.log`. Tensorboard or Weights & Biases (wandb) also log these metrics.
* **Validation:** Every `eval_every` steps (e.g., 5000), the trainer pauses to run `self.validate(split="val")`. This runs the model on the validation dataset in `torch.no_grad()` mode to check generalization performance without updating weights.
* **Checkpoints:** The `best_checkpoint.pt` is saved whenever the primary metric (e.g., `forces_mae`) reaches a new minimum.
