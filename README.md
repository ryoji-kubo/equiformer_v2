# EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations

**[Paper](https://arxiv.org/abs/2306.12059)** | **[OpenReview](https://openreview.net/forum?id=mCOBKZmrzD)** | **[Poster](docs/equiformer_v2_poster.pdf)**

This repository contains the official PyTorch implementation of the work "EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations" (ICLR 2024).
We provide the code for training the base model setting on the OC20 S2EF-2M and S2EF-All+MD datasets.

Additionally, EquiformerV2 has been incorporated into [OCP repository](https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/core/models/equiformer_v2) and used in [Open Catalyst demo](https://open-catalyst.metademolab.com/).

In our subsequent [work](https://arxiv.org/abs/2403.09549), we find that we can generalize self-supervised learning similar to BERT, which we call **DeNS** (**De**noising **N**on-Equilibrium **S**tructures), to 3D atomistic systems to improve the performance of EquiformerV2 on energy and force predictions. 
Please refer to the [paper](https://arxiv.org/abs/2403.09549) and the [code](https://github.com/atomicarchitects/DeNS) for further details.

The next generation, EquiformerV3, is [here](https://github.com/atomicarchitects/equiformer_v3).
We further improve efficiency, expressivity, and generality and achieve state-of-the-art results on direct and gradient prediction and on tasks requiring higher-order derivatives.


<p align="center">
	<img src="fig/equiformer_v2.png" alt="photo not available" width="98%" height="98%">
</p>

<p align="center">
	<img src="fig/equiformer_v2_speed_accuracy_tradeoffs.png" alt="photo not available" width="98%" height="98%">
</p>

<p align="center">
	<img src="fig/equiformer_v2_oc20_results.png" alt="photo not available" width="98%" height="98%">
</p>

<p align="center">
	<img src="fig/equiformer_v2_adsorbml_results.png" alt="photo not available" width="98%" height="98%">
</p>

<p align="center">
	<img src="fig/equiformer_v2_oc22_results.png" alt="photo not available" width="98%" height="98%">
</p>


## Content ##
0. [Environment Setup](#environment-setup)
0. [Changelog](#changelog)
0. [Training](#training)
0. [File Structure](#file-structure)
0. [Checkpoints](#checkpoints)
0. [Citation](#citation)
0. [Acknowledgement](#acknowledgement)



## Environment Setup ##


### Environment 

See [here](docs/env_setup.md) for setting up the environment.


### OC20

The OC20 S2EF dataset can be downloaded by following instructions in their [GitHub repository](https://github.com/Open-Catalyst-Project/ocp/blob/5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd/DATASET.md#download-and-preprocess-the-dataset).

For example, we can download the OC20 S2EF-2M dataset by running:
```
    cd ocp # no longer correct, use below cd
    cd fairchem
    python scripts/download_data.py --task s2ef --split "2M" --num-workers 8 --ref-energy
    # for downloading 200k split for debugging:
    python scripts/download_data.py --task s2ef --split "200k" --num-workers 8 --ref-energy
```
We also need to download the `"val_id"` data split to run training.

```bash
python scripts/download_data.py --task s2ef --split "val_id" --num-workers 8 --ref-energy
```

For validation on all four subsplits for inference, you have to repeat the above process for `"val_ood_ads"`, `"val_ood_cat"`, and `"val_ood_both"`.

After downloading, place the datasets under `datasets/oc20/` by using `ln -s`:
```
    cd datasets
    mkdir oc20
    cd oc20
    ln -s ~/ocp/data/s2ef s2ef # no longer correct
    ln -s ../../fairchem/data/s2ef s2ef
```

To train on different splits like All and All+MD, we can follow the same link above to download the datasets.

You also need to create `metadata.npz` for load balancing.

- We need to modify `fairchem/scripts/make_lmdb_sizes.py` and modify line 21 as follows:
```diff
    neighbors = None
-   if hasattr(data, "edge_index"):
+   if getattr(data, "edge_index", None) is not None:
        neighbors = data.edge_index.shape[1]
```

Afterwards, perform the following:
```bash
cd fairchem
python scripts/make_lmdb_sizes.py --data-path data/s2ef/200k/train --num-workers 8
```


## Changelog ##


Please refer to [here](docs/changelog.md).


## Training ##


### OC20

1. We train EquiformerV2 on the OC20 **S2EF-2M** dataset by running:
    
    ```bash
        sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_splits@2M_g@multi-nodes.sh
    ```
    The above script uses 2 nodes with 8 GPUs on each node.
    
    If there is an import error, it is possible that [`ocp/ocpmodels/common/utils.py`](https://github.com/Open-Catalyst-Project/ocp/blob/5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd/ocpmodels/common/utils.py#L329) is not modified. 
    Please follow [here](docs/env_setup.md) for details.

    We can also run training on 8 GPUs on 1 node:
    ```bash
        sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_splits@2M_g@8.sh
    ```

    We can also run training on 2 GPUs on 1 node for the 200k data split:
    ```bash
        sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_splits@200k_g@2.sh
    ```

    To run it sequentially with 1 GPU on the 200k data split:
    ```bash
    python main_oc20.py \
    --num-gpus 1 \
    --mode train \
    --config-yml 'oc20/configs/s2ef/200k/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml' \
    --run-dir 'models/oc20/s2ef/200k/equiformer_v2/N@12_L@6_M@2/bs@32_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@1' \
    --print-every 200 \
    --amp
    ```

    To run inference on the validation split with the pretrained model, first download the pretrained model following [this](#checkpoints). Then place the pretrained model in `checkpoints`. One example is specified in the following script
    ```bash
        sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_g@4_inf.sh
    ```

    We have also performed the training on a 200k split (`eq2_83M_200k.pt`) using the following two scripts:
    ```bash
    # for training
    sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_splits@200k_g@4_small_valid.sh
    # for testing
    scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_splits@200k_g@4_val_inf.sh
    ```


**Reproduction Result**

|Model	|Split	|Download	|val force MAE (meV / Å) |val energy MAE (meV) |
|---	|---	|---	|---	|---	| 
|EquiformerV2 (Reported)	|2M	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_83M_2M.pt) \| [config](oc20/configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2_epochs@30.yml)	|19.4 | 278 |
| **Reproduction of Inference** | | | | |
|Inf-only on val_iid split	|2M	|-|16.7 | 217 |
|Inf-only on val_ood_ads split	|2M	|-|18.1|261|
|Inf-only on val_ood_cat split	|2M	|-|19.3|271|
|Inf-only on val_ood_both split	|2M	|-|23.5|365|
|Inf-only on all val split (avg)	|2M	|-|19.4|278|
| **Reproduction of Training** | | | | |
|EquiformerV2 on val_iid split |200k |[checkpoint](checkpoints/eq2_83M_200k.pt) \| [config_train](oc20/configs/s2ef/200k/equiformer_v2/equiformer_v2_N@12_L@6_M@2_small_valid.yml) \| [config_eval](oc20/configs/s2ef/200k/equiformer_v2/equiformer_v2_N@12_L@6_M@2_inf.yml)	|31.2|347|



2. We train **EquiformerV2 (153M)** on OC20 **S2EF-All+MD** by running:
    ```bash
        sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@20_L@6_M@3_splits@all+md_g@multi-nodes.sh
    ```
    The above script uses 16 nodes with 8 GPUs on each node.

3. We train **EquiformerV2 (31M)** on OC20 **S2EF-All+MD** by running:
    ```bash
        sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@8_L@4_M@2_splits@all+md_g@multi-nodes.sh
    ```
    The above script uses 8 nodes with 8 GPUs on each node.
    
4. We can train EquiformerV2 with **DeNS** (**De**noising **N**on-Equilibrium **S**tructures) as an auxiliary task to further improve the performance on energy and force predictions. Please refer to the [code](https://github.com/atomicarchitects/DeNS) for details.


## File Structure ##

1. [`nets`](nets) includes code of different network architectures for OC20.
2. [`scripts`](scripts) includes scripts for training models on OC20.
3. [`main_oc20.py`](main_oc20.py) is the code for training, evaluating and running relaxation.
4. [`oc20/trainer`](oc20/trainer) contains code for the force trainer as well as some utility functions.
5. [`oc20/configs`](oc20/configs) contains config files for S2EF.


## Checkpoints ##

We provide the checkpoints of EquiformerV2 trained on S2EF-2M dataset for 30 epochs, EquiformerV2 (31M) trained on S2EF-All+MD, and EquiformerV2 (153M) trained on S2EF-All+MD.
|Model	|Split	|Download	|val force MAE (meV / Å) |val energy MAE (meV) |
|---	|---	|---	|---	|---	| 
|EquiformerV2	|2M	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_83M_2M.pt) \| [config](oc20/configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2_epochs@30.yml)	|19.4 | 278 |
|EquiformerV2 (31M)|All+MD |[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt) \| [config](oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml) |16.3 | 232 |
|EquiformerV2 (153M) |All+MD | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt) \| [config](oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml) |15.0 | 227 |


## Citation ##

Please consider citing the works below if this repository is helpful:

- [EquiformerV2](https://arxiv.org/abs/2306.12059):
    ```bibtex
    @inproceedings{
        equiformer_v2,
        title={{EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations}}, 
        author={Yi-Lun Liao and Brandon Wood and Abhishek Das* and Tess Smidt*},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2024},
        url={https://openreview.net/forum?id=mCOBKZmrzD}
    }
    ```

- [eSCN](https://arxiv.org/abs/2302.03655):
    ```bibtex
    @inproceedings{
        escn,
        title={{Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs}},
        author={Passaro, Saro and Zitnick, C Lawrence},
        booktitle={International Conference on Machine Learning (ICML)},
        year={2023}
    }
    ```

- [Equiformer](https://arxiv.org/abs/2206.11990):
    ```bibtex
    @inproceedings{
        equiformer,
        title={{Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs}},
        author={Yi-Lun Liao and Tess Smidt},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2023},
        url={https://openreview.net/forum?id=KwmPfARgOTD}
    }
    ```

- [OC20 dataset](https://arxiv.org/abs/2010.09990):
    ```bibtex
    @article{
        oc20,
        author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
        title = {{Open Catalyst 2020 (OC20) Dataset and Community Challenges}},
        journal = {ACS Catalysis},
        year = {2021},
        doi = {10.1021/acscatal.0c04525},
    }
    ```

Please direct questions to Yi-Lun Liao (ylliao@mit.edu).


## Acknowledgement ##

Our implementation is based on [PyTorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [e3nn](https://github.com/e3nn/e3nn), [timm](https://github.com/huggingface/pytorch-image-models), [ocp](https://github.com/Open-Catalyst-Project/ocp), [Equiformer](https://github.com/atomicarchitects/equiformer).