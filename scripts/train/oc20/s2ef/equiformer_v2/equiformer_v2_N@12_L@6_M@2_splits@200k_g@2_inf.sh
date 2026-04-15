# Running on Devide 2 and 3
CUDA_VISIBLE_DEVICES=2,3 python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 main_oc20.py \
    --distributed \
    --num-gpus 2 \
    --mode predict \
    --config-yml 'oc20/configs/s2ef/200k/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml' \
    --checkpoint 'checkpoints/eq2_83M_2M.pt' \
    --run-dir 'models/oc20/s2ef/200k/equiformer_v2/N@12_L@6_M@2/bs@32_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@2_inf' \
    --print-every 200 \
    --amp
