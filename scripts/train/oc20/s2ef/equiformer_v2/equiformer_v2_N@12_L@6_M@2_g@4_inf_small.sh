python -u -m torch.distributed.launch --nproc_per_node=4 main_oc20.py \
    --distributed \
    --num-gpus 4 \
    --mode validate \
    --config-yml 'oc20/configs/s2ef/200k/equiformer_v2/equiformer_v2_N@12_L@6_M@2_epochs@30_inf_small_batch_small.yml' \
    --checkpoint 'checkpoints/eq2_83M_2M.pt' \
    --run-dir 'models/oc20/s2ef/200k/equiformer_v2/N@12_L@6_M@2/bs@32_lr@2e-4_wd@1e-3_epochs@30_warmup-epochs@0.1_g@4_inf_small' \
    --print-every 200 \
    --amp
