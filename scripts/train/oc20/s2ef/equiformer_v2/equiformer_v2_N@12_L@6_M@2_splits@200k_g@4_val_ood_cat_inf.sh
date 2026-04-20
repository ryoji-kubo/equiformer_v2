python -u -m torch.distributed.launch --nproc_per_node=4 main_oc20.py \
    --distributed \
    --num-gpus 4 \
    --mode validate \
    --config-yml 'oc20/configs/s2ef/200k/equiformer_v2/equiformer_v2_N@12_L@6_M@2_epochs@30_val_ood_cat.yml' \
    --checkpoint 'checkpoints/eq2_83M_2M.pt' \
    --run-dir 'models/oc20/s2ef/200k/equiformer_v2/N@12_L@6_M@2/val_ood_cat_inf' \
    --print-every 200 \
    --amp
