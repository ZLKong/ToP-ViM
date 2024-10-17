#!/bin/bash
# conda activate <your_env>
# cd <path_to_Vim>/vim;
cd /home/yifan/mamba_prune/vim;

CUDA_VISIBLE_DEVICES="0,1,2,3"
python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.2" --master_port=29501  --use_env main.py \
        --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
        --batch-size 64 \
        --lr 5e-6 \
        --min-lr 1e-5 \
        --warmup-lr 1e-5 \
        --drop-path 0.0 \
        --weight-decay 1e-8 \
        --num_workers 25 \
        --data-path /home/imagenet \
        --output_dir ./output/direct_start_5_gap_9_ratio_0.7_fix_vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
        --epochs 30 \
        --finetune /home/yifan/mamba_prune/vim/Vim-small-midclstok/vim_s_midclstok_80p5acc.pth \
        --no_amp > "./ft_log/mamba_small_direct_progressive_start_5_gap_9_ratio_0.7_fix.txt" 2>&1 &
