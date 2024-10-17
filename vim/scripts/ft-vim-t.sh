##!/bin/bash
#conda activate <your_env>
cd /home/yifan/mamba_prune/vim;

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --num_workers 25 --data-path /home/zlkong/Documents/ImageNet_new --output_dir ./output/vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --epochs 30 --finetune <path_to_pt_ckpt> --no_amp

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 64 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --num_workers 25 --data-path /home/zlkong/Documents/ImageNet_new --output_dir ./output/vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --epochs 30 --no_amp

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
        --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
        --finetune /home/yifan/mamba_prune/vim/Vim-tiny-midclstok/vim_t_midclstok_76p1acc.pth \
        --batch-size 64 \
        --lr 5e-6 \
        --min-lr 1e-5 \
        --warmup-lr 1e-5 \
        --drop-path 0.0 \
        --weight-decay 1e-8 \
        --num_workers 25 \
        --data-path /home/imagenet \
        --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
        --epochs 30 \
        --no_amp > "./ft_log/mamba_progressive_start_10_gap_10_ratio_0.75_new.txt" 2>&1 &
        # --no_amp > "./training_log/mamba_direct_progressive_start_10_gap_10_ratio_0.75_new.txt" 2>&1 &