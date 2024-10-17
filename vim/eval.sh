cd /your/vim/directory  
CUDA_VISIBLE_DEVICES=0 


#tiny
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py  \
        --eval \
        --resume Vim-tiny-midclstok/vim_t_midclstok_76p1acc.pth \
        --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
        --data-path /home/imagenet



#small
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 --use_env main.py  \
        --eval \
        --resume Vim-small-midclstok/vim_s_midclstok_80p5acc.pth \
        --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
        --data-path /home/imagenet # > "./ft_log/mamba_small_progressive_start_5_gap_4_ratio_0.8_test.txt" 2>&1 &
