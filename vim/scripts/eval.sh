cd /home/yifan/mamba_prune/vim/;

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --eval --resume /home/zlkong/mamba_prune/vim/ckpt/Vim-tiny-midclstok/vim_t_midclstok_76p1acc.pth  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /home/zlkong/Documents/ImageNet_new