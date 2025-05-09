{
export NCCL_P2P_DISABLE=1;
export CUDA_VISIBLE_DEVICES="0";
export TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas;

python trainer_md17.py  --disable_compile --root .data/md17 --filename ft_md17_aspirin --min_lr 1e-6 --init_lr 5e-4  --scheduler cosine \
    --dataset md17 --dataset_arg md17_aspirin --weight_decay 1e-16 --max_epochs 1400 --batch_size 80  --inference_batch_size 64  \
    --trans_version v6 --attn_activation silu --warmup_steps 1000  --max_steps 2000000  --lr_cosine_length 2000000  
}