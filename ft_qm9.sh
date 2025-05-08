{
export CUDA_VISIBLE_DEVICES="0";
export TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas;

python trainer_qm9.py --root ./data/qm9 --filename homo --init_lr 5e-4 --min_lr 1e-6 --weight_decay 1e-16 --scheduler cosine  \
    --dataset qm9 --batch_size 128  --trans_version v6 --attn_activation silu  --warmup_steps 1000 \
    --max_steps 2000000  --max_epochs 1500  --lr_cosine_length 2000000 --dataset_arg homo 


exit
}