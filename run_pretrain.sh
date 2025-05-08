{
export CUDA_VISIBLE_DEVICES="0";
export TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas;

python trainer_pcqm4mv2.py --root ./data/pcqm4mv2 --filename pretrain  --train_size 3378406 --val_size 100 --test_size 100 \
    --dataset pcqm4mv2 --batch_size 128 --save_every_n_epochs 5  --test_every_n_epochs 1   --check_val_every_n_epoch 1 \
    --init_lr 5e-4 --min_lr 1e-6 --weight_decay 1e-16 --scheduler cosine --max_steps 4000000  --accumulate_grad_batches 2 --inference_batch_size 128 \
    --pos_mask --denoising --mask_ratio 0.25  --trans_version v6 --attn_activation silu --max_epochs 30 

exit
}
