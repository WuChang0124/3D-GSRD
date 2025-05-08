import math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from model.retrans import RelaTransEncoder
from model.autoencoder import AutoEncoder
import argparse
from data_provider.pcqm4mv2_dm import PCQM4MV2DM
from training_utils import custom_callbacks,load_encoder_params
from atomref import Atomref

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

class LinearWarmupCosineLRSchedulerV2:
    def __init__(
        self,
        optimizer,
        max_iters,
        min_lr,
        init_lr,
        warmup_iters=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.lr_decay_iters = max_iters

    def get_lr(self, it):
        # 1) linear warmup for warmup_steps steps
        if it < self.warmup_iters:
            return self.init_lr * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.init_lr - self.min_lr)

    def step(self, cur_step):
        lr = self.get_lr(cur_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
     
    
class AutoEncoderTrainer(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.criterion = nn.MSELoss()
        self.criterion_contrast = nn.CosineSimilarity(dim=1)
        self.save_hyperparameters(ignore='model')

    def forward(self, batch):
        if self.args.pos_mask and self.args.denoising:
            prediction, pred_noise,vec,rep_2d,rep_3d = self.model(batch)
            target = batch.mask_coord_label
            loss_pos = self.criterion(pred_noise,batch.pos_target[~batch.pos_mask])
            excess_norm = F.relu(vec.norm(dim=-1) - self.args.delta)
            loss_vec_norm = 0.01 * excess_norm.mean()
            loss1 = self.criterion(prediction,target)
            loss_contrast = -self.criterion_contrast(rep_2d,rep_3d).mean()
            return loss1,loss_contrast,loss_pos,loss_vec_norm
        
        if not self.args.pos_mask and self.args.denoising:
            pred_noise,vec,rep_2d,rep_3d = self.model(batch)
            loss_pos = self.criterion(pred_noise,batch.pos_target)
            loss_contrast = -self.criterion_contrast(rep_2d,rep_3d).mean()
            excess_norm = F.relu(vec.norm(dim=-1) - self.args.delta)
            loss_vec_norm = 0.01 * excess_norm.mean()
            return loss_contrast,loss_pos,loss_vec_norm
        
        if self.args.pos_mask and not self.args.denoising:
            prediction,vec, rep_2d,rep_3d = self.model(batch)
            target = batch.mask_coord_label
            loss1 = self.criterion(prediction,target)
            loss_contrast = -self.criterion_contrast(rep_2d,rep_3d).mean()
            excess_norm = F.relu(vec.norm(dim=-1) - self.args.delta)
            loss_vec_norm = 0.01 * excess_norm.mean()
            return loss1,loss_contrast,loss_vec_norm

    def on_train_epoch_start(self):
        if self.scheduler is not None and self.args.scheduler == 'cosine':
            self.scheduler.step()
    
    def on_train_step_start(self):
        if self.scheduler is not None and self.args.scheduler == 'frad_cosine':
            self.scheduler.step()

    def training_step(self, batch, batch_idx):
        if self.args.denoising and self.args.pos_mask:
            loss1,loss_contrast,loss_pos,loss_vec_norm = self(batch)
            loss = loss1 + loss_contrast + loss_pos*self.args.denoising_weight+loss_vec_norm 
            self.log('train_loss1', float(loss1), batch_size=self.args.batch_size)
            self.log('train_loss_contrast', float(loss_contrast), batch_size=self.args.batch_size)
            self.log('train_loss_pos', float(loss_pos), batch_size=self.args.batch_size)
            self.log('train_loss_vec_norm', float(loss_vec_norm), batch_size=self.args.batch_size)
            self.log('train_loss', float(loss), batch_size=self.args.batch_size)

        elif self.args.denoising and not self.args.pos_mask:
            loss_contrast,loss_pos,loss_vec_norm = self(batch)
            loss = loss_contrast + loss_pos+loss_vec_norm
            self.log('train_loss_contrast', float(loss_contrast), batch_size=self.args.batch_size)
            self.log('train_loss_pos', float(loss_pos), batch_size=self.args.batch_size)
            self.log('train_loss_vec_norm', float(loss_vec_norm), batch_size=self.args.batch_size)
            self.log('train_loss', float(loss), batch_size=self.args.batch_size)
        else:
            loss1,loss_contrast,loss_vec_norm = self(batch)
            loss = loss1 + loss_contrast+loss_vec_norm 
            self.log('train_loss1', float(loss1), batch_size=self.args.batch_size)
            self.log('train_loss_contrast', float(loss_contrast), batch_size=self.args.batch_size)
            self.log('train_loss_vec_norm', float(loss_vec_norm), batch_size=self.args.batch_size)
            self.log('train_loss', float(loss), batch_size=self.args.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.args.denoising and self.args.pos_mask:
            loss1,loss_contrast,loss_pos,loss_vec_norm = self(batch)
            loss = loss1 + loss_contrast + loss_pos*self.args.denoising_weight+loss_vec_norm 
            self.log('valid_loss1', float(loss1), batch_size=self.args.batch_size)
            self.log('valid_loss_contrast', float(loss_contrast), batch_size=self.args.batch_size)
            self.log('valid_loss_pos', float(loss_pos), batch_size=self.args.batch_size)
            self.log('valid_loss_vec_norm', float(loss_vec_norm), batch_size=self.args.batch_size)
            self.log('valid_loss', float(loss), batch_size=self.args.batch_size)

        elif self.args.denoising and not self.args.pos_mask:
            loss_contrast,loss_pos,loss_vec_norm = self(batch)
            loss = loss_contrast + loss_pos+loss_vec_norm
            self.log('valid_loss_contrast', float(loss_contrast), batch_size=self.args.batch_size)
            self.log('valid_loss_pos', float(loss_pos), batch_size=self.args.batch_size)
            self.log('valid_loss_vec_norm', float(loss_vec_norm), batch_size=self.args.batch_size)
            self.log('valid_loss', float(loss), batch_size=self.args.batch_size)
        else:
            loss1,loss_contrast,loss_vec_norm = self(batch)
            loss = loss1 + loss_contrast+loss_vec_norm 
            self.log('valid_loss1', float(loss1), batch_size=self.args.batch_size)
            self.log('valid_loss_contrast', float(loss_contrast), batch_size=self.args.batch_size)
            self.log('valid_loss_vec_norm', float(loss_vec_norm), batch_size=self.args.batch_size)
            self.log('valid_loss', float(loss), batch_size=self.args.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        if self.args.denoising and self.args.pos_mask:
            prediction,_,_,_,_ = self.model(batch)
            target = batch.mask_coord_label
        elif self.args.denoising and not self.args.pos_mask:
            prediction,_,_,_ = self.model(batch)
            target = batch.pos_target
        else:
            prediction,_,_,_ = self.model(batch)
            target = batch.mask_coord_label
        mae = float(np.mean(np.square(prediction.cpu().detach().numpy() - target.cpu().detach().numpy())))
        self.log('test_MAE', mae, batch_size=self.args.batch_size)
        return mae

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = self.args.warmup_steps
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        max_iters = self.args.max_epochs * len(self.trainer.train_dataloader) // self.args.accumulate_grad_batches
        assert max_iters > warmup_steps
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRSchedulerV2(optimizer, max_iters, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler in {'None', 'none'}:
            self.scheduler = None
        elif self.args.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(optimizer, self.args.max_epochs, eta_min = self.args.min_lr)
        elif self.args.scheduler == 'frad_cosine':
            self.scheduler = CosineAnnealingLR(optimizer, self.args.lr_cosine_length)
        else:
            raise NotImplementedError()
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.args.warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.args.warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.args.init_lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()
    

def main(args):
    L.seed_everything(args.seed,workers=True)
    dm = PCQM4MV2DM(args)
    dm.setup("fit")
    model = AutoEncoder(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks,
        prior_model=None,
        args=args
    )
    trainer_model = AutoEncoderTrainer(model, args)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/'),
        callbacks=custom_callbacks(args),
        check_val_every_n_epoch = args.check_val_every_n_epoch,
        detect_anomaly=args.detect_anomaly,
        limit_test_batches=1.0,  # Ensure all test batches are processed
    )

    if args.test_only:
        trainer.test(trainer_model, datamodule=dm)
    else:
        trainer.fit(trainer_model, datamodule=dm)
        trainer.test(trainer_model, datamodule=dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = PCQM4MV2DM.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--node_dim', type=int, default=63)
    parser.add_argument('--edge_dim', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_dim_2d', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_heads_2d', type=int, default=4)
    parser.add_argument('--encoder_blocks', type=int, default=8)
    parser.add_argument('--decoder_blocks', type=int, default=2)
    parser.add_argument('--disable_compile', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False, help='Only run the test using the last checkpoint')

    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=5000000)
    parser.add_argument('--save_every_n_epochs', type=int, default=50)
    parser.add_argument('--test_every_n_epochs', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-16)
    parser.add_argument('--scheduler', type=str, default="cosine")
    parser.add_argument('--init_lr', type=float, default=5e-5, help='optimizer init learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='optimizer min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--lr_factor', type=float, default=0.8)
    parser.add_argument('--lr_patience', type=int, default=15)
    parser.add_argument('--t0', type=int, default=100000)
    parser.add_argument('--tmult', type=int, default=2)
    parser.add_argument('--etamin', type=float, default=1e-7)

    parser.add_argument('--checkpoint_path', type=str, default="")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--pair_update', action='store_true', default=False)
    parser.add_argument('--trans_version', type=str, default='v6')
    parser.add_argument('--attn_activation', type=str, default='silu')
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='Mask ratio for the autoencoder')
    parser.add_argument('--dataset', type=str, default='pcqm4mv2', help='Dataset to use')
    parser.add_argument('--denoising_weight', type=float, default=0.1)
    parser.add_argument('--denoising', action='store_true', default=False)
    parser.add_argument('--lr_cosine_length', type=int, default=500000)
    parser.add_argument('--prior_model', action='store_true', default=False)
    parser.add_argument('--pos_mask', action='store_true', default=False)
    parser.add_argument('--delta', type=int, default=1000)

    args = parser.parse_args()
    main(args)
