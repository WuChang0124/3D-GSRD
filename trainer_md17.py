import math
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 
from lightning.pytorch.loggers import CSVLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from model.retrans import RelaTransEncoder
import argparse  
from data_provider.md17_dm import MD17DM
from training_utils import custom_callbacks,load_encoder_params
import os

torch.cuda.empty_cache()


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

class EncoderTrainer(L.LightningModule):
    def __init__(self, model, args, mean=None, std=None):
        super().__init__()
        self.model = model
        self.args = args
        self.energy_and_force = args.energy_and_force
        self.criterion = nn.L1Loss()  # loss function
        self.save_hyperparameters(ignore='model')
        self.mean = torch.scalar_tensor(0 if mean is None else mean)
        self.std = torch.scalar_tensor(1 if std is None else std)
        self.val_loss = None
        self.ema = None 
        self._reset_ema_dict()

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}

    
    def forward(self, batch, stage = "train"):

        pos_ = batch.pos.clone()
        if self.energy_and_force:
            pos_.requires_grad_(True)
        
        # label value
        target_e = batch.y
        target_e = target_e.unsqueeze(-1) 
        target_f = batch.dy
        
        pre_e, vec = self.model(batch, batch.x, batch.edge_index, batch.edge_attr, pos_)
        
        # energy_loss
        _pre_e = pre_e * self.std + self.mean
        loss_y = self.criterion(_pre_e, target_e)  

        # ema
        if self.args.ema_alpha_y < 1:
            if self.ema[stage + "_y"] is None:
                self.ema[stage + "_y"] = loss_y.detach()
            # apply exponential smoothing over batches to y
            loss_y = (
                self.args.ema_alpha_y * loss_y
                + (1 - self.args.ema_alpha_y) * self.ema[stage + "_y"]
            )
            self.ema[stage + "_y"] = loss_y.detach()

        excess_norm = F.relu(vec.norm(dim=-1) - self.args.delta)
        loss_vec_norm = 0.01 * excess_norm.mean()  

        # force_loss energy and force join train
        if self.energy_and_force:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(_pre_e)]
            pre_f = -torch.autograd.grad(
                [_pre_e],
                [pos_],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            loss_dy = self.criterion(pre_f, target_f)
            # ema
            if self.args.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.ema_alpha_dy * loss_dy
                    + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()
            
            return loss_y*self.args.e_w + loss_dy*self.args.f_w, loss_vec_norm
        
        else:
            return loss_y, loss_vec_norm
        

    def on_train_epoch_start(self):
        if self.scheduler is not None and self.args.scheduler == 'cosine':
            self.scheduler.step()
    
    def on_validation_epoch_end(self):
        if self.scheduler is not None and self.args.scheduler == 'reduce_on_plateau':
            self.scheduler.step(self.val_loss)


    def training_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):
            loss_mae, loss_vec_norm = self(batch, "train")
            loss = loss_mae + 0.01*loss_vec_norm
            self.log('train_loss', float(loss), batch_size=self.args.batch_size)
            self.log('train_loss_mae', float(loss_mae), batch_size=self.args.batch_size)
            self.log('train_loss_vec', float(loss_vec_norm), batch_size=self.args.batch_size)
            
            return loss

            
    def validation_step(self, batch, batch_idx): 
        with torch.set_grad_enabled(True):
            loss_mae, loss_vec_norm = self(batch, "val")
            loss = loss_mae + 0.01*loss_vec_norm
            self.val_loss = loss
            self.log('valid_loss', float(loss), batch_size=self.args.batch_size)
            self.log('val_loss_mae', float(loss_mae), batch_size=self.args.batch_size)
            self.log('train_loss_vec', float(loss_vec_norm), batch_size=self.args.batch_size)
            return loss

    def test_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):
            pos_ = batch.pos.clone()

            if self.energy_and_force:
                pos_.requires_grad_(True)

            target_e = batch.y
            target_f = batch.dy

            pre_e, vec = self.model(batch, batch.x, batch.edge_index, batch.edge_attr, pos_)

            _pre_e = pre_e * self.std + self.mean
            # e_mae
            if isinstance(target_e, torch.Tensor):
                e_mae = torch.mean(torch.abs(_pre_e - target_e)).cpu().item()
            else:
                e_mae = float(np.mean(np.absolute(_pre_e - target_e)))

            if self.energy_and_force:
                pre_f = -torch.autograd.grad(
                    outputs=_pre_e,
                    inputs=pos_,
                    grad_outputs=torch.ones_like(_pre_e),
                    create_graph=True,
                    retain_graph=True
                )[0]
                # f_mae
                if isinstance(target_f, torch.Tensor):
                    f_mae = torch.mean(torch.abs(pre_f - target_f)).cpu().item()
                else:
                    f_mae = float(np.mean(np.absolute(pre_f - target_f)))
    
                self.log('test_e_MAE', e_mae, batch_size=self.args.batch_size)
                self.log('test_f_MAE', f_mae, batch_size=self.args.batch_size)
                return e_mae, f_mae
        self.log('test_e_MAE', e_mae, batch_size=self.args.batch_size)
        return e_mae

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
        elif self.args.scheduler == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(optimizer, "min", factor=self.args.lr_factor, patience=self.args.lr_patience, min_lr=self.args.min_lr)
        elif self.args.scheduler == 'warmrestarts':
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.tmult, eta_min=args.etamin)
        else:
            raise NotImplementedError()
        return optimizer


def main(args):
    L.seed_everything(args.seed,workers=True)
    dm = MD17DM(args)
    train_dataloader = dm.train_dataloader()
    mean, std = dm.get_mean_std(train_dataloader) # get mean and std

    model = RelaTransEncoder(
    node_dim=args.node_dim,
    edge_dim=args.edge_dim,
    hidden_dim=args.hidden_dim,
    n_heads=args.n_heads,
    n_blocks=args.encoder_blocks,
    prior_model = None,
    args=args
    )
    
    trainer_model = EncoderTrainer(model, args, mean, std)
    
    if args.checkpoint_path:
        trainer_model.model = load_encoder_params(trainer_model.model, args.checkpoint_path)

    trainer_model.model = torch.compile(trainer_model.model, dynamic=True, fullgraph=False, disable=args.disable_compile)

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
        limit_test_batches=1.0  # Ensure all test batches are processed
    )

    if args.test_only:
        trainer.test(trainer_model, datamodule=dm, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(trainer_model, datamodule=dm)
        trainer.test(trainer_model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = MD17DM.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=0)  
    parser.add_argument('--filename', type=str, default='cosine')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--node_dim', type=int, default=63)
    parser.add_argument('--edge_dim', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_dim_2d', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_heads_2d', type=int, default=4)
    parser.add_argument('--encoder_blocks', type=int, default=12)
    parser.add_argument('--decoder_blocks', type=int, default=2)
    parser.add_argument('--disable_compile', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False, help='Only run the test using the last checkpoint')

    parser.add_argument('--max_epochs', type=int, default=3000)
    parser.add_argument('--max_steps', type=int, default=500000)
    parser.add_argument('--save_every_n_epochs', type=int, default=50)
    parser.add_argument('--test_every_n_epochs', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.0)  #default=1e-16
    parser.add_argument('--scheduler', type=str, default="cosine")
    parser.add_argument('--init_lr', type=float, default=5e-4, help='optimizer init learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='optimizer min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
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
    parser.add_argument('--pos_mask', action='store_true', default=False)
    parser.add_argument('--lr_cosine_length', type=int, default=500000)
    parser.add_argument('--prior_model', action='store_true', default=False)

    parser.add_argument('--e_w', type=float, default=0.2)
    parser.add_argument('--f_w', type=float, default=0.8)
    parser.add_argument('--energy_and_force', action='store_true', default=True)
    parser.add_argument('--ema_alpha_dy', type=float, default=1.0)
    parser.add_argument('--ema_alpha_y', type=float, default=0.05)  
    parser.add_argument('--lr_factor', type=float, default=0.8) # schedule-reduce_on_plateau
    parser.add_argument('--lr_patience', type=int, default=30)
    parser.add_argument('--delta', type=int, default=1000)

    args = parser.parse_args()

    main(args)