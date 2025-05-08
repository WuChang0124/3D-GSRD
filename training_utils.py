import torch
import lightning as L
from tqdm import tqdm


def load_encoder_params(model, ckpt_path):
    checkpoint = torch.load(ckpt_path,map_location='cpu')
    state_dict = checkpoint['state_dict']
    param_prefixes = ['model._orig_mod.encoder.node_embedding','model._orig_mod.encoder.edge_embedding','model._orig_mod.encoder.encoder_blocks','model._orig_mod.encoder.neigh_embedding','model._orig_mod.encoder.wf','model._orig_mod.encoder.node_proj','model._orig_mod.encoder.distance_expansion','model._orig_mod.encoder.out_norm_vec','model._orig_mod.encoder.norm1']
    selected_params = {k: v for k, v in state_dict.items() if any(k.startswith(prefix) for prefix in param_prefixes)}
    renamed_params = {k.replace('model._orig_mod.encoder.', ''): v for k, v in selected_params.items()}
    model.load_state_dict(renamed_params, strict=False)
    return model

class PeriodicTestCallback(L.Callback):
    def __init__(self, test_every_n_epochs):
        super().__init__()
        self.test_every_n_epochs = test_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.test_every_n_epochs == 0:
            self._run_test(trainer, pl_module)

    def _run_test(self, trainer, pl_module):
        training = pl_module.training

        pl_module.eval()

        test_loader = trainer.datamodule.test_dataloader()
        device = pl_module.device

        with torch.no_grad():
            pl_module.on_test_epoch_start()
            for batch_idx, batch in tqdm(enumerate(test_loader), desc="Testing: ", total=len(test_loader), leave=False):
                batch = batch.to(device)
                pl_module.test_step(batch, batch_idx)

            pl_module.on_test_epoch_end()

        pl_module.train(training)

class PeriodicPredictCallback(L.Callback):
    def __init__(self, generate_every_n_epochs):
        super().__init__()
        self.generate_every_n_epochs = generate_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.generate_every_n_epochs == 0:
            self._run_predict(trainer, pl_module)

    def _run_predict(self, trainer, pl_module):
        pl_module.eval()

        predict_loader = trainer.datamodule.predict_dataloader()
        device = pl_module.device

        with torch.no_grad():
            pl_module.on_predict_epoch_start()
            for batch_idx, batch in tqdm(enumerate(predict_loader), desc="Generating: ", total=len(predict_loader), leave=False):
                batch = batch.to(device)
                pl_module.predict_step(batch, batch_idx)

            pl_module.on_predict_epoch_end()

def custom_callbacks(args):
    callbacks = []
    callbacks.append(L.pytorch.callbacks.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/",
                                         filename='{epoch:02d}',
                                         every_n_epochs=args.save_every_n_epochs,
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))

    if args.test_every_n_epochs is not None:
        callbacks.append(PeriodicTestCallback(args.test_every_n_epochs))

    return callbacks

def suppress_warning():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

def device_cast(args_device: str):
    """
    Number of devices to train on (int), which devices to train on (list or str), or "auto".
    """
    try:
        if args_device == 'auto':
            devices = 'auto'
        elif args_device.startswith('[') and args_device.endswith(']'):
            devices = eval(args_device)
            assert isinstance(devices, list)
            assert all(isinstance(device, int) for device in devices)
        else:
            devices = int(args_device)
    except:
        raise NotImplementedError(f"devices should be a integer, a list (of integer), or 'auto', got {args_device}")

    return devices

def add_training_specific_args(parser):
    parser = parser.add_argument_group("Trainer")
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='auto')
    parser.add_argument('--precision', type=str, default='bf16-mixed')

    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    parser.add_argument('--cache_epoch', type=int, default=5)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--test_every_n_epochs', type=int, default=20)
    parser.add_argument('--generate_every_n_epochs', type=int, default=None)

    parser.add_argument('--disable_compile', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--ckpt_path', type=str, default=None)

def print_args(parser, args):
    LINE_LENGTH = 52
    print("=" * LINE_LENGTH)
    for group in parser._action_groups:
        if group.title not in ("positional arguments", "optional arguments"):
            title = group.title
            padding_length = LINE_LENGTH - len(title)
            left_padding = padding_length // 2
            right_padding = padding_length - left_padding

            print(f"{'-' * left_padding}{title}{'-' * right_padding}")

        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        for arg_name, arg_value in group_dict.items():
            if arg_name == "help":
                continue
            print(f"{arg_name.rjust(25)}: {arg_value}")

        # print("-" * LINE_LENGTH)

    print("=" * LINE_LENGTH)
