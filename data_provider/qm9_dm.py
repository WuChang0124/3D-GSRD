import argparse
from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import lightning as L
from pytorch_lightning.utilities import rank_zero_warn
from data_provider.qm9_dataset import QM9Dataset
from data_provider.utils import make_splits, MissingEnergyException
from torch_scatter import scatter
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch

class QM9Collater(object):
    def __init__(self, aug_rotation=True, aug_translation=True, aug_translation_scale=0.01):
        self.aug_rotation = aug_rotation
        self.aug_translation = aug_translation
        self.aug_translation_scale = aug_translation_scale

    def augmentation(self, data):
        bs = len(data['ptr']) - 1
        dtype = torch.float
        if self.aug_rotation:
            rot_aug = Rotation.random(bs)
            rot_aug = rot_aug[data.batch.numpy()]
            data['pos'] = torch.from_numpy(rot_aug.apply(data['pos'].numpy())).to(dtype)
            if 'pos_target' in data.keys():
                data['pos_target'] = torch.from_numpy(rot_aug.apply(data['pos_target'].numpy())).to(dtype)

        if self.aug_translation:
            trans_aug = self.aug_translation_scale * torch.randn(bs, 3, dtype=dtype)
            data['pos'] = data['pos'] + trans_aug[data.batch]

        return data

    def __call__(self, data_list):
        ## graph batch
        data_batch = Batch.from_data_list(data_list)
        data_batch['max_seqlen'] = int((data_batch['ptr'][1:] - data_batch['ptr'][:-1]).max())
        data_batch = self.augmentation(data_batch)
        data_batch.x = data_batch.x.to(torch.float)
        data_batch.edge_attr = data_batch.edge_attr.to(torch.float)
        return data_batch   

class QM9DM(L.LightningDataModule):
    def __init__(self, args):
        super(QM9DM, self).__init__()
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        dataset_factory = lambda t: QM9Dataset(args.root,dataset_arg=args.dataset_arg,transform=t)
        self.dataset = dataset_factory(None)
        self.args=args

    def setup(self, stage):
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.args.train_size,
            self.args.val_size,
            self.args.test_size,
            self.args.seed,
            join(self.args.root, "splits.npz"),
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )

        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

        self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def pos_std(self):
        return self._pos_std
    
    def _get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.args.batch_size
            shuffle = True
            pin_memory = True
            drop_last = True
        elif stage in ["val", "test"]:
            batch_size = self.args.inference_batch_size
            shuffle = False
            pin_memory = False
            drop_last = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn
        )

        return dl

    def get_energy_data(self, data):
        if data.y is None:
            raise MissingEnergyException()

        # remove atomref energies from the target energy
        atomref_energy = self.atomref.squeeze()[data.z].sum()
        return (data.y.squeeze() - atomref_energy).unsqueeze(dim=0).unsqueeze(dim=1)


    def _standardize(self):
        def get_energy(batch, atomref):
            if batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val"),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            #atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            atomref = None
            # extract energies from the data
            # atomref = self.atomref
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
            pos = torch.cat([batch.pos for batch in data],dim=0)
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        self._mean = torch.mean(ys).item()
        self._std = torch.std(ys).item()
        print(ys)
        print(self._mean,self._std)
    def collate_fn(self, batch):
        return QM9Collater(
            aug_rotation=True,
            aug_translation=self.args.aug_translation,
            aug_translation_scale=self.args.aug_translation_scale,
        )(batch)
    
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--root', type=str, default='./data', help='Root directory for dataset')
        parser.add_argument('--dataset_arg', type=str, default='homo', help='Property to predict')
        parser.add_argument('--train_size', type=int, default=100000, help='Size of the training set')
        parser.add_argument('--val_size', type=int, default=17748, help='Size of the validation set')
        parser.add_argument('--test_size', type=int, default=13083, help='Size of the test set')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--inference_batch_size', type=int, default=32, help='Batch size for inference')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        parser.add_argument('--aug_translation', action='store_true', default=False, help='Whether to apply translation augmentation')
        parser.add_argument('--aug_translation_scale', type=float, default=0.1, help='Translation scale for augmentation')
        return parent_parser
