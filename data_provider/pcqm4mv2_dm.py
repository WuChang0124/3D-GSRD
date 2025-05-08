import argparse
from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import lightning as L
from data_provider.pcqm4mv2 import PCQM4MV2Dataset
from data_provider.utils import make_splits
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch

class PCQM4MV2Collater(object):
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
            if 'mask_coord_label' in data.keys():
                data['mask_coord_label'] = data.pos[data.pos_mask]
            if 'pos_target' in data.keys():
                data['pos_target'] = torch.from_numpy(rot_aug.apply(data['pos_target'].numpy())).to(dtype)

        if self.aug_translation:
            trans_aug = self.aug_translation_scale * torch.randn(bs, 3, dtype=dtype)
            data['pos'] = data['pos'] + trans_aug[data.batch]
            if 'mask_coord_label' in data.keys():
                data['mask_coord_label'] = data.pos[data.pos_mask]
        return data

    def __call__(self, data_list):
        ## graph batch
        data_batch = Batch.from_data_list(data_list)
        data_batch['max_seqlen'] = int((data_batch['ptr'][1:] - data_batch['ptr'][:-1]).max())
        data_batch = self.augmentation(data_batch)
        data_batch.x = data_batch.x.to(torch.float)
        data_batch.edge_attr = data_batch.edge_attr.to(torch.float)
        return data_batch


class PCQM4MV2DM(L.LightningDataModule):
    def __init__(self, args):
        super(PCQM4MV2DM, self).__init__()
        self._saved_dataloaders = dict()
        self.dataset = PCQM4MV2Dataset(args.root,args.denoising,mask_ratio=args.mask_ratio)
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

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

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

    def collate_fn(self, batch):
        return PCQM4MV2Collater(
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

