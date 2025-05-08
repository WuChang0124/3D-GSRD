import argparse
import torch
from torch_geometric.data import DataLoader
import lightning as L
from data_provider.md17_dataset import MD17
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch
from data_provider.utils import make_splits
from torch.utils.data import Subset

class MD17Collater(object):
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


class MD17DM(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        dataset_factory = lambda t: MD17(args, transform=t)
        self.dataset = dataset_factory(None)
        self.args = args
        print("len of dataset:", len(self.dataset))

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.args.train_size,
            self.args.val_size,
            len(self.dataset) - self.args.train_size - self.args.val_size,
            self.args.seed
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.valid_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)
    

    def get_mean_std(self, dataloader):
        ys = torch.cat([batch.y.squeeze() for batch in dataloader])
        return ys.mean(), ys.std()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=False,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=False,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=False,
            drop_last=False
        )

    def collate_fn(self, batch):
        return MD17Collater(
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
 