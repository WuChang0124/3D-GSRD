from itertools import product
import random
import os
import torch
from tqdm import tqdm
from rdkit import Chem
from typing import List
from data_provider.featurization import featurize_mol
from torch_geometric.data import (InMemoryDataset, Data)

MOL_LST = None



class PCQM4MV2(InMemoryDataset):
    def __init__(
        self,
        root: str,
        mask_ratio: float = 0.15
    ) -> None:
        super().__init__(root)
        self.mask_ratio = mask_ratio
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['pcqm4m-v2-train.sdf']  

    @property
    def processed_file_names(self) -> List[str]:
        return ['pcqm4mv2.pt']

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    

    def process(self) -> None:
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)
        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            data = featurize_mol(mol, 'merge')
            assert ((data.x == 0.0) | (data.x == 1.0)).all()
            assert ((data.edge_attr == 0.0) | (data.edge_attr == 1.0)).all()
            data.x = data.x.bool()
            data.edge_attr = data.edge_attr.bool()
            data.edge_index = data.edge_index.to(torch.int16)

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            data['idx']= i
            data['pos'] = pos
            data['pos'] -= data['pos'].mean(dim=0, keepdim=True)

            data_list.append(data)
        self.save(data_list, self.processed_paths[0])

    def compute_pos_std(self) -> float:
        all_pos = []
        for i in tqdm(range(len(self))):
            data = self.get(i)  
            pos = data.pos  # [num_atoms, 3]
            all_pos.append(pos)
        
        all_pos = torch.cat(all_pos, dim=0)  # [total_atoms, 3]
        std = torch.std(all_pos).item()
        print(f"{std}")
        return std

def transform(data):
    noise = torch.randn_like(data.pos)
    data.pos_target = noise
    if 'pos_mask' in data.keys():
        data.pos[~data.pos_mask] = data.pos[~data.pos_mask] + noise[~data.pos_mask] * 0.04
    else:
        data.pos = data.pos + noise * 0.04
    return data  

class PCQM4MV2Dataset(PCQM4MV2):
    def __init__(self, root: str, denoising,mask_ratio: float = 0.15) -> None:
        super().__init__(root)
        self.mask_ratio = mask_ratio
        self.denoising = denoising
        
    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, idx: int) -> Data:
        data = super().__getitem__(idx)

        data['edge_index'] = data.edge_index.to(torch.long)
        data['edge_attr'] = data.edge_attr.float()
        full_edge_index, full_edge_attr = get_full_edge(data.x, data.edge_index, data.edge_attr)
        data['edge_index'] = full_edge_index
        data['edge_attr'] = full_edge_attr
        data['x'] = data.x.float()
        num_atoms = data.z.size(0)
        sample_size = max(1, int(num_atoms * self.mask_ratio + 1))
        masked_atom_indices = random.sample(range(num_atoms), sample_size)
        masked_atom_indices = torch.tensor(masked_atom_indices, dtype=torch.long)
        
        pos_mask = torch.zeros(data.pos.shape[0], dtype=torch.bool)
        pos_mask[masked_atom_indices] = True

        data.pos_mask = pos_mask
        data.mask_coord_label = data.pos[pos_mask] # [num_masked_atoms, 3]
        if self.denoising:
            data = transform(data)
        return data
    

def get_full_edge(x, edge_index, edge_attr):
    num_atom = x.shape[0]
    indices = list(product(range(num_atom), range(num_atom)))
    indices.sort(key=lambda x: (x[0], x[1]))
    full_edge_index = torch.LongTensor(indices).t() # [2, N_edge]
    full_edge_attr = torch.zeros((num_atom, num_atom, edge_attr.size(1)), dtype=edge_attr.dtype) # [num_atom, num_atom, edge_nf]
    full_edge_attr[edge_index[0], edge_index[1]] = edge_attr
    full_edge_attr = torch.cat([full_edge_attr, torch.eye(num_atom, dtype=full_edge_attr.dtype).unsqueeze(-1)], dim=-1) # [num_atom, num_atom, edge_nf + 1], add self-loop edge_attr
    full_edge_attr = full_edge_attr.reshape(-1, full_edge_attr.size(-1)) # [N_edge, edge_nf + 1]
    return full_edge_index, full_edge_attr


