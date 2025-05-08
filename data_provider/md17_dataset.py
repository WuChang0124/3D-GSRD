import os.path as osp
import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, download_url

from rdkit import Chem
from itertools import product
from openbabel import openbabel, pybel

from data_provider.featurization import featurize_mol


data_smiles = {
    'md17_aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
    'md17_benzene2017': 'C1=CC=CC=C1', 
    'md17_ethanol': 'CCO', 
    'md17_malonaldehyde': 'C(C=O)C=O', 
    'md17_naphthalene': 'C1=CC=C2C=CC=CC2=C1', 
    'md17_salicylic': 'C1=CC=C(C(=C1)C(=O)O)O', 
    'md17_toluene': 'CC1=CC=CC=C1', 
    'md17_uracil': 'C1=CNC(=O)NC1=O'
}


def correct_mol(mol, right_mol):
    mol_atoms = mol.GetAtoms()
    right_mol_atoms = right_mol.GetAtoms()
    mol_bonds = mol.GetBonds()
    right_mol_bonds = right_mol.GetBonds()
    corrected_mol = Chem.RWMol(mol)

    for bond in right_mol_bonds:
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        mol_bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)

        if mol_bond is None or mol_bond.GetBondType() != bond_type:
            if mol_bond is not None:
                corrected_mol.RemoveBond(atom1_idx, atom2_idx)  
            corrected_mol.AddBond(atom1_idx, atom2_idx, bond_type)  

    for bond in mol_bonds:
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        if right_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is None:
            corrected_mol.RemoveBond(atom1_idx, atom2_idx)

    return corrected_mol.GetMol()

# openbabel
def construct_mol(atoms, coordinates, title=None):
    mol = openbabel.OBMol()
    for atom, (x, y, z) in zip(atoms, coordinates):
        ob_atom = mol.NewAtom()
        ob_atom.SetAtomicNum(atom.item())
        ob_atom.SetVector(x.item(), y.item(), z.item())  # item()--将tensor数据转变成int数值
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    if title:
        mol.SetTitle(title)
    return mol

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


class MD17(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`MD17` dataset 
        which is from `"Machine learning of accurate energy-conserving molecular force fields" <https://advances.sciencemag.org/content/3/5/e1603015.short>`_ paper. 
        MD17 is a collection of eight molecular dynamics simulations for small organic molecules. 
    
        Args:
            root (string): The dataset folder will be located at root/name.
            name (string): The name of dataset. Available dataset names are as follows: :obj:`aspirin`, :obj:`benzene_old`, :obj:`ethanol`, :obj:`malonaldehyde`, 
                :obj:`naphthalene`, :obj:`salicylic`, :obj:`toluene`, :obj:`uracil`. (default: :obj:`benzene_old`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        >>> dataset = MD17(name='aspirin')
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(batch=[672], force=[672, 3], pos=[672, 3], ptr=[33], y=[32], z=[672])

        Where the attributes of the output data indicates:
    
        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The property (energy) for the graph (molecule).
        * :obj:`force`: The 3D force for atoms.
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs

    """
    def __init__(self, args, transform = None, pre_transform = None, pre_filter = None):

        self.root = args.root
        self.name = args.dataset_arg
        self.args = args
        self.folder = osp.join(self.root, self.name)
        self.url = 'http://quantum-machine.org/gdml/data/npz/' + self.name + '.npz'
        super(MD17, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return self.name + '.npz'

    @property
    def processed_file_names(self):
        # processed_paths[0] [1] [2]
        return [self.name + '_pyg.pt', self.name + '.sdf', self.name + '_right.sdf']

    def download(self):
        if osp.exists(osp.join(self.raw_dir, self.raw_file_names)):
            print(f"File {self.raw_file_names} already exists, skipping download.")
            return
        print(f"Downloading {self.url}...")
        download_url(self.url, self.raw_dir)


    def process(self):
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))
        E = data['E']  # energy
        F = data['F']  # force
        R = data['R']  # pos
        z = data['z']  # atom type

        sdf_file_path = self.processed_paths[1]

        if not os.path.exists(sdf_file_path):
            babel_mol_list = []  # mol
            for i in tqdm(range(len(E))):
                R_i = torch.tensor(R[i],dtype=torch.float32)
                z_i = torch.tensor(z,dtype=torch.int64)
                # featurize
                babel_mol = construct_mol(z_i, R_i)
                babel_mol_list.append(babel_mol)
            output = pybel.Outputfile("sdf", self.processed_paths[1], overwrite=True)
            for mol in babel_mol_list:
                mol = pybel.Molecule(mol)
                output.write(mol)
            output.close()

        suppl = Chem.SDMolSupplier(self.processed_paths[1], removeHs=False, sanitize=False)

        data_list = []

        r_mol = Chem.MolFromSmiles(data_smiles[self.name])
        r_mol = Chem.AddHs(r_mol)


        for new_id, mol in enumerate(tqdm(suppl)):
            try:
                Chem.SanitizeMol(mol)
            except:
                continue
            sdf_right_mol_path = self.processed_paths[2]
            if Chem.MolToSmiles(mol, isomericSmiles=True) != Chem.MolToSmiles(r_mol, isomericSmiles=True):
                if not os.path.exists(sdf_right_mol_path):
                    continue
                else:
                    right_suppl= Chem.SDMolSupplier(self.processed_paths[2], removeHs=False, sanitize=False)
                    right_mol = right_suppl[0]
                    mol = correct_mol(mol, right_mol)
                    try:
                        Chem.SanitizeMol(mol)
                    except:
                        continue
            else:
                if not os.path.exists(sdf_right_mol_path):
                    # right_sdf
                    writer = Chem.SDWriter(sdf_right_mol_path)
                    writer.write(mol)  # write right mol to .sdf
                    writer.close()

            # featurize
            R_i = torch.tensor(R[new_id],dtype=torch.float32)
            z_i = torch.tensor(z,dtype=torch.int64)
            E_i = torch.tensor(E[new_id],dtype=torch.float32)
            F_i = torch.tensor(F[new_id],dtype=torch.float32)

            data = featurize_mol(mol, 'merge')
            full_edge_index, full_edge_attr = get_full_edge(data.x, data.edge_index, data.edge_attr)
            data['edge_index'] = full_edge_index
            data['edge_attr'] = full_edge_attr
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(R_i, dtype=torch.float)

            data['idx']=new_id
            data['pos'] = pos
            data['pos'] -= data['pos'].mean(dim=0, keepdim=True)

            data['y'] = torch.tensor(E_i)
            data['dy'] = F_i

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        data, slices = self.collate(data_list)
        print('Saving to:', self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])



      

