# 3D-GSRD: 3D Molecular Graph Auto-Encoder with Selective Re-mask Decoding

## Requirements

You can create the environment for 3D-GSRD by running the following command in order:

```bash
conda create -n rep python=3.8
conda install pytorch==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install lightning rdkit
conda install -c conda-forge openbabel
```

## Pretraining

```bash
bash run_pretrain.sh
```

## Finetuning for QM9

The `checkpoint_path` argument should be set to the path of the actual pretrained model checkpoint. In this script, the subtask is set to `homo`, but it can be replaced with other subtasks such as `dipole_moment`, `isotropic_polarizability`, `lumo`, `gap`, `electronic_spatial_extent`, `zpve`, `energy_U0`, `energy_U`, `enthalpy_H`, `free_energy`, `heat_capacity`. Run the following script:

```bash
bash ft_qm9.sh
```

Please note that if the `dataset_arg` is set to `energy_U0`、`energy_U`、`enthalpy_H` and `free_energy`, add `--prior_model` in the script.

## Finetuning for MD17

The `checkpoint_path` argument should be set to the path of the actual pretrained model checkpoint. In this script, the subtask is set to `md17_aspirin`, but it can be replaced with other subtasks such as `md17_benzene2017`, `md17_ethanol`, `md17_malonaldehyde`, `md17_naphthalene`, `md17_salicylic`, `md17_toluene`, `md17_uracil`. Run the following script:

```bash
bash ft_md17.sh
```
