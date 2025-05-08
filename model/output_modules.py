from typing import Optional
import torch
from torch import nn
from abc import ABCMeta, abstractmethod
import numpy as np
from torch_scatter import scatter_sum

atomic_masses = np.array([
    1.0, 1.008, 4.002602, 6.94, 9.0121831,
    10.81, 12.011, 14.007, 15.999, 18.998403163,
    20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983,
    40.078, 44.955908, 47.867, 50.9415, 51.9961,
    54.938044, 55.845, 58.933194, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.921595, 78.971,
    79.904, 83.798, 85.4678, 87.62, 88.90584,
    91.224, 92.90637, 95.95, 97.90721, 101.07,
    102.9055, 106.42, 107.8682, 112.414, 114.818,
    118.71, 121.76, 127.6, 126.90447, 131.293,
    132.90545196, 137.327, 138.90547, 140.116, 140.90766,
    144.242, 144.91276, 150.36, 151.964, 157.25,
    158.92535, 162.5, 164.93033, 167.259, 168.93422,
    173.054, 174.9668, 178.49, 180.94788, 183.84,
    186.207, 190.23, 192.217, 195.084, 196.966569,
    200.592, 204.38, 207.2, 208.9804, 208.98243,
    209.98715, 222.01758, 223.01974, 226.02541, 227.02775,
    232.0377, 231.03588, 238.02891, 237.04817, 244.06421,
    243.06138, 247.07035, 247.07031, 251.07959, 252.083,
    257.09511, 258.09843, 259.101, 262.11, 267.122,
    268.126, 271.134, 270.133, 269.1338, 278.156,
    281.165, 281.166, 285.177, 286.182, 289.19,
    289.194, 293.204, 293.208, 294.214,
])
  
    
class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(OutputModel, self).__init__()

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    def __init__(self, hidden_dim):
        super(Scalar, self).__init__()
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class EquivariantScalar(OutputModel):
    def __init__(self, hidden_dim):
        super(EquivariantScalar, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_dim,
                    hidden_dim // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_dim // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0


class DipoleMoment(Scalar):
    def __init__(self, hidden_dim):
        super(DipoleMoment, self).__init__(
            hidden_dim
        )
        atomic_mass = torch.from_numpy(atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter_sum(mass * pos, batch, dim=0) / scatter_sum(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class EquivariantDipoleMoment(EquivariantScalar):
    def __init__(self, hidden_dim):
        super(EquivariantDipoleMoment, self).__init__(
            hidden_dim
        )
        atomic_mass = torch.from_numpy(atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter_sum(mass * pos, batch, dim=0) / scatter_sum(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x + v.squeeze()

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class ElectronicSpatialExtent(OutputModel):
    def __init__(self, hidden_dim):
        super(ElectronicSpatialExtent, self).__init__()
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        atomic_mass = torch.from_numpy(atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter_sum(mass * pos, batch, dim=0) / scatter_sum(mass, batch, dim=0)

        x = torch.norm(pos - c[batch], dim=1, keepdim=True) ** 2 * x
        return x


class EquivariantElectronicSpatialExtent(ElectronicSpatialExtent):
    pass


class EquivariantVectorOutput(EquivariantScalar):
    def __init__(self, hidden_dim):
        super(EquivariantVectorOutput, self).__init__(
            hidden_dim
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_dim,
        out_channels,
        intermediate_channels=None,
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_dim

        self.vec1_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vec2_proj = nn.Linear(hidden_dim, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.SiLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v