from itertools import product
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
from typing import Tuple, Optional
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_sum, scatter
from model.output_modules import EquivariantDipoleMoment, EquivariantElectronicSpatialExtent, Scalar
            
def coord2dist(x, edge_index):
    # coordinates to distance
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    return radial

def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class EquivariantLayerNorm(nn.Module):
    r"""Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None) # Without bias term to preserve equivariance!

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.

        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = (
            s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        )
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(
            -2, -1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64) # Need double precision for accurate inversion.
        input = self.mean_center(input)
        # We use different diagonal elements in case input matrix is approximately zero,
        # in which case all singular values are equal which is problematic for backprop.
        # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(
            self.weight.dtype
        ) * self.weight.reshape(1, 1, self.normalized_shape[0])

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, "
            "elementwise_linear={elementwise_linear}".format(**self.__dict__)
        )


class GaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features"""
    def __init__(self, K, *args, **kwargs):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, *args, **kwargs):
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)

class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, use_mask=False):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.use_mask = use_mask
        if self.use_mask:
            self.mask_token = nn.Parameter(torch.ones(1))

    def forward(self, distances, edge_mask=None):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            if self.use_mask and edge_mask is not None:
                cutoffs[edge_mask] = self.mask_token
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            if self.use_mask and edge_mask is not None:
                cutoffs[edge_mask] = self.mask_token
            return cutoffs
        
class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True, use_mask=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)
        
        self.use_mask = use_mask
        if use_mask:
            self.mask_token = nn.Parameter(torch.zeros(1, num_rbf))

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist, edge_mask=None):
        dist = dist.unsqueeze(-1)
        out = self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
        if edge_mask is not None:
            out[edge_mask] = self.mask_token
        return out

class DMTBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, node_dim, edge_dim, time_dim, num_heads, 
                 cond_time=True, mlp_ratio=4, act=nn.GELU, dropout=0.0, pair_update=True, trans_version='v3', attn_activation='softmax',dataset='qm9'):
        super().__init__()
        self.dropout = dropout
        self.act = act()
        self.cond_time = cond_time
        self.pair_update = pair_update
        
        if not self.pair_update:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_dim, edge_dim * 2), 
                nn.GELU(), 
                nn.Linear(edge_dim * 2, edge_dim),
                nn.LayerNorm(edge_dim),
            )

        self.trans_version = trans_version
        if trans_version == 'v3':
            self.attn_mpnn = TransLayerOptimV3(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout)
        elif trans_version == 'v4':
            self.attn_mpnn = TransLayerOptimV4(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout)
        elif trans_version == 'v5':
            self.attn_mpnn = TransLayerOptimV5(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout, attn_activation=attn_activation,dataset=dataset)
        elif trans_version == 'v6':
            self.attn_mpnn = TransLayerOptimV6(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout, attn_activation=attn_activation,dataset=dataset)
        else:
            raise ValueError(f"Invalid transformer version")

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)
        
        if pair_update:
            self.node2edge_lin = nn.Linear(node_dim * 2 + edge_dim, edge_dim)
            # Feed forward block -> edge.
            self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
            self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)
        
        # equivariant edge update layer
        if self.cond_time:
            self.node_time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, node_dim * 6)
            )
            # Normalization for MPNN
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)

            if self.pair_update:
                self.edge_time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, edge_dim * 6)
                )
                self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
                self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            if self.pair_update:
                self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)
                self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)
        
        self.scale1 = nn.Parameter(torch.ones(1, node_dim) * 1e-2, requires_grad=True)
        self.scale2 = nn.Parameter(torch.ones(1, 3, node_dim) * 1e-2, requires_grad=True)

    def _ff_block_node(self, x):
        x = F.dropout(self.act(self.ff_linear1(x)), p=self.dropout, training=self.training)
        return F.dropout(self.ff_linear2(x), p=self.dropout, training=self.training)

    def _ff_block_edge(self, x):
        x = F.dropout(self.act(self.ff_linear3(x)), p=self.dropout, training=self.training)
        return F.dropout(self.ff_linear4(x), p=self.dropout, training=self.training)

    def forward(self, h, edge_attr, edge_index, node_time_emb=None, edge_time_emb=None, dist=None, dist_emb=None, edge_vec=None, edge_mask=None,vec=None):
        """
        A more optimized version of forward_old using torch.compile
        Params:
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
        """
        h_in_node = h
        h_in_edge = edge_attr

        if self.cond_time:
            ## prepare node features
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            
            ## prepare edge features
            if self.pair_update:
                edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                    self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)
                edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)
            else:
                edge_attr = self.edge_emb(edge_attr)
            
            # apply transformer-based message passing, update node features and edge features (FFN + norm)
            h_node = self.attn_mpnn(h, edge_index, edge_attr, edge_mask=edge_mask)

            ## update node features
            h_out = self.node_update(h_in_node, h_node, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp)
            
            ## update edge features
            if self.pair_update:
                # h_edge = torch.cat([h_node[edge_index[0]], h_node[edge_index[1]]], dim=-1)
                h_edge = h_node[edge_index.transpose(0, 1)].flatten(1, 2) # shape [N_edge, 2 * edge_hid_dim]
                h_edge = torch.cat([h_edge, h_in_edge], dim=-1)
                h_edge_out = self.edge_update(h_in_edge, h_edge, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp)
            else:
                h_edge_out = h_in_edge
        else:
            ## prepare node features
            h = self.norm1_node(h)

            ## prepare edge features
            if self.pair_update:
                edge_attr = self.norm1_edge(edge_attr)
            else:
                edge_attr = self.edge_emb(edge_attr)

            # apply transformer-based message passing, update node features and edge features (FFN + norm)
            if self.trans_version == 'v5':
                h_node = self.attn_mpnn(h, edge_index, edge_attr, dist=dist, dist_emb=dist_emb, edge_vec=edge_vec)
            elif self.trans_version == 'v6':
                h_node,dvec = self.attn_mpnn(h, edge_index, edge_attr, dist=dist, dist_emb=dist_emb, edge_vec=edge_vec,vec=vec)
            else:
                h_node = self.attn_mpnn(h, edge_index, edge_attr)
            
            ## update edge features
            if self.pair_update:
                # h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
                h_edge = h_node[edge_index.transpose(0, 1)].flatten(1, 2) # shape [N_edge, 2 * edge_hid_dim]
                h_edge = torch.cat([h_edge, h_in_edge], dim=-1)
                h_edge_out = self.edge_update(h_in_edge, h_edge)
            else:
                h_edge_out = h_in_edge
        if self.trans_version == 'v6':
            h_node = h_node * self.scale1
            dvec = dvec * self.scale2
            return h_node, dvec
        else:
            h_out = self.node_update(h_in_node, h_node)
            return h_out, h_edge_out

    def node_update(self, h_in_node, h_node, node_gate_msa=None, node_shift_mlp=None, node_scale_mlp=None, node_gate_mlp=None):
        h_node = h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        _h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) if self.cond_time else \
                self.norm2_node(h_node)
        h_out = h_node + node_gate_mlp * self._ff_block_node(_h_node) if self.cond_time else \
                h_node + self._ff_block_node(_h_node)
        return h_out
    
    def edge_update(self, h_in_edge, h_edge, edge_gate_msa=None, edge_shift_mlp=None, edge_scale_mlp=None, edge_gate_mlp=None):
        h_edge = self.node2edge_lin(h_edge)
        h_edge = h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        _h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
                self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(_h_edge) if self.cond_time else \
                    h_edge + self._ff_block_edge(_h_edge)
        return h_edge_out
    
class TransLayerOptimV3(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransLayerOptimV3, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_q = nn.Linear(in_channels + edge_dim, heads * out_channels, bias=bias)
        self.lin_kv = nn.Linear(in_channels + edge_dim, heads * out_channels * 2, bias=bias)
        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_kv.reset_parameters()
        self.proj.reset_parameters()
    
    
    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                edge_mask: OptTensor = None
                ) -> Tensor:
        """"""
        x_feat = x

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, x_feat=x_feat, edge_attr=edge_attr)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    def message(self, x_feat_i: Tensor, x_feat_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:
        query_ij = self.lin_q(torch.cat([x_feat_i, edge_attr], dim=-1)).view(-1, self.heads, self.out_channels)
        edge_key_ij, edge_value_ij = self.lin_kv(torch.cat([x_feat_j, edge_attr], dim=-1)).view(-1, self.heads, 2, self.out_channels).unbind(dim=2) # shape [N * N, heads, out_channels]

        alpha_ij = (query_ij * edge_key_ij).sum(dim=-1) / math.sqrt(self.out_channels) # shape [N * N, heads]
        alpha_ij = softmax(alpha_ij, index, ptr, size_i) 
        alpha_ij = F.dropout(alpha_ij, p=self.dropout, training=self.training)

        # node feature message
        msg = edge_value_ij * alpha_ij.view(-1, self.heads, 1) # shape [N * N, heads, out_channels]
        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class TransLayerOptimV4(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransLayerOptimV4, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_kv = nn.Linear(in_channels + edge_dim, heads * out_channels * 2, bias=bias)
        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_kv.reset_parameters()
        self.proj.reset_parameters()
    
    
    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                edge_mask: OptTensor = None
                ) -> Tensor:
        """"""
        x_feat = x

        query = self.lin_q(x_feat)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, x_feat=x_feat, edge_attr=edge_attr)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    def message(self, query_i: Tensor, x_feat_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:
        query_i = query_i.view(-1, self.heads, self.out_channels)
        edge_key_ij, edge_value_ij = self.lin_kv(torch.cat([x_feat_j, edge_attr], dim=-1)).view(-1, self.heads, 2, self.out_channels).unbind(dim=2) # shape [N * N, heads, out_channels]

        alpha_ij = (query_i * edge_key_ij).sum(dim=-1) / math.sqrt(self.out_channels) # shape [N * N, heads]
        alpha_ij = softmax(alpha_ij, index, ptr, size_i) 
        alpha_ij = F.dropout(alpha_ij, p=self.dropout, training=self.training)

        # node feature message
        msg = edge_value_ij * alpha_ij.view(-1, self.heads, 1) # shape [N * N, heads, out_channels]
        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



class TransLayerOptimV5(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, attn_activation='silu', dataset='qm9',**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransLayerOptimV5, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.dataset = dataset

        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=bias) #[hidden_dim,hidden_dim]
        self.lin_kv = nn.Linear(in_channels + edge_dim, heads * out_channels * 2, bias=bias) #[hidden_dim+edge_dim,hidden_dim*2]
        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias) #[hidden_dim,hidden_dim]
        self.lin_dkv = nn.Sequential(
            nn.Linear(in_channels // 4, heads * out_channels * 2, bias=bias),
            nn.SiLU()
        ) #[hidden_dim//4,hidden_dim*2]
        
        self.attn_activation = attn_activation
        
        self.cutoff = CosineCutoff(0, 5, use_mask=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_kv.reset_parameters()
        self.proj.reset_parameters()
    
    
    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                dist: OptTensor = None,
                dist_emb: OptTensor = None,
                edge_vec: OptTensor = None,
                edge_mask: OptTensor = None
                ) -> Tensor:
        """"""
        x_feat = x #[num_nodes,hidden_dim]

        query = self.lin_q(x_feat) #[num_nodes,hidden_dim]
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, x_feat=x_feat, edge_attr=edge_attr, dist_emb=dist_emb, dist=dist, edge_mask=edge_mask)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    def message(self, query_i: Tensor, x_feat_j: Tensor,
                edge_attr: OptTensor, dist_emb: OptTensor, dist: OptTensor, edge_mask: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:
        # query_i:[num_edges,hidden_dim] x_feat_j:[num_edges,hidden_dim] edge_attr:[num_edges,hidden_dim//4]
        query_i = query_i.view(-1, self.heads, self.out_channels) 
        edge_key_ij, edge_value_ij = self.lin_kv(torch.cat([x_feat_j, edge_attr], dim=-1)).view(-1, self.heads, 2, self.out_channels).unbind(dim=2) # shape [N * N, heads, out_channels]

        dk, dv = self.lin_dkv(dist_emb).view(-1, self.heads, 2, self.out_channels).unbind(dim=2) # shape [N * N, heads, out_channels]

        alpha_ij = (query_i * edge_key_ij * dk).sum(dim=-1) / math.sqrt(self.out_channels) # shape [N * N, heads]
        
        if self.attn_activation == 'silu':
            alpha_ij = F.silu(alpha_ij)
        elif self.attn_activation == 'softmax':
            alpha_ij = softmax(alpha_ij, index, ptr, size_i) 
        else:
            raise ValueError(f"Invalid activation function")
        alpha_ij = F.dropout(alpha_ij, p=self.dropout, training=self.training)

        if self.dataset == 'qm9' or self.dataset == 'md17':
            msg = dv * edge_value_ij * alpha_ij.view(-1, self.heads, 1) # shape [N * N, heads, out_channels]
        else:
            cutoff = self.cutoff(dist, edge_mask)
            # node feature message
            msg = cutoff.view(-1, 1, 1) * dv * edge_value_ij * alpha_ij.view(-1, self.heads, 1) # shape [N * N, heads, out_channels]
        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class TransLayerOptimV6(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, attn_activation='silu', dataset='qm9',**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransLayerOptimV6, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.dataset = dataset

        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=bias) #[hidden_dim,hidden_dim]
        self.lin_k = nn.Linear(in_channels + edge_dim, heads * out_channels, bias=bias) #[hidden_dim+edge_dim,hidden_dim]
        self.lin_v = nn.Linear(in_channels + edge_dim, heads * out_channels * 3, bias=bias)

        self.proj = nn.Linear(out_channels, out_channels*3, bias=bias) #[hidden_dim,hidden_dim]
        self.lin_dk = nn.Sequential(
            nn.Linear(in_channels // 4, in_channels // 2, bias=bias),
            nn.SiLU(),
            nn.Linear(in_channels // 2, heads * out_channels, bias=bias)
        ) #[hidden_dim//4,hidden_dim*2]
        self.lin_dv = nn.Sequential(
            nn.Linear(in_channels // 4, in_channels // 2, bias=bias),
            nn.SiLU(),
            nn.Linear(in_channels // 2, heads * out_channels * 3, bias=bias),
        ) #[hidden_dim//4,hidden_dim*2]

        self.vec_proj = nn.Linear(heads *out_channels, heads *out_channels * 3, bias=False)

        self.attn_activation = attn_activation
        
        self.cutoff = CosineCutoff(0, 5, use_mask=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()
        self.proj.reset_parameters()
        self.vec_proj.reset_parameters()
    
    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                dist: OptTensor = None,
                dist_emb: OptTensor = None,
                edge_vec: OptTensor = None,
                edge_mask: OptTensor = None,
                vec: OptTensor = None
                ) -> Tensor:
        """"""
        x_feat = x #[num_nodes,hidden_dim]
        query = self.lin_q(x_feat) #[num_nodes,hidden_dim]
        # propagate_type: (x: PairTensor, edge_attr: OptTensor) 
        vec1,vec2,vec3 = self.vec_proj(vec).view(-1,3,self.heads, 3, self.out_channels).unbind(dim=3)
        vec = vec.reshape(-1, 3, self.heads, self.out_channels) 
        vec_dot = (vec1 * vec2).sum(dim=1) #[-1,heads,out_channels]
        out_x,vec = self.propagate(edge_index, query=query, x_feat=x_feat,vec=vec, edge_attr=edge_attr, dist_emb=dist_emb, dist=dist, edge_mask=edge_mask,edge_vec = edge_vec)
        out_x = out_x.view(-1, self.out_channels)
        q1,q2,q3 = self.proj(out_x).view(-1, self.heads, 3, self.out_channels).unbind(dim=2)
        dx = q2 + q3*vec_dot
        dvec = vec3 * q1.unsqueeze(1) + vec
        dx = dx.reshape(-1,self.out_channels*self.heads)
        dvec = dvec.reshape(-1,3,self.out_channels*self.heads)

        return dx,dvec

    def message(self, query_i: Tensor, x_feat_j: Tensor,vec_j: Tensor,
                edge_attr: OptTensor, dist_emb: OptTensor, dist: OptTensor, edge_mask: OptTensor,edge_vec: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:
        # query_i:[num_edges,hidden_dim] x_feat_j:[num_edges,hidden_dim] edge_attr:[num_edges,hidden_dim//4]
        query_i = query_i.view(-1, self.heads, self.out_channels) 
        edge_key_ij = self.lin_k(torch.cat([x_feat_j, edge_attr], dim=-1)).view(-1, self.heads, self.out_channels) # shape [N * N, heads, out_channels]
        edge_value_ij = self.lin_v(torch.cat([x_feat_j, edge_attr], dim=-1)).view(-1, self.heads,self.out_channels*3) # shape [N * N, heads, out_channels*3]

        dk = self.lin_dk(dist_emb).view(-1, self.heads, self.out_channels) # shape [N * N, heads, out_channels]
        dv = self.lin_dv(dist_emb).view(-1, self.heads, self.out_channels*3) # shape [N * N, heads, out_channels*3]

        alpha_ij = (query_i * edge_key_ij * dk).sum(dim=-1) / math.sqrt(self.out_channels) # shape [N * N, heads]
    
        if self.attn_activation == 'silu':
            alpha_ij = F.silu(alpha_ij)
        elif self.attn_activation == 'softmax':
            alpha_ij = softmax(alpha_ij, index, ptr, size_i) 
        else:
            raise ValueError(f"Invalid activation function")
        alpha_ij = F.dropout(alpha_ij, p=self.dropout, training=self.training)

        if self.dataset == 'qm9' or self.dataset == 'md17':
            h,s1,s2 = (dv * edge_value_ij).view(-1, self.heads, 3, self.out_channels).unbind(dim=2) # shape [N * N, heads, out_channels]
            msg = h* alpha_ij.view(-1, self.heads, 1) # shape [N * N, heads, out_channels]
            vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * edge_vec.unsqueeze(
            2
            ).unsqueeze(3)
        else:
            cutoff = self.cutoff(dist, None)
            h,s1,s2 = (dv * edge_value_ij).view(-1, self.heads, 3, self.out_channels).unbind(dim=2) # shape [N * N, heads, out_channels]
            # node feature message
            msg = cutoff.view(-1, 1, 1) * h * alpha_ij.view(-1, self.heads, 1) # shape [N * N, heads, out_channels]
            vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * edge_vec.unsqueeze(
            2
            ).unsqueeze(3)
        return msg,vec_j

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec
    
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    

class NodeEmbedding(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_uniform_(self.mask_token)
    
    def forward(self, node_feature, node_mask=None):
        node_feature = self.node_embedding(node_feature)
        if node_mask is not None:
            node_feature[node_mask] = self.mask_token.to(node_feature.dtype)
        return node_feature

class RelaTransEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_heads, n_blocks,prior_model,args):
        super().__init__()
        self.args = args

        self.node_embedding = NodeEmbedding(node_dim, hidden_dim)
        self.neigh_embedding = NodeEmbedding(node_dim, hidden_dim)

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        self.wf = nn.Linear(hidden_dim//4,hidden_dim,bias=False)
        self.node_proj = nn.Linear(2*hidden_dim,hidden_dim)

        self.distance_expansion = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=hidden_dim//4, trainable=False, use_mask=True)

        self.encoder_blocks = nn.ModuleList([
            DMTBlock(hidden_dim, edge_dim=hidden_dim // 4, time_dim=hidden_dim, num_heads=n_heads, cond_time=False, dropout=args.dropout, pair_update=args.pair_update, trans_version=args.trans_version, attn_activation=args.attn_activation,dataset=args.dataset)
            for _ in range(n_blocks)
        ])
        self.prior_model = prior_model
        self.out_norm_vec = EquivariantLayerNorm(hidden_dim)

        if args.dataset == 'qm9' or args.dataset == 'md17':
            if args.dataset_arg == 'dipole_moment':
                self.output_model = EquivariantDipoleMoment(args.hidden_dim)
            elif args.dataset_arg == 'electronic_spatial_extent':
                self.output_model = EquivariantElectronicSpatialExtent(args.hidden_dim)
            else:
                self.output_model = Scalar(args.hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        
    def forward(self, data, node_feature, edge_index, edge_feature, position, pos_mask=None):
        masked_position = position
        if pos_mask is not None:
            masked_position = position[~pos_mask] 
            node_feature = node_feature[~pos_mask]

            edge_mask = pos_mask[edge_index].any(dim=0)
            edge_index = edge_index[:, ~edge_mask]  
            edge_feature = edge_feature[~edge_mask]
            nodes = torch.unique(edge_index) 
            num_nodes = nodes.size(0)
            mapping = -torch.ones(edge_index.max().item() + 1, dtype=torch.long, device=edge_index.device) 
            mapping[nodes] = torch.arange(num_nodes, device=edge_index.device)
            edge_index = mapping[edge_index]
        else: 
            edge_mask = None

        node = torch.cat([node_feature, masked_position], dim=-1) #[num_nodes,node_dim+3]
        node_neigh = self.neigh_embedding(node, None) #[num_nodes,hidden_dim]
        node_input = self.node_embedding(node, None) #[num_nodes,hidden_dim]

        distance = torch.sum((masked_position[edge_index[0]] - masked_position[edge_index[1]] + 1e-5) ** 2, dim=-1, keepdim=True)
        distance = torch.sqrt(distance) #[num_edges,1]
        
        edge_vec = (masked_position[edge_index[0]] - masked_position[edge_index[1]]) / distance
        edge_vec = torch.nan_to_num(edge_vec, nan=0.0, posinf=0.0, neginf=0.0) #[num_edges,3]

        distance_embedding = self.distance_expansion(distance.squeeze(-1), None) #[num_edges,hidden_dim//4]
        distance_h = self.wf(distance_embedding) #[num_edges,hidden_dim]
        neighbor = node_neigh.index_select(0, edge_index[1])*distance_h #[num_edges,hidden_dim]
        x_neighbors = torch.zeros(
            node_feature.shape[0], self.args.hidden_dim, dtype=node_feature.dtype, device=node_feature.device
        ).index_add(0, edge_index[0], neighbor.to(node_feature.dtype)) #[num_nodes,hidden_dim]
        h = self.node_proj(torch.cat([node_input,x_neighbors],dim=1)) #[num_nodes,hidden_dim]

        vec = torch.zeros(h.size(0), 3, h.size(1), device=h.device)

        edge_h = self.edge_embedding(edge_feature) #[num_edges,hidden_dim//4]

        for lidx,block in enumerate(self.encoder_blocks):
            dx,dvec = block(h, edge_h, edge_index, dist=distance, dist_emb=distance_embedding, edge_vec=edge_vec, edge_mask=edge_mask,vec=vec)
            h = h + dx
            vec = vec + dvec

        h = self.norm1(h)
        if self.args.dataset == 'qm9' or self.args.dataset == 'md17':
            if self.args.dataset_arg == 'dipole_moment':
                pred = self.output_model.pre_reduce(x=h,v=vec,z=data.z,pos=data.pos,batch=data.batch)
                pred = scatter_sum(pred, data.batch, dim=0)
                pred = self.output_model.post_reduce(pred)
            elif self.args.dataset_arg == 'electronic_spatial_extent':
                pred = self.output_model.pre_reduce(x=h,v=vec,z=data.z,pos=data.pos,batch=data.batch)
                pred = scatter_sum(pred, data.batch, dim=0)
                pred = self.output_model.post_reduce(pred)
            elif self.args.dataset_arg == 'energy_U0' or self.args.dataset_arg == 'energy_U' or self.args.dataset_arg == 'enthalpy_H' or self.args.dataset_arg == 'free_energy':
                pred = self.output_model.pre_reduce(x=h,v=vec,z=data.z,pos=data.pos,batch=data.batch)
                pred = self.prior_model.pre_reduce(pred,data.z)
                pred = scatter_sum(pred, data.batch, dim=0)
                pred = self.output_model.post_reduce(pred)
            else:
                pred = self.output_model.pre_reduce(x=h,v=vec,z=data.z,pos=data.pos,batch=data.batch)
                pred = scatter_sum(pred, data.batch, dim=0)
                pred = self.output_model.post_reduce(pred)
                return pred, vec

        return h, vec

class RelaTrans2DEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_heads, n_blocks,args):
        super().__init__()

        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim +1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # transformer blocks
        self.encoder_blocks = nn.ModuleList([
            DMTBlock(hidden_dim, edge_dim=hidden_dim // 4, time_dim=hidden_dim, num_heads=n_heads, cond_time=False, dropout=args.dropout, pair_update=args.pair_update, trans_version='v3', attn_activation=args.attn_activation,dataset=args.dataset)
            for _ in range(n_blocks)
        ])

        self.head = nn.Linear(hidden_dim, 4*hidden_dim)

        self.predictor = nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim, bias=True),
                                        nn.BatchNorm1d(4*hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(4*hidden_dim, 4*hidden_dim,bias=True)) # output layer
        
    def forward(self, data, node_feature, edge_index, edge_feature):
        h = self.node_embedding(node_feature)
        edge_h = self.edge_embedding(edge_feature)

        for block in self.encoder_blocks:
            h, edge_h = block(h, edge_h, edge_index)
         
        rep = self.predictor(h)
        h = self.head(h)

        return rep,h
