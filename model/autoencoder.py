import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from model.retrans import RelaTransEncoder, RelaTrans2DEncoder,EquivariantLayerNorm
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
from typing import Tuple, Optional
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from model.output_modules import EquivariantVectorOutput

class DecoderBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, hidden_dim, num_heads, act=nn.GELU, dropout=0.0, attn_activation='softmax'):
        super().__init__()
        self.dropout = dropout
        self.act = act()

        self.attn_mpnn = TransLayerOptimV7(hidden_dim, hidden_dim // num_heads, num_heads, dropout=dropout, attn_activation=attn_activation)

    def forward(self, edge_index,h, vec=None):
        """
        A more optimized version of forward_old using torch.compile
        Params:
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
        """
        h_in_node = h

        h_node,dvec = self.attn_mpnn(edge_index,h, vec=vec)

        h_out = h_in_node + h_node
        vec = dvec + vec
        return h_out, vec

class TransLayerOptimV7(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0,
                 bias: bool = True, attn_activation='silu',**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransLayerOptimV7, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=bias) #[hidden_dim,hidden_dim]
        self.lin_k = nn.Linear(in_channels, heads * out_channels, bias=bias) #[hidden_dim,hidden_dim]
        self.lin_v = nn.Linear(in_channels, heads * out_channels * 2, bias=bias)
        self.proj = nn.Linear(out_channels, out_channels*3, bias=bias) #[hidden_dim,hidden_dim]
        self.lin_dk = nn.Sequential(
            nn.Linear(in_channels // 4, heads * out_channels, bias=bias),
            nn.SiLU()
        ) #[hidden_dim//4,hidden_dim*2]
        self.lin_dv = nn.Sequential(
            nn.Linear(in_channels // 4, heads * out_channels * 2, bias=bias),
            nn.SiLU()
        ) #[hidden_dim//4,hidden_dim*2]

        self.vec_proj = nn.Linear(heads *out_channels, heads *out_channels * 3, bias=False)

        self.attn_activation = attn_activation
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()
        self.proj.reset_parameters()
        self.vec_proj.reset_parameters()
    
    def forward(self, edge_index,x: OptTensor,
                vec: OptTensor = None
                ) -> Tensor:
        """"""
        x_feat = x #[num_nodes,hidden_dim]

        query = self.lin_q(x_feat) #[num_nodes,hidden_dim]
        # propagate_type: (x: PairTensor, edge_attr: OptTensor) 
        vec1,vec2,vec3 = self.vec_proj(vec).view(-1,3,self.heads, 3, self.out_channels).unbind(dim=3)
        vec = vec.reshape(-1, 3, self.heads, self.out_channels) 
        vec_dot = (vec1 * vec2).sum(dim=1) #[-1,heads,out_channels]
        out_x,vec = self.propagate(edge_index,query=query, x_feat=x_feat,vec=vec)
        out_x = out_x.view(-1, self.out_channels)
        q1,q2,q3 = self.proj(out_x).view(-1, self.heads, 3, self.out_channels).unbind(dim=2)
        dx = q2 + q3*vec_dot
        dvec = vec3 * q1.unsqueeze(1) + vec
        dx = dx.reshape(-1,self.out_channels*self.heads)
        dvec = dvec.reshape(-1,3,self.out_channels*self.heads)
        return dx,dvec

    def message(self, query_i: Tensor, x_feat_j: Tensor,vec_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:
        # query_i:[num_edges,hidden_dim] x_feat_j:[num_edges,hidden_dim]
        query_i = query_i.view(-1, self.heads, self.out_channels) 
        key_ij = self.lin_k(x_feat_j).view(-1, self.heads, self.out_channels) # shape [N * N, heads, out_channels]
        value_ij = self.lin_v(x_feat_j).view(-1, self.heads,self.out_channels*2) # shape [N * N, heads, out_channels*3]

        alpha_ij = (query_i * key_ij).sum(dim=-1) / math.sqrt(self.out_channels) # shape [N * N, heads]
        
        if self.attn_activation == 'silu':
            alpha_ij = F.silu(alpha_ij)
        elif self.attn_activation == 'softmax':
            alpha_ij = softmax(alpha_ij, index, ptr, size_i) 
        else:
            raise ValueError(f"Invalid activation function")
        alpha_ij = F.dropout(alpha_ij, p=self.dropout, training=self.training)

        h,s1= value_ij.view(-1, self.heads,2, self.out_channels).unbind(dim=2) # shape [N * N, heads, out_channels]
        # node feature message
        msg = h * alpha_ij.view(-1, self.heads, 1) # shape [N * N, heads, out_channels]
        vec_j = vec_j * s1.unsqueeze(1) 
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
    
class StructureUnawareDecoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_blocks,args):
        super().__init__()
        self.args = args
        self.decoder_mask_token = nn.Parameter(torch.randn(1, hidden_dim))
        torch.nn.init.normal_(self.decoder_mask_token, mean=0.0, std=1.0)

        # transformer blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads=n_heads, dropout=args.dropout, attn_activation=args.attn_activation)
            for _ in range(n_blocks)
        ])
        self.out_norm_vec = EquivariantLayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        if args.pos_mask and args.denoising:
            self.coordinate_head = EquivariantVectorOutput(hidden_dim)
            self.denoising_head = EquivariantVectorOutput(hidden_dim)
        if not args.pos_mask and args.denoising:
            self.denoising_head = EquivariantVectorOutput(hidden_dim)
        if args.pos_mask and not args.denoising:
            self.coordinate_head = EquivariantVectorOutput(hidden_dim)
        
    def forward(self,data, rep, rep_2d, vec_mask, pos_mask=None):
        node_feature = data.x
        h = torch.zeros(node_feature.shape[0], self.args.hidden_dim, dtype=node_feature.dtype, device=node_feature.device)
        vec = torch.zeros(h.size(0), 3, h.size(1), device=h.device)
        if pos_mask is not None:
            h[~pos_mask] = rep
            h[pos_mask] = self.decoder_mask_token
            vec[~pos_mask] = vec_mask
            vec[pos_mask] = 0.0
        h = h + rep_2d
        
        for block in self.decoder_blocks:
            h, vec = block(data.edge_index,h, vec=vec)
        vec = self.out_norm_vec(vec)
        h = self.norm1(h)
        if self.args.pos_mask and self.args.denoising:
            coordinates = self.coordinate_head.pre_reduce(x=h,v=vec,z=data.z,pos=data.pos,batch=data.batch)
            coordinates = coordinates[pos_mask]
            pred_noise = self.denoising_head.pre_reduce(x=h,v=vec,z=data.z,pos=data.pos,batch=data.batch)
            pred_noise = pred_noise[~pos_mask]
            return coordinates,pred_noise
        if not self.args.pos_mask and self.args.denoising:
            pred_noise = self.denoising_head.pre_reduce(x=h,v=vec,z=data.z,pos=data.pos,batch=data.batch)
            return pred_noise
        if self.args.pos_mask and not self.args.denoising:
            coordinates = self.coordinate_head.pre_reduce(x=h,v=vec,z=data.z,pos=data.pos,batch=data.batch)
            coordinates = coordinates[pos_mask]
            return coordinates    
        
class AutoEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_heads, encoder_blocks, decoder_blocks, prior_model, args):
        super().__init__()
        self.args = args
        self.encoder = RelaTransEncoder(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_blocks=args.encoder_blocks,
        prior_model = prior_model,
        args=args
        )

        self.encoder2d = RelaTrans2DEncoder(
            node_dim=node_dim-3,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim_2d,
            n_heads=args.n_heads_2d,
            n_blocks=args.encoder_blocks,
            args=args
        )

        self.decoder = StructureUnawareDecoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=decoder_blocks,
            args=args
        )

    def forward(self, data):
        rep_2d,h_2d = self.encoder2d(
            data=data,
            node_feature=data.x,
            edge_index=data.edge_index,
            edge_feature=data.edge_attr,
        )

        rep,vec = self.encoder(
            data=data,
            node_feature=data.x,
            edge_index=data.edge_index,
            edge_feature=data.edge_attr,
            position=data.pos,
            pos_mask=data.pos_mask if self.args.pos_mask else None
        )
        
        if self.args.pos_mask and self.args.denoising:
            pred_coords,pred_noise = self.decoder(
                data=data, 
                rep=rep,
                rep_2d=h_2d.detach(),
                vec_mask = vec,
                pos_mask=data.pos_mask
            )
            return pred_coords, pred_noise,vec, rep_2d[~data.pos_mask],rep.detach()
        if not self.args.pos_mask and self.args.denoising:
            pred_noise = self.decoder(
                data=data, 
                rep=rep,
                rep_2d=h_2d.detach(),
                vec_mask = vec,
                pos_mask=None
            )
            return pred_noise,vec, rep_2d[~data.pos_mask],rep.detach()
        if self.args.pos_mask and not self.args.denoising:
            pred_coords = self.decoder(
                    data=data, 
                    rep=rep,
                    rep_2d=h_2d.detach(),
                    vec_mask = vec,
                    pos_mask=data.pos_mask
            )
            return pred_coords,vec, rep_2d[~data.pos_mask],rep.detach()


