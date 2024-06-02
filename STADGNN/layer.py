import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class multi_shallow_embedding(nn.Module):

    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()

        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))

    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)

    def forward(self, device):
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')
        adj_flat = adj.reshape(self.num_graphs, -1)
        indices = adj_flat.topk(k=self.k)[1].reshape(-1)

        idx = torch.tensor([i // self.k for i in range(indices.size(0))], device=device)

        adj_flat = torch.zeros_like(adj_flat).clone()
        adj_flat[idx, indices] = 1.
        adj = adj_flat.reshape_as(adj)

        return adj


class Group_Linear(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()

        self.out_channels = out_channels
        self.groups = groups

        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups,
                                   bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.group_mlp.reset_parameters()

    def forward(self, x: Tensor, is_reshape: False):
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups

        if not is_reshape:
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        x = x.transpose(1, 2).reshape(B, G * C, N, -1)

        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)
        return out


class DenseGCNConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Group_Linear(in_channels, out_channels, groups, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        init.zeros_(self.bias)

    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj

    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        adj = self.norm(adj, add_loop).unsqueeze(1)
        x = self.lin(x, False)

        out = torch.matmul(adj, x)

        B, C, _, N, _ = out.size()
        out = out.transpose(2, 3).reshape(B, C, N, -1)

        if self.bias is not None:
            out = out.transpose(1, -1) + self.bias
            out = out.transpose(1, -1)

        return out


class DenseGINConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, eps=0, train_eps=True):
        super().__init__()
        self.mlp = Group_Linear(in_channels, out_channels, groups, bias=False)

        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)

    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj

    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        B, C, N, _ = x.size()
        G = adj.size(0)

        adj = self.norm(adj, add_loop=False)

        x = x.reshape(B, C, N, G, -1).transpose(2, 3)

        out = torch.matmul(adj, x)

        # DYNAMIC
        x_pre = x[:, :, :-1, ...]

        out[:, :, 1:, ...] = out[:, :, 1:, ...] + x_pre

        if add_loop:
            out = (1 + self.eps) * x + out

        out = self.mlp(out, True)
        C = out.size(1)
        out = out.transpose(2, 3).reshape(B, C, N, -1)

        return out


class Dense_TimeDiffPool2d(nn.Module):

    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()

        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))

        self.re_param = Parameter(Tensor(kern_size, 1))

    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')

    def forward(self, x: Tensor, adj: Tensor):
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)

        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))

        return out, out_adj