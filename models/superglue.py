#!/usr/bin/env python3
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
from copy import deepcopy
from itertools import permutations
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    for i in range(n):
        channels[i] = (int)(channels[i])
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """Normalize keypoints locations based on image image_shape"""
    # _, _, height, width = image_shape
    # one = kpts.new_tensor(1)
    # size = torch.stack([one * width, one * height])[None]
    # center = size / 2
    # scaling = size.max(1, keepdim=True).values * 0.7
    # return (kpts - center[:, None, :]) / scaling[:, None, :]

    return torch.stack(
        [(kpts[:, :, 0] - 320) / 320, (kpts[:, :, 1] - 240) / 240], dim=2
    )


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        if torch.any(kpts.isnan()) or torch.any(scores.isnan()):
            print("input KENC nan")
            exit()
        inputs = torch.cat(
            [kpts.transpose(1, 2).float(), scores.unsqueeze(1).float()], dim=1
        )
        return self.encoder(inputs)


class myKeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, feature_dim)
        )
        # MLP([3] + layers + [feature_dim])
        # nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        if torch.any(kpts.isnan()) or torch.any(scores.isnan()):
            print("input mKENC nan")
            exit()
        inputs = torch.cat(
            [kpts.transpose(1, 2).float(), scores.unsqueeze(1).float()], dim=1
        )
        inputs = torch.permute(inputs, (0, 2, 1))
        outputs = self.encoder(inputs)
        return torch.permute(outputs, (0, 2, 1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        # from torch_geometric.nn import MLP as pygMLP
        # self.mlp = pygMLP([feature_dim * 2, feature_dim, feature_dim], plain_last=True)
        # self.mlp = nn.Sequential([nn.linear()])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        # full connection between x and source
        message = self.attn(x, source, source)

        return self.mlp(torch.cat([x, message], dim=1))
        # print(torch.permute(torch.cat([x, message], dim=1), (0,2,1)).shape)
        # return torch.permute(self.mlp(torch.permute(torch.cat([x, message], dim=1), (0,2,1))), (0,2,1))


from itertools import product, permutations


def generate_edges_intra(len1, len2, with_type: bool = False):
    edges1 = (
        torch.tensor(
            list(permutations(range(len1), 2)), dtype=torch.long, requires_grad=False
        )
        .t()
        .contiguous()
    )
    edges2 = (
        torch.tensor(
            list(permutations(range(len2), 2)), dtype=torch.long, requires_grad=False
        )
        .t()
        .contiguous()
        + len1
    )
    edges = torch.cat([edges1, edges2], dim=1).cuda()
    edge_type = torch.zeros(len1 * (len1 - 1) + len2 * (len2 - 1), dtype=int).cuda()

    if with_type == False:
        return edges
    else:
        return edges, edge_type


def generate_edges_cross(len1, len2, with_type: bool = False):
    edges1 = (
        torch.tensor(list(product(range(len1), range(len1, len1 + len2))))
        .t()
        .contiguous()
    )
    edges2 = (
        torch.tensor(list(product(range(len1, len1 + len2), range(len1))))
        .t()
        .contiguous()
    )
    edges = torch.cat([edges1, edges2], dim=1).cuda()
    edge_type = torch.ones(len1 * len2 + len2 * len1, dtype=int).cuda()
    if with_type == False:
        return edges
    else:
        return edges, edge_type


def generate_edges_union(len1, len2, with_type: bool = False):
    if with_type:
        edges_inter, edge_type_inter = generate_edges_intra(len1, len2, with_type)
        edges_cross, edge_type_cross = generate_edges_cross(len1, len2, with_type)
        edge_type = torch.cat([edge_type_inter, edge_type_cross], dim=0).cuda()
    else:
        edges_inter = generate_edges_intra(len1, len2, with_type)
        edges_cross = generate_edges_cross(len1, len2, with_type)
    edges = torch.cat([edges_inter, edges_cross], dim=1)
    if with_type == False:
        return edges
    else:
        return edges, edge_type


class myAttentionalPropagation(nn.Module):
    def __init__(self, model: str, feature_dim: int, num_heads: int = 1):
        super().__init__()
        from torch_geometric.nn import GATConv, RGATConv

        self.model = model
        if model == "gat":
            self.conv = GATConv(
                in_channels=feature_dim,
                out_channels=feature_dim,
                heads=num_heads,
                concat=False,
            )  # .cuda()
        elif model == "rgat":
            self.conv = RGATConv(
                in_channels=feature_dim,
                out_channels=feature_dim,
                heads=num_heads,
                concat=False,
                num_relations=2,
            )

    def forward(self, x, edge_index, edge_type=None):
        # for PyG
        batch_size, nnodes, nfeatures = x.shape
        # print('x', x.shape)
        from torch_geometric.data import Batch, Data

        batch_list = []
        for i in range(x.shape[0]):
            batch_list.append(
                Data(x=x[i, :, :], edge_index=edge_index, edge_type=edge_type)
            )
        batch = Batch.from_data_list(batch_list)
        # print(
        #     "batch.x, .edge_index",
        #     batch.x.device,
        #     batch.x.shape,
        #     batch.edge_index.shape,
        # )
        if self.model == "gat":
            result = self.conv(x=batch.x, edge_index=batch.edge_index)
        else:
            result = self.conv(
                x=batch.x, edge_index=batch.edge_index, edge_type=batch.edge_type
            )
        return torch.reshape(result, (batch_size, nnodes, nfeatures))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, num_heads):
        super().__init__()

        self.feature_dim = feature_dim
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]
        )
        self.names = layer_names

    def forward(self, desc0, desc1):
        i = 0
        for layer, name in zip(self.layers, self.names):
            i += 1
            layer.attn.prob = []
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            # skip conn
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            if torch.any(desc0.isnan()) or torch.any(desc1.isnan()):
                print("desc nan:", i)
                exit()
        return desc0, desc1


class myGAT(nn.Module):
    def __init__(self, model: str, feature_dim: int, layer_names: list, num_heads):
        super().__init__()
        from torch_geometric.nn import MLP as pygMLP
        from torch_geometric.nn.norm import InstanceNorm, BatchNorm

        self.feature_dim = feature_dim
        self.layers = [
            (
                myAttentionalPropagation(
                    model=model, feature_dim=feature_dim, num_heads=num_heads
                ).cuda(),
                MLP(
                    [
                        feature_dim * (1 + 1),
                        feature_dim * (1 + 1),
                        feature_dim,
                    ]
                ).cuda(),
            )
            for _ in range(len(layer_names))
        ]
        self.names = layer_names

    def forward(self, desc0, desc1):
        # (B, D, N), (B, D, N)

        x = torch.cat([desc0, desc1], dim=2).cuda()
        size0, size1 = desc0.shape[2], desc1.shape[2]

        for (mp, mlp), name in zip(self.layers, self.names):
            x = torch.permute(x, (0, 2, 1)).float()  # -> (B, N, D)
            # 1. aggregation: in feature_dim, out feature_dim * 1
            if name == "cross":
                edges = generate_edges_cross(size0, size1)
            elif name == "self":
                edges = generate_edges_intra(size0, size1)
            elif name == "union":
                edges = generate_edges_union(size0, size1)
            msg = mp(x, edges)
            # 2. cat with x: in feature_dim, out feature_dim*(1 + 1)
            # 3. pass through a MLP: in feature_dim * 2, out feature_dim (internal dim see init)
            # 4. skip connection: in feature_dim, out feature_dim
            xmsg = torch.cat([x, msg], dim=2)  # -> (B, N, D*(1+1))
            xmsg = torch.permute(xmsg, (0, 2, 1))  # -> (B, D*(1+1), N)
            x = torch.permute(x, (0, 2, 1))  # -> (B, D, N)
            x += mlp(xmsg)  # -> (B, D, N)
        desc0 = x[:, :, :size0]
        desc1 = x[:, :, size0:]
        return desc0, desc1


class myWholeGAT(nn.Module):
    def __init__(self, model: str, feature_dim: int, layer_names: list, num_heads):
        super().__init__()
        from torch_geometric.nn import MLP as pygMLP
        from torch_geometric.nn.norm import InstanceNorm, BatchNorm

        self.feature_dim = feature_dim
        from torch_geometric.nn import GATConv

        gat_config = {
            "in_channels": feature_dim,
            "out_channels": feature_dim,
            "num_relations": 2,
            "heads": num_heads,
            "concat": False,
        }
        self.conv = []
        self.mlp = []
        for i in range(len(layer_names)):
            self.conv.append(GATConv(**gat_config).cuda())
            self.mlp.append(pygMLP([feature_dim * 2, feature_dim]).cuda())
        self.names = layer_names

    def forward(self, desc0, desc1):
        # (B, D, N), (B, D, N)

        # print(desc0.shape, desc1.shape, desc0.device, desc1.device)
        x = torch.cat([desc0, desc1], dim=2).cuda()
        # print(x.shape, x.device)
        size0, size1 = desc0.shape[2], desc1.shape[2]
        edges_intra = generate_edges_intra(size0, size1)
        edges_cross = generate_edges_cross(size0, size1)
        # edges, edge_type = generate_edges_union(size0, size1, True)
        # print('edges', edges.shape, edge_type.shape)
        x = torch.permute(x, (0, 2, 1)).float()  # -> (B, N, D)

        batch_size, num_nodes, nfeatures = x.shape
        # print('x', x.shape)
        from torch_geometric.data import Batch, Data

        batch_intra_list = []
        batch_cross_list = []
        batch_x_list = []
        # x_ = x.deepcopy()
        for i in range(x.shape[0]):
            batch_intra_list.append(
                Data(x=x[i, :, :], edge_index=edges_intra, num_nodes=num_nodes)
            )
            batch_cross_list.append(
                Data(x=x[i, :, :], edge_index=edges_cross, num_nodes=num_nodes)
            )
            batch_x_list.append(Data(x=x[i, :, :]))
        batch_intra = Batch.from_data_list(batch_intra_list)
        batch_cross = Batch.from_data_list(batch_cross_list)
        batch_x = Batch.from_data_list(batch_x_list)

        now_x = batch_x.x
        print("batch_x", batch_x.x.shape)

        for i in range(len(self.names)):
            if self.names[i] == "self":
                # batch_intra.x = now_x
                msg1 = self.conv[i](x=now_x, edge_index=batch_intra.edge_index).relu()
                msg2 = self.mlp[i](torch.cat([now_x, msg1], dim=1))
                now_x += msg2
                # now_x = batch_intra.x
            else:
                # batch_cross.x = now_x
                msg1 = self.conv[i](x=now_x, edge_index=batch_cross.edge_index).relu()
                msg2 = self.mlp[i](torch.cat([now_x, msg1], dim=1))
                now_x += msg2
                # now_x = batch_cross.x

        x = torch.reshape(now_x, (batch_size, num_nodes, nfeatures))
        x = torch.permute(x, (0, 2, 1))  # -> (B, D, N)
        desc0 = x[:, :, :size0]
        desc1 = x[:, :, size0:]
        # print(desc0.shape, desc1.shape, desc0.device, desc1.device)
        return desc0, desc1


class myRGAT(nn.Module):
    def __init__(self, model: str, feature_dim: int, layer_names: list, num_heads):
        super().__init__()
        from torch_geometric.nn import MLP as pygMLP
        from torch_geometric.nn.norm import InstanceNorm, BatchNorm

        self.feature_dim = feature_dim
        self.layers = [
            (
                myAttentionalPropagation(
                    model=model, feature_dim=feature_dim, num_heads=num_heads
                ).cuda(),
                MLP(
                    [
                        feature_dim * (1 + 1),
                        feature_dim * (1 + 1),
                        feature_dim,
                    ]
                ).cuda(),
            )
            for _ in range(len(layer_names))
        ]
        self.names = layer_names

    def forward(self, desc0, desc1):
        # (B, D, N), (B, D, N)

        # print(desc0.shape, desc1.shape, desc0.device, desc1.device)
        x = torch.cat([desc0, desc1], dim=2).cuda()
        # print(x.shape, x.device)
        size0, size1 = desc0.shape[2], desc1.shape[2]
        # edges_intra = generate_edges_intra(size0, size1)
        # edges_cross = generate_edges_cross(size0, size1)
        edges, edge_type = generate_edges_union(size0, size1, True)
        # print('edges', edges.shape, edge_type.shape)
        for (mp, mlp), name in zip(self.layers, self.names):
            x = torch.permute(x, (0, 2, 1)).float()  # -> (B, N, D)
            # print('x', x.shape, x.device)
            # print(name)
            # 1. aggregation: in feature_dim, out feature_dim * num_heads
            msg = mp(x, edges, edge_type)
            # print('msg', msg.shape, msg.device)
            # 2. cat with x: in feature_dim, out feature_dim*(num_heads + 1)
            # 3. pass through a MLP: in feature_dim * 2, out feature_dim (internal dim see init)
            # 4. skip connection: in feature_dim, out feature_dim
            xmsg = torch.cat([x, msg], dim=2)  # -> (B, N, D*(num_heads+1))
            xmsg = torch.permute(xmsg, (0, 2, 1))  # -> (B, D*(num_heads+1), N)
            # print('xmsg', xmsg.shape)
            x = torch.permute(x, (0, 2, 1))  # -> (B, D, N)
            # print('x', x.shape, x.device)
            x += mlp(xmsg)  # -> (B, D, N)
            # print('x', x.shape, x.device)
        desc0 = x[:, :, :size0]
        desc1 = x[:, :, size0:]
        # print(desc0.shape, desc1.shape, desc0.device, desc1.device)
        return desc0, desc1


class myWholeRGAT(nn.Module):
    def __init__(
        self,
        model: str,
        feature_dim: int,
        layer_names: list,
        num_heads: int = 1,
        edge_pool: list = None,
    ):
        super().__init__()
        from torch_geometric.nn import MLP as pygMLP
        from torch_geometric.nn.norm import InstanceNorm, BatchNorm

        self.feature_dim = feature_dim
        from torch_geometric.nn import RGATConv

        rgat_config = {
            "in_channels": feature_dim,
            "out_channels": feature_dim,
            "num_relations": 2,
            "heads": num_heads,
            "concat": False,
        }
        self.conv = []
        self.lin = []
        self.norm = []
        self.pool = edge_pool
        for i in range(len(layer_names)):
            self.conv.append(RGATConv(**rgat_config).cuda())
            self.lin.append(nn.Linear(feature_dim * 2, feature_dim).cuda())
            self.norm.append(BatchNorm(feature_dim).cuda())
        self.names = layer_names

    def edge_pool(self, batch, attention_weights, k):
        from torch_geometric.data import Batch, Data

        # from torch_geometric.utils import unbatch_edge_index
        batch_list = Batch.to_data_list(batch)
        # edge_index_list = unbatch_edge_index(batch.edge_index)
        batch_size = len(batch_list)

        id_start = 0
        for b in range(batch_size):
            num_edges = batch_list[b].edge_index.shape[-1]
            id_end = id_start + num_edges
            attention_weights_now = torch.mean(
                attention_weights[id_start:id_end, :], dim=1
            )
            # print(attention_weights_now.shape)
            topk_ids = torch.topk(attention_weights_now, int(k * num_edges)).indices

            batch_list[b].edge_index = batch_list[b].edge_index[:, topk_ids]
            batch_list[b].edge_type = batch_list[b].edge_type[topk_ids]

            id_start = id_end
        batch = Batch.from_data_list(batch_list)
        return batch

    def forward(self, desc0, desc1):
        # (B, D, N), (B, D, N)
        from torch_geometric.data import Batch, Data

        # print(desc0.shape, desc1.shape, desc0.device, desc1.device)
        x = torch.cat([desc0, desc1], dim=2).cuda()
        # print(x.shape, x.device)
        size0, size1 = desc0.shape[2], desc1.shape[2]
        # edges_intra = generate_edges_intra(size0, size1)
        # edges_cross = generate_edges_cross(size0, size1)
        edges, edge_type = generate_edges_union(size0, size1, True)
        # print('edges', edges.shape, edge_type.shape)
        x = torch.permute(x, (0, 2, 1)).float()  # -> (B, N, D)

        batch_size, num_nodes, nfeatures = x.shape
        # print('x', x.shape)

        batch_list = []
        # batch_intra_list = []
        # batch_cross_list = []
        for i in range(x.shape[0]):
            batch_list.append(Data(x=x[i, :, :], edge_index=edges, edge_type=edge_type))
            # batch_intra_list.append(Data(edge_index=edges_intra, num_nodes=num_nodes))
            # batch_cross_list.append(Data(edge_index=edges_cross, num_nodes=num_nodes))
        batch = Batch.from_data_list(batch_list)
        # batch_intra = Batch.from_data_list(batch_intra_list)
        # batch_cross = Batch.from_data_list(batch_cross_list)
        from torch_geometric.nn.pool import knn_graph

        for i in range(len(self.names)):
            # print("x", batch.x.shape)
            msg1, (_, attention_weights) = self.conv[i](
                x=batch.x,
                edge_index=batch.edge_index,
                edge_type=batch.edge_type,
                return_attention_weights=True,
            )
            msg1 = msg1.relu()
            # print('attn', attention_weights.shape)
            # print("msg1", msg1.shape)
            msg2 = self.lin[i](torch.cat([batch.x, msg1], dim=1))
            # print("msg2", msg2.shape)
            msg3 = self.norm[i](msg2)
            batch.x += msg3
            if self.pool[i] != 1:
                batch = self.edge_pool(
                    batch, attention_weights=attention_weights, k=self.pool[i]
                )
            # print("i", i, self.names[i])
            # print("batch.edge_index", batch.edge_index.shape)

        x = torch.reshape(
            batch.x,
            (batch_size, num_nodes, nfeatures),
        )
        x = torch.permute(x, (0, 2, 1))  # -> (B, D, N)
        desc0 = x[:, :, :size0]
        desc1 = x[:, :, size0:]
        # print(desc0.shape, desc1.shape, desc0.device, desc1.device)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        if torch.any(u.isnan()) or torch.any(v.isnan()):
            print("S nan", _)
            exit()
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """

    default_config = {
        # 'descriptor_dim': 128,
        # 'weights': 'indoor',
        # 'keypoint_encoder': [32, 64, 128],
        # 'GNN_layers': ['self', 'cross'] * 9,
        # 'sinkhorn_iterations': 100,
        # 'match_threshold': 0.2,
        "load_ckpt": None,
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.config.update(self.config[self.config["model_config_name"]])
        print("SuperGlue Config:", self.config)

        if config["model_name"] == "ori":
            self.kenc = KeypointEncoder(
                self.config["descriptor_dim"], self.config["keypoint_encoder"]
            )
        else:
            self.kenc = myKeypointEncoder(
                self.config["descriptor_dim"], self.config["keypoint_encoder"]
            )

        if config["model_name"] == "ori":
            self.gnn = AttentionalGNN(
                self.config["descriptor_dim"], self.config["GNN_layers"], num_heads=4
            )

        elif config["model_name"] == "gat":
            self.gnn = myGAT(
                self.config["model_name"],
                self.config["descriptor_dim"],
                self.config["GNN_layers"],
                num_heads=4,
            )
        # elif config['model'] == 'wgat':
        #     self.gnn = myWholeGAT(
        #         self.config["model_name"],
        #         self.config["descriptor_dim"],
        #         self.config["GNN_layers"],
        #         num_heads=4,
        #     )

        elif config["model_name"] == "rgat":
            self.gnn = myRGAT(
                self.config["model_name"],
                self.config["descriptor_dim"],
                self.config["GNN_layers"],
                num_heads=4,
            ).cuda()

        elif config["model_name"] == "wrgat":
            self.gnn = myWholeRGAT(
                self.config["model_name"],
                self.config["descriptor_dim"],
                self.config["GNN_layers"],
                num_heads=4,
                edge_pool=[0.1, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1],
            ).cuda()

        self.final_proj = nn.Conv1d(
            self.config["descriptor_dim"],
            self.config["descriptor_dim"],
            kernel_size=1,
            bias=True,
        )
        # self.final_proj = nn.Linear(
        #     self.config["descriptor_dim"],
        #     self.config["descriptor_dim"],
        #     bias=True,
        # )

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)

        if self.config["load_ckpt_path"] != None:
            # assert self.config['weights'] in ['indoor', 'outdoor']
            path = Path(__file__).parent.parent
            path = path / f'{self.config["load_ckpt_path"]}'
            print(torch.load(path))
            self.load_state_dict(torch.load(path))
            print(f"Loaded SuperGlue model ({path})")

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # originally double

        # for k, v in data.items():
        #     print(k)
        #     if torch.is_tensor(v):
        #         print(v.shape)
        # print('Entering superglue')
        desc0, desc1 = data["descriptors0"].float(), data["descriptors1"].float()
        kpts0, kpts1 = data["keypoints0"].float(), data["keypoints1"].float()

        desc0 = desc0.permute((0, 2, 1))
        desc1 = desc1.permute((0, 2, 1))
        # kpts0 = kpts0.permute((0, 2, 1))
        # kpts1 = kpts1.permute((0, 2, 1))

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                "matches0": kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                "matches1": kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                "matching_scores0": kpts0.new_zeros(shape0)[0],
                "matching_scores1": kpts1.new_zeros(shape1)[0],
                "skip_train": True,
            }

        # file_name = data["file_name"]
        # all_matches = data["all_matches"]
        # all_matches = data["all_matches"].permute(
        #     1, 2, 0
        # )  # shape=torch.Size([1, 87, 2])

        # Keypoint normalization.
        # print(data["image0_shape"], data["image0_shape"].shape)
        kpts0 = normalize_keypoints(kpts0, data["image0_shape"])
        kpts1 = normalize_keypoints(kpts1, data["image1_shape"])
        # print(kpts0)
        # print(kpts0.shape)

        if torch.any(desc0.isnan()) or torch.any(desc1.isnan()):
            print("before KENC nan: desc")
            exit()

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data["scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["scores1"])

        if torch.any(desc0.isnan()) or torch.any(desc1.isnan()):
            print("after KENC nan")
            exit()

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        # print('gnn out desc0 desc1', desc0.shape, desc0.shape)
        if torch.any(desc0.isnan()) or torch.any(desc1.isnan()):
            print("desc nan")
            exit()
        # Final MLP projection.
        # desc0 = torch.permute(desc0, (0, 2, 1)).float()
        # desc1 = torch.permute(desc1, (0, 2, 1)).float()
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # mdesc0 = torch.permute(mdesc0, (0, 2, 1)).float()
        # mdesc1 = torch.permute(mdesc1, (0, 2, 1)).float()
        if torch.any(mdesc0.isnan()) or torch.any(mdesc1.isnan()):
            print("mdesc nan")
            exit()
        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / self.config["descriptor_dim"] ** 0.5

        if torch.any(scores.isnan()):
            print("scores nan: before OT")
            exit()
        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score, iters=self.config["sinkhorn_iterations"]
        )
        if torch.any(scores.isnan()):
            print("scores nan")
            exit()

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        # print('mscore0', mscores0.shape)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config["match_threshold"])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # check if indexed correctly

        partial_assignment_matrix = data["partial_assignment_matrix"]
        # batch_size, num_nodes, _ = all_matches.shape
        batch_size = partial_assignment_matrix.shape[0]
        num_nodes = partial_assignment_matrix.shape[1] - 1
        all_matches = num_nodes * torch.ones((batch_size, num_nodes, 2), dtype=int)
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if partial_assignment_matrix[b][i][j] == 1:
                        all_matches[b][i][0] = j
                        all_matches[b][j][1] = i
        # print(all_matches.shape)
        # print(len(data))
        # for k, v in data.items():
        #     if k != 'file_name':
        #         print(k, v.shape)
        # scores = nn.functional.softmax(scores, dim=2)
        total_loss = 0
        # print('batch_size', batch_size)
        batch_loss = []
        for b in range(batch_size):
            loss = []
            # loss_matrix = - partial_assignment_matrix[b] * scores[b]
            # loss_mean = torch.mean(loss_matrix)
            for i in range(num_nodes):
                y = all_matches[b][i][0]
                x = all_matches[b][i][1]
                # print(i, y, i, x)
                # print(scores[b][i][y].item(), scores[b][y][i].item())
                # print(scores[b][i][y].exp().item(), scores[b][y][i].exp().item())
                # print(torch.log(scores[b][i][y].exp()).item(), torch.log(scores[b][y][i].exp()).item())
                if y != num_nodes:
                    # loss.append(-0.5 * torch.log(scores[b][i][y].exp()))
                    loss.append(-0.5 * torch.log(scores[b][i][y]))
                else:
                    # loss.append(-torch.log(scores[b][i][y].exp()))
                    loss.append(-torch.log(scores[b][i][y]))
                if x != num_nodes:
                    # loss.append(-0.5 * torch.log(scores[b][x][i].exp()))
                    loss.append(-0.5 * torch.log(scores[b][x][i]))
                else:
                    # loss.append(-torch.log(scores[b][x][i].exp()))
                    loss.append(-torch.log(scores[b][x][i]))
                # print(loss[-1].item())
                # print('-----')
            loss_pt = torch.stack(loss)
            # print(loss_pt)
            loss_mean = torch.mean(loss_pt)
            if torch.any(loss_mean.isnan()):
                print("loss mean nan", b)
                exit()

            batch_loss.append(loss_mean.item())
            total_loss += loss_mean

        total_loss /= batch_size

        return {
            "matches0": indices0,  # use -1 for invalid match
            "matches1": indices1,  # use -1 for invalid match
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "loss": total_loss,  # loss_mean[0],
            "skip_train": False,
        }

        # scores big value or small value means confidence? log can't take neg value
