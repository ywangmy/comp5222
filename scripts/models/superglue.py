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
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
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
        [(kpts[:, :, 0] - 320) / 320, (kpts[:, :, 0] - 240) / 240], dim=2
    )


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = torch.cat(
            [kpts.transpose(1, 2).float(), scores.unsqueeze(1).float()], dim=1
        )
        return self.encoder(inputs)


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
        # self.mlp = pygMLP([feature_dim * 2, feature_dim, feature_dim], plain_last=False)
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
    edges_inter, edge_type_inter = generate_edges_intra(len1, len2, with_type)
    edges_cross, edge_type_cross = generate_edges_cross(len1, len2, with_type)
    edges = torch.cat([edges_inter, edges_cross], dim=1).cuda()
    edge_type = torch.cat([edge_type_inter, edge_type_cross], dim=0).cuda()
    return edges, edge_type


class myAttentionalPropagation(nn.Module):
    def __init__(self, model: str, feature_dim: int, num_heads: int = 1):
        super().__init__()
        from torch_geometric.nn import GATConv, RGATConv

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
        print(
            "batch.x, .edge_index",
            batch.x.device,
            batch.x.shape,
            batch.edge_index.shape,
        )
        # print(batch, batch[0])
        # result_list = []
        # for i in range(x.shape[0]): # edge_index are common
        #     print('x[i], edge_index', x[i].shape, edge_index.shape)
        #     result_list.append(self.conv(x=x[i], edge_index=edge_index))
        # return torch.stack(result_list, dim=0)
        return torch.reshape(
            self.conv(
                x=batch.x, edge_index=batch.edge_index, edge_type=batch.edge_type
            ),
            (batch_size, nnodes, nfeatures),
        )


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, num_heads):
        super().__init__()

        self.feature_dim = feature_dim
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]
        )
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            # skip conn
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
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
                # nn.Sequential(
                #     nn.Linear(feature_dim * (1 + 1), feature_dim).cuda(),
                #     # BatchNorm(feature_dim).cuda(),
                #     nn.ReLU().cuda(),
                # )
                # pygMLP([feature_dim * 2, feature_dim], plain_last=False).cuda()
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
        for (mp, mlp), name in zip(self.layers, self.names):
            x = torch.permute(x, (0, 2, 1)).float()  # -> (B, N, D)
            # print('x', x.shape, x.device)

            # print(name)
            # 1. aggregation: in feature_dim, out feature_dim * num_heads
            if name == "cross":
                edges = generate_edges_cross(size0, size1)
            elif name == "self":
                edges = generate_edges_intra(size0, size1)
            msg = mp(x, edges)
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
            # exit()
        # x = torch.permute(x, (0, 2, 1)).float()  # -> (B, D, N)
        desc0 = x[:, :, :size0]
        desc1 = x[:, :, size0:]
        # print(desc0.shape, desc1.shape, desc0.device, desc1.device)
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
        from torch_geometric.nn.pool import knn_graph

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
        self.pool = []
        for i in range(len(layer_names)):
            self.conv.append(RGATConv(**rgat_config).cuda())
            self.lin.append(nn.Linear(feature_dim * 2, feature_dim).cuda())
            self.norm.append(BatchNorm(feature_dim).cuda())
            if edge_pool == None:
                self.pool.append(None)
            else:
                self.pool.append(knn_graph)
        self.names = layer_names

    def forward(self, desc0, desc1):
        # (B, D, N), (B, D, N)

        # print(desc0.shape, desc1.shape, desc0.device, desc1.device)
        x = torch.cat([desc0, desc1], dim=2).cuda()
        # print(x.shape, x.device)
        size0, size1 = desc0.shape[2], desc1.shape[2]
        edges_intra = generate_edges_intra(size0, size1)
        edges_cross = generate_edges_cross(size0, size1)
        edges, edge_type = generate_edges_union(size0, size1, True)
        # print('edges', edges.shape, edge_type.shape)
        x = torch.permute(x, (0, 2, 1)).float()  # -> (B, N, D)

        batch_size, num_nodes, nfeatures = x.shape
        # print('x', x.shape)
        from torch_geometric.data import Batch, Data

        batch_list = []
        batch_intra_list = []
        batch_cross_list = []
        for i in range(x.shape[0]):
            batch_list.append(Data(x=x[i, :, :], edge_index=edges, edge_type=edge_type))
            batch_intra_list.append(Data(edge_index=edges_intra, num_nodes=num_nodes))
            batch_cross_list.append(Data(edge_index=edges_cross, num_nodes=num_nodes))
        batch = Batch.from_data_list(batch_list)
        batch_intra = Batch.from_data_list(batch_intra_list)
        batch_cross = Batch.from_data_list(batch_cross_list)

        for i in range(len(self.names)):
            msg1 = self.conv[i](
                x=batch.x, edge_index=batch.edge_index, edge_type=batch.edge_type
            ).relu()
            msg2 = self.lin[i](torch.cat([batch.x, msg1], dim=1))
            msg3 = self.norm[i](msg2)
            batch.x += msg3
            if self.pool[i] != None:
                pooled_x_intra, batch_intra.edge_index = self.pool[i](
                    batch.x, batch_intra.edge_index, batch.batch
                )

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
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        print("SuperGlue Config:", self.config)

        self.kenc = KeypointEncoder(
            self.config["descriptor_dim"], self.config["keypoint_encoder"]
        )

        if config["model"] == "ori":
            self.gnn = AttentionalGNN(
                self.config["descriptor_dim"], self.config["GNN_layers"], num_heads=4
            )

        elif config["model"] == "gat":
            self.gnn = myGAT(
                self.config["model"],
                self.config["descriptor_dim"],
                self.config["GNN_layers"],
                num_heads=4,
            )
        # elif config['model'] == 'wgat':
        #     self.gnn = myWholeGAT(
        #         self.config["model"],
        #         self.config["descriptor_dim"],
        #         self.config["GNN_layers"],
        #         num_heads=4,
        #     )

        elif config["model"] == "rgat":
            self.gnn = myRGAT(
                self.config["model"],
                self.config["descriptor_dim"],
                self.config["GNN_layers"],
                num_heads=4,
            ).cuda()

        elif config["model"] == "wrgat":
            self.gnn = myWholeRGAT(
                self.config["model"],
                self.config["descriptor_dim"],
                self.config["GNN_layers"],
                num_heads=4,
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

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # originally double

        # print('Entering superglue')
        desc0, desc1 = data["descriptors0"].float(), data["descriptors1"].float()
        kpts0, kpts1 = data["keypoints0"].float(), data["keypoints1"].float()

        # desc0 = desc0.transpose(0, 1)
        # desc1 = desc1.transpose(0, 1)
        # kpts0 = torch.reshape(kpts0, (1, -1, 2))
        # kpts1 = torch.reshape(kpts1, (1, -1, 2))

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
        all_matches = data["all_matches"]
        # all_matches = data["all_matches"].permute(
        #     1, 2, 0
        # )  # shape=torch.Size([1, 87, 2])

        # Keypoint normalization.
        # print(data["image0_shape"], data["image0_shape"].shape)
        kpts0 = normalize_keypoints(kpts0, data["image0_shape"])
        kpts1 = normalize_keypoints(kpts1, data["image1_shape"])
        # print(kpts0)
        # print(kpts0.shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data["scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["scores1"])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        # print('gnn out desc0 desc1', desc0.shape, desc0.shape)

        # Final MLP projection.
        # desc0 = torch.permute(desc0, (0, 2, 1)).float()
        # desc1 = torch.permute(desc1, (0, 2, 1)).float()
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # mdesc0 = torch.permute(mdesc0, (0, 2, 1)).float()
        # mdesc1 = torch.permute(mdesc1, (0, 2, 1)).float()

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / self.config["descriptor_dim"] ** 0.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score, iters=self.config["sinkhorn_iterations"]
        )

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config["match_threshold"])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # check if indexed correctly

        batch_size, num_matches = all_matches.shape[0], all_matches.shape[1]
        total_loss = 0
        # print('batch_size', batch_size)
        batch_loss = []
        for b in range(batch_size):
            loss = []
            for i in range(num_matches):
                x = all_matches[b][i][0]
                y = all_matches[b][i][1]
                if x == -1 and y == -1:  # padding values
                    continue
                loss.append(-torch.log(scores[b][x][y].exp()))
                # loss.append(-torch.log( scores[0][x][y].exp() )) # check batch size == 1 ?
            # for p0 in unmatched0:
            #     loss += -torch.log(scores[0][p0][-1])
            # for p1 in unmatched1:
            #     loss += -torch.log(scores[0][-1][p1])
            # print('loss', loss)
            loss_mean_unreshaped = torch.mean(torch.stack(loss))
            batch_loss.append(loss_mean_unreshaped.item())

            # print('loss_mean_unreshaped', loss_mean_unreshaped)
            # loss_mean = torch.reshape(loss_mean_unreshaped, (1, -1))
            # print('loss_mean', loss_mean)
            total_loss += loss_mean_unreshaped
        # print(batch_loss)
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
