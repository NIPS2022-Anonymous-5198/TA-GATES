# -*- coding: utf-8 -*-
import abc
import copy
import os
import re
import random
import collections
import itertools
import yaml

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import nasbench
from nasbench import api
from nasbench.lib import graph_util, config

from my_nas import utils
from my_nas.utils.exception import expect
from my_nas.common import SearchSpace
from my_nas.utils import DenseGraphConvolution, DenseGraphFlow
from my_nas.evaluator.arch_network import ArchEmbedder


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT_NODE = 6

VERTICES = 7
MAX_EDGES = 9
_nasbench_cfg = config.build_config()


def parent_combinations_old(adjacency_matrix, node, n_parents=2):
    """Get all possible parent combinations for the current node."""
    if node != 1:
        # Parents can only be nodes which have an index that is lower than the current index,
        # because of the upper triangular adjacency matrix and because the index is also a
        # topological ordering in our case.
        return itertools.combinations(np.argwhere(adjacency_matrix[:node, node] == 0).flatten(),
                                      n_parents)  # (e.g. (0, 1), (0, 2), (1, 2), ...
    else:
        return [[0]]


def parent_combinations(node, num_parents):
    if node == 1 and num_parents == 1:
        return [(0,)]
    else:
        return list(itertools.combinations(list(range(int(node))), num_parents))


def upscale_to_nasbench_format(adjacency_matrix):
    """
    The search space uses only 4 intermediate nodes, rather than 5 as used in nasbench
    This method adds a dummy node to the graph which is never used to be compatible with nasbench.
    :param adjacency_matrix:
    :return:
    """
    return np.insert(
        np.insert(adjacency_matrix,
                  5, [0, 0, 0, 0, 0, 0], axis=1),
        5, [0, 0, 0, 0, 0, 0, 0], axis=0)


def _literal_np_array(arr):
    if arr is None:
        return None
    return "np.array({})".format(np.array2string(arr, separator=",").replace("\n", " "))


class _ModelSpec(api.ModelSpec):
    def __repr__(self):
        return "_ModelSpec({}, {}; pruned_matrix={}, pruned_ops={})".format(
            _literal_np_array(self.original_matrix),
            self.original_ops,
            _literal_np_array(self.matrix),
            self.ops,
        )

    def hash_spec(self, *args, **kwargs):
        return super(_ModelSpec, self).hash_spec(_nasbench_cfg["available_ops"])


class NasBench101SearchSpace(SearchSpace):
    NAME = "nasbench-101"

    def __init__(
        self,
        multi_fidelity=False,
        load_nasbench=True,
        compare_reduced=True,
        compare_use_hash=False,
        validate_spec=True,
    ):
        super(NasBench101SearchSpace, self).__init__()

        self.ops_choices = ["conv1x1-bn-relu",
                            "conv3x3-bn-relu", "maxpool3x3", "none"]

        mynas_ops = [
            "conv_bn_relu_1x1",
            "conv_bn_relu_3x3",
            "max_pool_3x3",
            "none",
        ]

        self.op_mapping = {k: v for k, v in zip(self.ops_choices, mynas_ops)}

        self.ops_choice_to_idx = {
            choice: i for i, choice in enumerate(self.ops_choices)
        }

        # operations: "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"
        self.multi_fidelity = multi_fidelity
        self.load_nasbench = load_nasbench
        self.compare_reduced = compare_reduced
        self.compare_use_hash = compare_use_hash

        self.num_vertices = VERTICES
        self.max_edges = MAX_EDGES
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * \
            (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices)  # 3 + 1 (none)
        self.num_ops = self.num_vertices - 2  # 5
        self.idx = np.triu_indices(self.num_vertices, k=1)
        self.validate_spec = validate_spec

        if self.load_nasbench:
            self._init_nasbench()

    def __getstate__(self):
        state = super(NasBench101SearchSpace, self).__getstate__().copy()
        del state["nasbench"]
        return state

    def __setstate__(self, state):
        super(NasBench101SearchSpace, self).__setstate__(state)
        if self.load_nasbench:
            # slow, comment this if do not need to load nasbench API when pickle load from disk
            self._init_nasbench()

    def pad_archs(self, archs):
        return [self._pad_arch(arch) for arch in archs]

    def _pad_arch(self, arch):
        # padding for batchify training
        adj, ops = arch
        # all normalize the the reduced one
        spec = self.construct_modelspec(edges=None, matrix=adj, ops=ops)
        adj, ops = spec.matrix, self.op_to_idx(spec.ops)
        num_v = adj.shape[0]
        if num_v < VERTICES:
            padded_adj = np.concatenate(
                (adj[:-1], np.zeros((VERTICES - num_v, num_v), dtype=np.int8), adj[-1:])
            )
            padded_adj = np.concatenate(
                (
                    padded_adj[:, :-1],
                    np.zeros((VERTICES, VERTICES - num_v)),
                    padded_adj[:, -1:],
                ),
                axis=1,
            )
            padded_ops = ops + [3] * (7 - num_v)
            adj, ops = padded_adj, padded_ops
        return (adj, ops)

    def _random_sample_ori(self):
        while 1:
            matrix = np.random.choice(
                [0, 1], size=(self.num_vertices, self.num_vertices)
            )
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(
                self.ops_choices[:-1], size=(self.num_vertices)
            ).tolist()
            ops[0] = "input"
            ops[-1] = "output"
            spec = _ModelSpec(matrix=matrix, ops=ops)
            if self.validate_spec and not self.nasbench.is_valid(spec):
                continue
            return NasBench101Rollout(
                spec.original_matrix,
                ops=self.op_to_idx(spec.original_ops),
                search_space=self,
            )

    def _random_sample_me(self):
        while 1:
            splits = np.array(
                sorted(
                    [0]
                    + list(
                        np.random.randint(
                            0, self.max_edges + 1, size=self.num_possible_edges - 1
                        )
                    )
                    + [self.max_edges]
                )
            )
            edges = np.minimum(splits[1:] - splits[:-1], 1)
            matrix = self.edges_to_matrix(edges)
            ops = np.random.randint(0, self.num_op_choices, size=self.num_ops)
            rollout = NasBench101Rollout(matrix, ops, search_space=self)
            try:
                self.nasbench._check_spec(rollout.genotype)
            except api.OutOfDomainError:
                # ignore out-of-domain archs (disconnected)
                continue
            else:
                return rollout

    # optional API
    def genotype_from_str(self, genotype_str):
        return eval(genotype_str)
        return eval(re.search("(_ModelSpec\(.+);", genotype_str).group(1) + ")")

    # ---- APIs ----
    def random_sample(self):
        m, ops = self.sample(True)
        if len(ops) < len(m) - 2:
            ops.append("none")
        return NasBench101Rollout(m, [self.ops_choices.index(op) for op in ops], search_space=self)
        return self._random_sample_ori()

    def genotype(self, arch):
        # return the corresponding ModelSpec
        # edges, ops = arch
        matrix, ops = arch
        return self.construct_modelspec(edges=None, matrix=matrix, ops=ops)

    def rollout_from_genotype(self, genotype):
        return NasBench101Rollout(
            genotype.original_matrix,
            ops=self.op_to_idx(genotype.original_ops),
            search_space=self,
        )

    def plot_arch(self, genotypes, filename, label, plot_format="pdf", **kwargs):
        graph = genotypes.visualize()
        graph.format = "pdf"
        graph.render(filename, view=False)
        return filename + ".{}".format(plot_format)

    def distance(self, arch1, arch2):
        pass

    # ---- helpers ----
    def _init_nasbench(self):
        # the arch -> performances dataset
        self.base_dir = os.path.join(
            utils.get_mynas_dir("AWNAS_DATA", "data"), "nasbench-101"
        )
        if self.multi_fidelity:
            self.nasbench = api.NASBench(
                os.path.join(self.base_dir, "nasbench_full.tfrecord")
            )
        else:
            self.nasbench = api.NASBench(
                os.path.join(self.base_dir, "nasbench_only108.tfrecord")
            )

    def edges_to_matrix(self, edges):
        matrix = np.zeros(
            [self.num_vertices, self.num_vertices], dtype=np.int8)
        matrix[self.idx] = edges
        return matrix

    def op_to_idx(self, ops):
        return [
            self.ops_choice_to_idx[op] for op in ops if op not in {"input", "output"}
        ]

    def matrix_to_edges(self, matrix):
        return matrix[self.idx]

    def matrix_to_connection(self, matrix):
        edges = matrix[self.idx].astype(np.bool)
        node_connections = {}
        concat_nodes = []
        for from_, to_ in zip(self.idx[0][edges], self.idx[1][edges]):
            # index of nodes starts with 1 rather than 0
            if to_ < len(matrix) - 1:
                node_connections.setdefault(to_, []).append(from_)
            else:
                if from_ >= len(matrix) - 2:
                    continue
                concat_nodes.append(from_)
        return node_connections, concat_nodes

    def construct_modelspec(self, edges, matrix, ops):
        if matrix is None:
            assert edges is not None
            matrix = self.edges_to_matrix(edges)

        # expect(graph_util.num_edges(matrix) <= self.max_edges,
        #        "number of edges could not exceed {}".format(self.max_edges))

        labeling = [self.ops_choices[op_ind] for op_ind in ops]
        labeling = ["input"] + list(labeling) + ["output"]
        model_spec = _ModelSpec(matrix, labeling)
        return model_spec

    def random_sample_arch(self):
        # not uniform, and could be illegal,
        #   if there is not edge from the INPUT or no edge to the OUTPUT,
        # Just check and reject for now
        return self.random_sample().arch

    def batch_rollouts(self, batch_size, shuffle=True, max_num=None):
        len_ = ori_len_ = len(self.nasbench.fixed_statistics)
        if max_num is not None:
            len_ = min(max_num, len_)
        list_ = list(self.nasbench.fixed_statistics.values())
        indexes = np.arange(ori_len_)
        np.random.shuffle(indexes)
        ind = 0
        while ind < len_:
            end_ind = min(len_, ind + batch_size)
            yield [
                NasBench101Rollout(
                    list_[r_ind]["module_adjacency"],
                    self.op_to_idx(list_[r_ind]["module_operations"]),
                    search_space=self,
                )
                for r_ind in indexes[ind:end_ind]
            ]
            ind = end_ind

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]


# ---- embedders for NASBench-101 ----
# TODO: the multi stage trick could apply for all the embedders
class NasBench101_LSTMSeqEmbedder(ArchEmbedder):
    NAME = "nb101-lstm"

    def __init__(
        self,
        search_space,
        num_hid=100,
        emb_hid=100,
        num_layers=1,
        use_mean=False,
        use_hid=False,
        schedule_cfg=None,
    ):
        super(NasBench101_LSTMSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.emb_hid = emb_hid
        self.use_mean = use_mean
        self.use_hid = use_hid

        self.op_emb = nn.Embedding(
            self.search_space.num_op_choices, self.emb_hid)
        self.conn_emb = nn.Embedding(2, self.emb_hid)

        self.rnn = nn.LSTM(
            input_size=self.emb_hid,
            hidden_size=self.num_hid,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.out_dim = num_hid
        self._triu_indices = np.triu_indices(VERTICES, k=1)

    def forward(self, archs, any_time=False):
        x_1 = np.array([arch[0][self._triu_indices] for arch in archs])
        x_2 = np.array([arch[1] for arch in archs])

        conn_embs = self.conn_emb(torch.LongTensor(
            x_1).to(self.op_emb.weight.device))
        op_embs = self.op_emb(torch.LongTensor(
            x_2).to(self.op_emb.weight.device))
        emb = torch.cat((conn_embs, op_embs), dim=-2)

        out, (h_n, _) = self.rnn(emb)

        if self.use_hid:
            y = h_n[0]
        else:
            if self.use_mean:
                y = torch.mean(out, dim=1)
            else:
                # return final output
                y = out[:, -1, :]
        if any_time:
            y = [y]
        return y


class NasBench101_SimpleSeqEmbedder(ArchEmbedder):
    NAME = "nb101-seq"

    def __init__(self, search_space, use_all_adj_items=False, schedule_cfg=None):
        super(NasBench101_SimpleSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.use_all_adj_items = use_all_adj_items
        self.out_dim = 49 + 5 if use_all_adj_items else 21 + 5
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, archs):
        if self.use_all_adj_items:
            x = np.concatenate(
                [
                    np.array([arch[0].reshape(-1) for arch in archs]),
                    np.array([arch[1] for arch in archs]),
                ],
                axis=-1,
            )
        else:
            triu_indices = np.triu_indices(VERTICES, k=1)
            x_1 = np.array([arch[0][triu_indices] for arch in archs])
            x_2 = np.array([arch[1] for arch in archs])
            x = np.concatenate([x_1, x_2], axis=-1)
        return self._placeholder_tensor.new(x)


class NasBench101ArchEmbedder(ArchEmbedder):
    NAME = "nb101-gcn"

    def __init__(
        self,
        search_space,
        embedding_dim=48,
        hid_dim=48,
        gcn_out_dims=[128, 128],
        gcn_kwargs=None,
        dropout=0.0,
        use_global_node=False,
        use_final_only=False,
        schedule_cfg=None,
    ):
        super(NasBench101ArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.none_op_ind = self.search_space.none_op_ind
        self.embedding_dim = embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_global_node = use_global_node
        self.use_final_only = use_final_only
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices

        self.input_op_emb = nn.Embedding(1, self.embedding_dim)
        # zero is ok
        self.output_op_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)))
        # requires_grad=False)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(
                torch.zeros((1, self.embedding_dim)))

        self.op_emb = nn.Embedding(self.num_op_choices, self.embedding_dim)
        self.x_hidden = nn.Linear(self.embedding_dim, self.hid_dim)

        # init graph convolutions
        self.gcns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphConvolution(
                in_dim, dim, **(gcn_kwargs or {})))
            in_dim = dim
        self.gcns = nn.ModuleList(self.gcns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.weight.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.weight.new(
            [arch[1] for arch in archs]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device=adjs.device)
            tmp_cat = torch.cat(
                (
                    tmp_ones,
                    (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                    tmp_ones,
                ),
                dim=2,
            )
            adjs = torch.cat(
                (
                    torch.cat((adjs, tmp_cat), dim=1),
                    torch.zeros(
                        (adjs.shape[0], self.vertices + 1, 1), device=adjs.device
                    ),
                ),
                dim=2,
            )
        node_embs = self.op_emb(op_inds)  # (batch_size, vertices - 2, emb_dim)
        b_size = node_embs.shape[0]
        if self.use_global_node:
            node_embs = torch.cat(
                (
                    self.input_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    node_embs,
                    self.output_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        else:
            node_embs = torch.cat(
                (
                    self.input_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    node_embs,
                    self.output_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_inds

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        adjs, x, op_inds = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding
            # throw away padded info here
            y = torch.cat(
                (
                    y[:, :-1, :]
                    * (op_inds != self.none_op_ind)[:,
                                                    :, None].to(torch.float32),
                    y[:, -1:, :],
                ),
                dim=1,
            )
            y = torch.mean(y, dim=1)  # average across nodes (bs, god)
        return y


class NasBench101FlowArchEmbedder(ArchEmbedder):
    NAME = "nb101-flow"

    def __init__(
        self,
        search_space,
        op_embedding_dim=48,
        node_embedding_dim=48,
        hid_dim=96,
        gcn_out_dims=[128, 128],
        share_op_attention=False,
        other_node_zero=False,
        gcn_kwargs=None,
        use_bn=False,
        use_global_node=False,
        use_final_only=False,
        input_op_emb_trainable=False,
        dropout=0.0,
        schedule_cfg=None,
    ):
        super(NasBench101FlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.use_global_node = use_global_node
        self.share_op_attention = share_op_attention
        self.input_op_emb_trainable = input_op_emb_trainable
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.none_op_ind

        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        # Maybe separate output node?
        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad=not other_node_zero
        )
        # self.middle_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)
        # # zero is ok
        # self.output_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)

        # the last embedding is the output op emb
        self.input_op_emb = nn.Parameter(
            torch.zeros(1, self.op_embedding_dim),
            requires_grad=self.input_op_emb_trainable,
        )
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(
                torch.zeros((1, self.op_embedding_dim)))
            self.vertices += 1

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(
                self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim if not self.share_op_attention else dim,
                    has_attention=not self.share_op_attention,
                    **(gcn_kwargs or {})
                )
            )
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.vertices))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.new([arch[1] for arch in archs]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device=adjs.device)
            tmp_cat = torch.cat(
                (
                    tmp_ones,
                    (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                    tmp_ones,
                ),
                dim=2,
            )
            adjs = torch.cat(
                (
                    torch.cat((adjs, tmp_cat), dim=1),
                    torch.zeros(
                        (adjs.shape[0], self.vertices, 1), device=adjs.device),
                ),
                dim=2,
            )
        # (batch_size, vertices - 2, op_emb_dim)
        op_embs = self.op_emb(op_inds)
        b_size = op_embs.shape[0]
        # the input one should not be relevant
        if self.use_global_node:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        else:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        node_embs = torch.cat(
            (
                self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                self.other_node_emb.unsqueeze(0).repeat(
                    [b_size, self.vertices - 1, 1]),
            ),
            dim=1,
        )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs, op_inds

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding

            # throw away padded info here
            if self.use_global_node:
                y = torch.cat(
                    (
                        y[:, :-2, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -2:, :],
                    ),
                    dim=1,
                )
            else:
                y = torch.cat(
                    (
                        y[:, :-1, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -1:, :],
                    ),
                    dim=1,
                )

            y = torch.mean(y, dim=1)  # average across nodes (bs, god)
        return y


class NasBench101FbFlowArchEmbedder(ArchEmbedder):
    NAME = "nb101-fbflow"

    def __init__(
            self,
            search_space,
            op_embedding_dim=48,
            node_embedding_dim=48,
            hid_dim=96,
            gcn_out_dims=[128, 128, 128, 128, 128],
            share_op_attention=False,
            other_node_zero=False,
            gcn_kwargs=None,
            use_bn=False,
            use_global_node=False,
            use_final_only=False,
            input_op_emb_trainable=False,
            dropout=0.0,

            ## newly added
            # construction (tagates)
            num_time_steps=2,
            fb_conversion_dims=[128, 128],
            backward_gcn_out_dims=[128, 128, 128, 128, 128],
            updateopemb_method="concat_ofb", # concat_ofb, concat_fb, concat_b
            updateopemb_dims=[128],
            updateopemb_scale=0.1,
            b_use_bn=False,
            # construction (l): concat arch-level zeroshot as l
            concat_arch_zs_as_l_dimension=None,
            concat_l_layer=0,
            # construction (symmetry breaking)
            symmetry_breaking_method=None, # None, "random", "param_zs", "param_zs_add"
            concat_param_zs_as_opemb_dimension=None,
            concat_param_zs_as_opemb_mlp=[64, 128],
            concat_param_zs_as_opemb_scale=0.1,

            # gradident flow configurations
            detach_vinfo=False,
            updateopemb_detach_opemb=True,
            updateopemb_detach_finfo=True,

            mask_nonparametrized_ops=False,
            schedule_cfg=None,
    ):
        super(NasBench101FbFlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.use_global_node = use_global_node
        self.share_op_attention = share_op_attention
        self.input_op_emb_trainable = input_op_emb_trainable
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.none_op_ind

        # newly added
        self.detach_vinfo = detach_vinfo
        self.num_time_steps = num_time_steps
        self.fb_conversion_dims = fb_conversion_dims
        self.backward_gcn_out_dims = backward_gcn_out_dims
        self.b_use_bn = b_use_bn
        self.updateopemb_method = updateopemb_method
        self.updateopemb_detach_opemb = updateopemb_detach_opemb
        self.updateopemb_detach_finfo = updateopemb_detach_finfo
        self.updateopemb_dims = updateopemb_dims
        self.updateopemb_scale = updateopemb_scale
        # concat arch-level zs as l
        self.concat_arch_zs_as_l_dimension = concat_arch_zs_as_l_dimension
        self.concat_l_layer = concat_l_layer
        if self.concat_arch_zs_as_l_dimension is not None:
            assert self.concat_l_layer < len(self.fb_conversion_dims)
        # symmetry breaking
        self.symmetry_breaking_method = symmetry_breaking_method
        self.concat_param_zs_as_opemb_dimension = concat_param_zs_as_opemb_dimension
        assert self.symmetry_breaking_method in {None, "param_zs", "random", "param_zs_add"}
        self.concat_param_zs_as_opemb_scale = concat_param_zs_as_opemb_scale

        if self.symmetry_breaking_method == "param_zs_add":
            in_dim = self.concat_param_zs_as_opemb_dimension
            self.param_zs_embedder = []
            for embedder_dim in concat_param_zs_as_opemb_mlp:
                self.param_zs_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.param_zs_embedder.append(nn.ReLU(inplace=False))
                in_dim = embedder_dim
            self.param_zs_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.param_zs_embedder = nn.Sequential(*self.param_zs_embedder)
        
        self.mask_nonparametrized_ops = mask_nonparametrized_ops
        
        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        # Maybe separate output node?
        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad=not other_node_zero
        )
        # self.middle_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)
        # # zero is ok
        # self.output_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)

        # the last embedding is the output op emb
        self.input_op_emb = nn.Parameter(
            torch.zeros(1, self.op_embedding_dim),
            requires_grad=self.input_op_emb_trainable,
        )
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(
                torch.zeros((1, self.op_embedding_dim)))
            self.vertices += 1

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(
                self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else\
                    (self.op_embedding_dim if not self.share_op_attention else dim),
                    has_attention=not self.share_op_attention,
                    **(gcn_kwargs or {})
                )
            )
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.vertices))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

        # init backward graph convolutions
        self.b_gcns = []
        self.b_bns = []
        if self.concat_arch_zs_as_l_dimension is not None \
           and self.concat_l_layer == len(self.fb_conversion_dims) - 1:
            in_dim = self.fb_conversion_dims[-1] + self.concat_arch_zs_as_l_dimension
        else:
            in_dim = self.fb_conversion_dims[-1]
        for dim in self.backward_gcn_out_dims:
            self.b_gcns.append(
                DenseGraphFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else self.op_embedding_dim,
                    has_attention=not self.share_op_attention,
                    reverse=True,
                    **(gcn_kwargs or {})
                )
            )
            in_dim = dim
            if self.b_use_bn:
                self.b_bns.append(nn.BatchNorm1d(self.vertices))
        self.b_gcns = nn.ModuleList(self.b_gcns)
        if self.b_use_bn:
            self.b_bns = nn.ModuleList(self.b_bns)
        self.num_b_gcn_layers = len(self.b_gcns)

        # init the network to convert forward output info into backward input info
        if self.num_time_steps > 1:
            self.fb_conversion_list = []
            dim = self.gcn_out_dims[-1]
            num_fb_layers = len(fb_conversion_dims)
            self._num_before_concat_l = None
            for i_dim, fb_conversion_dim in enumerate(fb_conversion_dims):
                self.fb_conversion_list.append(nn.Linear(dim, fb_conversion_dim))
                if i_dim < num_fb_layers - 1:
                    self.fb_conversion_list.append(nn.ReLU(inplace=False))
                if self.concat_arch_zs_as_l_dimension is not None and \
                   self.concat_l_layer == i_dim:
                    dim = fb_conversion_dim + self.concat_arch_zs_as_l_dimension
                    self._num_before_concat_l = len(self.fb_conversion_list)
                else:
                    dim = fb_conversion_dim
            self.fb_conversion = nn.Sequential(*self.fb_conversion_list)

            # init the network to get delta op_emb
            if self.updateopemb_method == "concat_ofb":
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1] \
                         + self.op_embedding_dim
            elif self.updateopemb_method == "concat_fb":
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1]
            elif self.updateopemb_method == "concat_b":
                in_dim = self.backward_gcn_out_dims[-1]
            else:
                raise NotImplementedError()

            self.updateop_embedder = []
            for embedder_dim in self.updateopemb_dims:
                self.updateop_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.updateop_embedder.append(nn.ReLU(inplace=False))
                in_dim = embedder_dim
            self.updateop_embedder.append(nn.Linear(in_dim, self.op_embedding_dim)) 
            self.updateop_embedder = nn.Sequential(*self.updateop_embedder)

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.new([arch[1] for arch in archs]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device=adjs.device)
            tmp_cat = torch.cat(
                (
                    tmp_ones,
                    (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                    tmp_ones,
                ),
                dim=2,
            )
            adjs = torch.cat(
                (
                    torch.cat((adjs, tmp_cat), dim=1),
                    torch.zeros(
                        (adjs.shape[0], self.vertices, 1), device=adjs.device),
                ),
                dim=2,
            )
        # (batch_size, vertices - 2, op_emb_dim)
        op_embs = self.op_emb(op_inds)
        b_size = op_embs.shape[0]
        # the input one should not be relevant
        if self.use_global_node:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        else:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        node_embs = torch.cat(
            (
                self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                self.other_node_emb.unsqueeze(0).repeat(
                    [b_size, self.vertices - 1, 1]),
            ),
            dim=1,
        )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs, op_inds

    def _forward_pass(self, x, adjs, auged_op_emb) -> Tensor:
        # --- forward pass ---
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, auged_op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        
        return y
    
    def _backward_pass(self, y, adjs, zs_as_l, auged_op_emb) -> Tensor:
        # --- backward pass ---
        # get the information of the output node
        # b_info = torch.cat(
        #     (
        #         torch.zeros([y.shape[0], self.vertices - 1, y.shape[-1]], device=y.device),
        #         y[:, -1:, :]
        #     ),
        #     dim=1
        # )
        b_info = y[:, -1:, :]
        if self.detach_vinfo:
            b_info = b_info.detach()
        if self.concat_arch_zs_as_l_dimension:
            # process before concat l
            b_info = self.fb_conversion[:self._num_before_concat_l](b_info)
            # concat l
            b_info = torch.cat((b_info, zs_as_l.unsqueeze(-2)), dim=-1)
            if not self.concat_l_layer == len(self.fb_conversion_list) - 1:
                # process after concat l
                b_info = self.fb_conversion[self._num_before_concat_l:](b_info)
        else:
            b_info = self.fb_conversion(b_info)
        b_info = torch.cat(
            (
                torch.zeros([y.shape[0], self.vertices - 1, b_info.shape[-1]], device=y.device),
                b_info
            ),
            dim=1
        )

        # start backward flow
        b_adjs = adjs.transpose(1, 2)
        b_y = b_info
        for i_layer, gcn in enumerate(self.b_gcns):
            b_y = gcn(b_y, b_adjs, auged_op_emb)
            if self.b_use_bn:
                shape_y = b_y.shape
                b_y = self.b_bns[i_layer](b_y.reshape(shape_y[0], -1, shape_y[-1]))\
                            .reshape(shape_y)
            if i_layer != self.num_b_gcn_layers - 1:
                b_y = F.relu(b_y)
                b_y = F.dropout(b_y, self.dropout, training=self.training)
        
        return b_y

    def _update_op_emb(self, y, b_y, op_emb, concat_op_emb_mask) -> Tensor:
        # --- UpdateOpEmb ---
        if self.updateopemb_method == "concat_ofb":
            in_embedding = torch.cat(
                (
                    op_emb.detach() if self.updateopemb_detach_opemb else op_emb,
                    y.detach() if self.updateopemb_detach_finfo else y,
                    b_y
                ),
                dim=-1)
        elif self.updateopemb_method == "concat_fb":
            in_embedding = torch.cat(
                (
                    y.detach() if self.updateopemb_detach_finfo else y,
                    b_y
                ), dim=-1)
        elif self.updateopemb_method == "concat_b":
            in_embedding = b_y
        update = self.updateop_embedder(in_embedding)

        if self.mask_nonparametrized_ops:
            update = update * concat_op_emb_mask

        op_emb = op_emb + self.updateopemb_scale * update
        return op_emb

    def _final_process(self, y: Tensor, op_inds) -> Tensor:
        ## --- output ---
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding

            # throw away padded info here
            if self.use_global_node:
                y = torch.cat(
                    (
                        y[:, :-2, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -2:, :],
                    ),
                    dim=1,
                )
            else:
                y = torch.cat(
                    (
                        y[:, :-1, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -1:, :],
                    ),
                    dim=1,
                )

            y = torch.mean(y, dim=1)  # average across nodes (bs, god)
        
        return y

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        if isinstance(archs, tuple):
            if len(archs) == 2:
                archs, zs_as_p = archs
                zs_as_l = None
            elif len(archs) == 3:
                archs, zs_as_l, zs_as_p = archs
            else:
                raise Exception()
        else:
            zs_as_l = zs_as_p = None

        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        concat_op_emb_mask = ((op_inds == 0) | (op_inds == 1))
        concat_op_emb_mask = F.pad(concat_op_emb_mask, (1, 1), mode="constant")
        concat_op_emb_mask = concat_op_emb_mask.unsqueeze(-1).to(torch.float32)

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_emb).normal_() * 0.1
            op_emb = op_emb + noise
        elif self.symmetry_breaking_method == "param_zs":
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            assert zs_as_p.shape[-1] == self.concat_param_zs_as_opemb_dimension
        elif self.symmetry_breaking_method == "param_zs_add":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            zs_as_p = self.param_zs_embedder(zs_as_p)
            op_emb = op_emb + zs_as_p * self.concat_param_zs_as_opemb_scale
        #elif self.symmetry_breaking_method == "param_zs_add":
        #    zs_as_p = self.input_op_emb.new(np.array(zs_as_p))


        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)

        for t in range(self.num_time_steps):
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_emb = torch.cat((op_emb, zs_as_p), dim=-1)
            else:
                auged_op_emb = op_emb

            y = self._forward_pass(x, adjs, auged_op_emb)
       
            if t == self.num_time_steps - 1:
                break

            b_y = self._backward_pass(y, adjs, zs_as_l, auged_op_emb)
            op_emb = self._update_op_emb(y, b_y, op_emb, concat_op_emb_mask)

        ## --- output ---
        # y: (batch_size, vertices, gcn_out_dims[-1])
        return self._final_process(y, op_inds)


class NasBench101FbFlowAnyTimeArchEmbedder(NasBench101FbFlowArchEmbedder):
    NAME = "nb101-fbflow-anytime"

    def forward(self, archs, any_time: bool = False):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        if not any_time:
            return super(NasBench101FbFlowAnyTimeArchEmbedder, self).forward(archs)

        if isinstance(archs, tuple):
            if len(archs) == 2:
                archs, zs_as_p = archs
                zs_as_l = None
            elif len(archs) == 3:
                archs, zs_as_l, zs_as_p = archs
            else:
                raise Exception()
        else:
            zs_as_l = zs_as_p = None

        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        concat_op_emb_mask = ((op_inds == 0) | (op_inds == 1))
        concat_op_emb_mask = F.pad(concat_op_emb_mask, (1, 1), mode="constant")
        concat_op_emb_mask = concat_op_emb_mask.unsqueeze(-1).to(torch.float32)
        
        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_emb).normal_() * 0.1
            op_emb = op_emb + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            assert zs_as_p.shape[-1] == self.concat_param_zs_as_opemb_dimension
        elif self.symmetry_breaking_method == "param_zs_add":
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            zs_as_p = self.param_zs_embedder(zs_as_p)
            op_emb = op_emb + zs_as_p * self.concat_param_zs_as_opemb_scale

        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)

        y_cache = []

        for t in range(self.num_time_steps):
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_emb = torch.cat((op_emb, zs_as_p), dim=-1)
            else:
                auged_op_emb = op_emb

            y = self._forward_pass(x, adjs, auged_op_emb)
            y_cache.append(self._final_process(y, op_inds))
       
            if t == self.num_time_steps - 1:
                break

            b_y = self._backward_pass(y, adjs, zs_as_l, auged_op_emb)
            op_emb = self._update_op_emb(y, b_y, op_emb, concat_op_emb_mask)
        
        return y_cache
