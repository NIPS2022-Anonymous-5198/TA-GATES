import os
import re
import copy
import random
import pickle
import itertools
import collections
from typing import List, Optional, NamedTuple
from collections import defaultdict, OrderedDict
import contextlib

import six
import yaml
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from nas_201_api import NASBench201API as API

from my_nas import utils, ops
from my_nas.common import SearchSpace
from my_nas.evaluator.arch_network import ArchEmbedder
from my_nas.utils import (
    DenseGraphSimpleOpEdgeFlow,
    DenseGraphConvolution
)

VERTICES = 4


class NasBench201SearchSpace(SearchSpace):
    NAME = "nasbench-201"

    def __init__(
        self,
        num_layers=17,
        vertices=4,
        load_nasbench=True,
        ops_choices=(
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
        ),
    ):
        super(NasBench201SearchSpace, self).__init__()

        self.ops_choices = ops_choices
        self.ops_choice_to_idx = {
            choice: i for i, choice in enumerate(self.ops_choices)
        }

        self.load_nasbench = load_nasbench
        self.num_vertices = vertices
        self.num_layers = num_layers
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices)  # 5
        self.num_ops = self.num_vertices * (self.num_vertices - 1) // 2
        self.idx = np.tril_indices(self.num_vertices, k=-1)
        self.genotype_type = str

        if self.load_nasbench:
            self._init_nasbench()

    def canonicalize(self, rollout):
        # TODO
        arch = rollout.arch
        num_vertices = rollout.search_space.num_vertices
        op_choices = rollout.search_space.ops_choices
        S = []
        S.append("0")
        res = ""
        for i in range(1, num_vertices):
            preS = []
            s = ""
            for j in range(i):
                if ((int(arch[i][j]) == 0) or (S[j] == "#")):
                    s = "#"
                elif (int(arch[i][j]) == 1):
                    s = S[j]
                else:
                    s = "(" + S[j] + ")" + "@" + op_choices[int(arch[i][j])]
                preS.append(s)
            preS.sort()
            s = ""
            for j in range(i):
                s = s + preS[j]
            S.append(s)
            res = s
        return res

    def __getstate__(self):
        state = super(NasBench201SearchSpace, self).__getstate__().copy()
        if "api" in state:
            del state["api"]
        return state

    def __setstate__(self, state):
        super(NasBench201SearchSpace, self).__setstate__(state)
        if self.load_nasbench:
            self._init_nasbench()

    # optional API
    def genotype_from_str(self, genotype_str):
        return genotype_str

    # ---- APIs ----
    def random_sample(self):
        return NasBench201Rollout(self.random_sample_arch(), search_space=self)

    def genotype(self, arch):
        # return the corresponding ModelSpec
        # edges, ops = arch
        return self.matrix2str(arch)

    def rollout_from_genotype(self, genotype):
        return NasBench201Rollout(API.str2matrix(genotype), search_space=self)

    def plot_arch(self, genotypes, filename, label, plot_format="pdf", **kwargs):
        matrix = self.str2matrix(genotypes)

        from graphviz import Digraph

        graph = Digraph(
            format=plot_format,
            # https://stackoverflow.com/questions/4714262/graphviz-dot-captions
            body=['label="{l}"'.format(l=label), "labelloc=top", "labeljust=left"],
            edge_attr=dict(fontsize="20", fontname="times"),
            node_attr=dict(
                style="filled",
                shape="rect",
                align="center",
                fontsize="20",
                height="0.5",
                width="0.5",
                penwidth="2",
                fontname="times",
            ),
            engine="dot",
        )
        graph.body.extend(["rankdir=LR"])
        graph.node(str(0), fillcolor="darkseagreen2")
        graph.node(str(self.num_vertices - 1), fillcolor="palegoldenrod")
        [
            graph.node(str(i), fillcolor="lightblue")
            for i in range(1, self.num_vertices - 1)
        ]

        for to_, from_ in zip(*self.idx):
            op_name = self.ops_choices[int(matrix[to_, from_])]
            if op_name == "none":
                continue
            graph.edge(str(from_), str(to_), label=op_name, fillcolor="gray")

        graph.render(filename, view=False)
        fnames = []
        fnames.append(("cell", filename + ".{}".format(plot_format)))
        return fnames

    def distance(self, arch1, arch2):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201", "nasbench-201-differentiable"]

    def mutate(self, rollout):  # pylint: disable=arguments-differ
        rand_ind = np.random.randint(0, self.idx[0].shape[0])
        neighbor_choice = np.random.randint(0, self.num_op_choices)
        arch_mat = rollout.arch
        while neighbor_choice == arch_mat[self.idx[0][rand_ind], self.idx[1][rand_ind]]:
            neighbor_choice = np.random.randint(0, self.num_op_choices)
        new_arch_mat = copy.deepcopy(arch_mat)
        new_arch_mat[self.idx[0][rand_ind], self.idx[1][rand_ind]] = neighbor_choice
        return NasBench201Rollout(new_arch_mat, self)

    # ---- helpers ----
    def matrix2str(self, arch):
        node_strs = []
        for i_node in range(1, self.num_vertices):
            node_strs.append(
                "|"
                + "|".join(
                    [
                        "{}~{}".format(
                            self.ops_choices[int(arch[i_node, i_input])], i_input
                        )
                        for i_input in range(0, i_node)
                    ]
                )
                + "|"
            )
        return "+".join(node_strs)

    def str2matrix(self, str_):
        arch = np.zeros((self.num_vertices, self.num_vertices))
        split_str = str_.split("+")
        for ind, s in enumerate(split_str):
            geno = [name for name in s.split("|") if name != ""]
            for g in geno:
                name, conn = g.split("~")
                to_ = ind + 1
                from_ = int(conn)
                arch[to_][from_] = self.ops_choices.index(name)
        return arch

    def _init_nasbench(self):
        # the arch -> performances dataset
        self.base_dir = os.path.join(
            utils.get_mynas_dir("AWNAS_DATA", "data"), "nasbench-201"
        )
        self.api = API(os.path.join(self.base_dir, "NAS-Bench-201-v1_0-e61699.pth"))

    def op_to_idx(self, ops):
        return [self.ops_choice_to_idx[op] for op in ops]

    def random_sample_arch(self):
        arch = np.zeros((self.num_vertices, self.num_vertices))
        arch[np.tril_indices(self.num_vertices, k=-1)] = np.random.randint(
            low=0, high=self.num_op_choices, size=self.num_ops
        )
        return arch

    def batch_rollouts(self, batch_size, shuffle=True, max_num=None):
        len_ = ori_len_ = len(self.api)
        if max_num is not None:
            len_ = min(max_num, len_)
        indexes = np.arange(ori_len_)
        np.random.shuffle(indexes)
        ind = 0
        while ind < len_:
            end_ind = min(len_, ind + batch_size)
            yield [
                NasBench201Rollout(
                    matrix=self.api.str2matrix(self.api.arch(r_ind)), search_space=self
                )
                for r_ind in indexes[ind:end_ind]
            ]
            ind = end_ind


# ---- embedders for NASBench-201 ----
class NasBench201_LineGraphEmbedder(ArchEmbedder):
    NAME = "nb201-linegcn"

    def __init__(
        self,
        search_space,
        op_embedding_dim=48,
        hid_dim=96,
        gcn_out_dims=[128, 128],
        dropout=0.0,
        gcn_kwargs=None,
        use_bn=False,
        use_cat=False,
        schedule_cfg=None,
    ):
        super(NasBench201_LineGraphEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.use_bn = use_bn
        self.dropout = dropout
        self.use_cat = use_cat

        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices

        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.x_hidden = nn.Linear(self.op_embedding_dim, self.hid_dim)

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphConvolution(in_dim, dim, **(gcn_kwargs or {})))
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.vertices))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim * (1 if not self.use_cat else 6)

        adj = torch.tensor(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                ],
                dtype=np.float32,
            )
        )
        self.register_buffer("adj", adj)
        self.idx = list(zip(*[[1, 0], [2, 0], [3, 0], [2, 1], [3, 1], [3, 2]]))

    def embed_and_transform_arch(self, archs):
        op_inds = self.op_emb.weight.new([arch[self.idx] for arch in archs]).long()
        embs = self.op_emb(op_inds)  # batch_size x 6 x op_embedding_dim
        b_size = embs.shape[0]
        x = self.x_hidden(embs)
        adjs = self.adj.unsqueeze(0).repeat([b_size, 1, 1])
        return adjs, x

    def forward(self, archs):
        adjs, x = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        return y.reshape(y.shape[0], -1) if self.use_cat else torch.mean(y, dim=1)


class NasBench201_LSTMSeqEmbedder(ArchEmbedder):
    NAME = "nb201-lstm"

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
        super(NasBench201_LSTMSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.emb_hid = emb_hid
        self.use_mean = use_mean
        self.use_hid = use_hid

        self.op_emb = nn.Embedding(self.search_space.num_op_choices, self.emb_hid)

        self.rnn = nn.LSTM(
            input_size=self.emb_hid,
            hidden_size=self.num_hid,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.out_dim = num_hid
        self._tril_indices = np.tril_indices(self.search_space.num_vertices, k=-1)

    def forward(self, archs):
        x = [arch[self._tril_indices] for arch in archs]
        embs = self.op_emb(torch.LongTensor(x).to(self.op_emb.weight.device))
        out, (h_n, _) = self.rnn(embs)

        if self.use_hid:
            y = h_n[0]
        else:
            if self.use_mean:
                y = torch.mean(out, dim=1)
            else:
                # return final output
                y = out[:, -1, :]
        return y


class NasBench201_SimpleSeqEmbedder(ArchEmbedder):
    NAME = "nb201-seq"

    def __init__(self, search_space, schedule_cfg=None):
        super(NasBench201_SimpleSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.out_dim = self.search_space.num_ops
        self.num_node = self.search_space.num_vertices

        self._tril_indices = np.tril_indices(self.num_node, k=-1)
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, archs):
        x = [arch[self._tril_indices] for arch in archs]
        return self._placeholder_tensor.new(x)


class NasBench201FlowArchEmbedder(ArchEmbedder):
    NAME = "nb201-flow"

    def __init__(
        self,
        search_space,
        op_embedding_dim=48,
        node_embedding_dim=48,
        hid_dim=96,
        gcn_out_dims=[128, 128],
        share_op_attention=False,
        gcn_kwargs=None,
        use_bn=False,
        use_final_only=False,
        reverse=False,
        share_self_op_emb=False,
        dropout=0.0,
        init_input_node_emb=True,
        schedule_cfg=None,
    ):
        super(NasBench201FlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.share_op_attention = share_op_attention
        self.share_self_op_emb = share_self_op_emb
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.none_op_ind
        self.reverse = reverse # a toy exp
        self.init_input_node_emb = init_input_node_emb

        if self.init_input_node_emb:
            self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        else:
            self.input_node_emb = None

        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad=False
        )

        # the last embedding is the output op emb
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        if self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(
                torch.FloatTensor(self.op_embedding_dim).normal_()
            )
        else:
            self.self_op_emb = None

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphSimpleOpEdgeFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim if not self.share_op_attention else dim,
                    has_attention=not self.share_op_attention,
                    reverse=self.reverse,
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

    def embed_and_transform_arch(self, archs, input_node_emb):
        adjs = self.op_emb.weight.new(archs).long()
        op_embs = self.op_emb(adjs)  # (batch_size, vertices, vertices, op_emb_dim)
        b_size = op_embs.shape[0]
        if self.init_input_node_emb:
            node_embs = torch.cat(
                (
                    self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1]),
                ),
                dim=1,
            )
        else:
            node_embs = torch.cat(
                (
                    input_node_emb.unsqueeze(-2),
                    self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1]),
                ),
                dim=1,
            )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs

    def forward(self, archs, input_node_emb=None):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        adjs, x, op_embs = self.embed_and_transform_arch(archs, input_node_emb)
        y = x
        if self.reverse:
            adjs = adjs.transpose(1, 2)
            x = torch.cat(
                (
                    x[:, 1:, :],
                    x[:, 0:1, :],
                ), dim=1)
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, op_embs, self_op_emb=self.self_op_emb)
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
            if not self.reverse:
                y = y[:, -1, :]
            else:
                y = y[:, 0, :]
        else:
            if not self.reverse:
                y = y[:, 1:, :]  # do not keep the inputs node embedding
            else:
                y = y[:, :-1, :]
            y = torch.mean(y, dim=1)  # average across nodes (bs, god)
        return y


class NB201FBFlowArchEmbedder(ArchEmbedder):
    NAME = "nb201-fbflow"

    def __init__(
            self,
            search_space,
            op_embedding_dim=48,
            node_embedding_dim=48,
            hid_dim=96,
            gcn_out_dims=[128, 128, 128],
            share_op_attention=False,
            gcn_kwargs=None,
            use_bn=False,
            use_final_only=False,
            share_self_op_emb=False,
            dropout=0.0,
            init_input_node_emb=True,

            ## newly added
            # construction configurations
            # construction (tagates)
            num_time_steps=2,
            fb_conversion_dims=[128, 128],
            backward_gcn_out_dims=[128, 128, 128],
            updateopemb_method="concat_ofb_message", # concat_ofb_message, concat_ofb
            updateopemb_scale=0.1,
            updateopemb_dims=[128],
            b_use_bn=False,
            # construction (l): concat arch-level zeroshot as l
            concat_arch_zs_as_l_dimension=None,
            concat_l_layer=0,
            # construction (symmetry breaking)
            symmetry_breaking_method=None, # None, "random", "param_zs", "param_zs_add"
            concat_param_zs_as_opemb_dimension=None,
            concat_param_zs_as_opemb_mlp = [64, 128],
            param_zs_add_coeff = 1.,

            # gradient flow configurations
            detach_vinfo=False,
            updateopemb_detach_opemb=True,
            updateopemb_detach_finfo=True,

            schedule_cfg=None,
    ):
        super(NB201FBFlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.share_op_attention = share_op_attention
        self.share_self_op_emb = share_self_op_emb
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.none_op_ind
        self.init_input_node_emb = init_input_node_emb

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

        if self.symmetry_breaking_method == "param_zs_add":
            in_dim = self.concat_param_zs_as_opemb_dimension
            self.param_zs_embedder = []
            for embedder_dim in concat_param_zs_as_opemb_mlp:
                self.param_zs_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.param_zs_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.param_zs_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.param_zs_embedder = nn.Sequential(*self.param_zs_embedder)
            self.param_zs_add_coeff = param_zs_add_coeff

        ## --- init GATES parts ---
        if self.init_input_node_emb:
            self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        else:
            self.input_node_emb = None

        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad=False
        )

        # the last embedding is the output op emb
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        if self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(
                torch.FloatTensor(self.op_embedding_dim).normal_()
            )
        else:
            self.self_op_emb = None

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(self.op_embedding_dim, self.gcn_out_dims[0])

        if self.num_time_steps > 1 and "message" in self.updateopemb_method:
            addi_kwargs = {"return_message": True}
            self.use_message = True
        else:
            addi_kwargs = {}
            self.use_message = False

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        gcn_kwargs = copy.deepcopy(gcn_kwargs) if gcn_kwargs is not None else {}
        gcn_kwargs.update(addi_kwargs)
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphSimpleOpEdgeFlow(
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

        # --- init TA-GATES parts ---
        if self.num_time_steps > 1:
            # for caculating parameterized op mask, only update the emb of parametrized operations
            self._parametrized_op_emb = [
                [float("conv" in op_name)] for op_name in self.search_space.ops_choices]
            print("is parametrized op: ", self._parametrized_op_emb)
            self._parametrized_op_emb = nn.Parameter(
                torch.tensor(self._parametrized_op_emb, dtype=torch.float32), requires_grad=False)

            # init backward graph convolutions
            self.b_gcns = []
            self.b_bns = []
            if self.concat_arch_zs_as_l_dimension is not None \
               and self.concat_l_layer == self.fb_conversion_dims - 1:
                in_dim = self.fb_conversion_dims[-1] + self.concat_arch_zs_as_l_dimension
            else:
                in_dim = self.fb_conversion_dims[-1]
            b_gcn_kwargs = copy.deepcopy(gcn_kwargs) if gcn_kwargs is not None else {}
            b_gcn_kwargs.update(addi_kwargs)
            # the final output node (concated all internal nodes in DARTS & NB301)
            for dim in self.backward_gcn_out_dims:
                self.b_gcns.append(DenseGraphSimpleOpEdgeFlow(
                    in_dim, dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else self.op_embedding_dim,
                    reverse=True,
                    **(b_gcn_kwargs or {})))
                in_dim = dim
                if self.use_bn:
                    self.b_bns.append(nn.BatchNorm1d(self.vertices))
            self.b_gcns = nn.ModuleList(self.b_gcns)
            if self.b_use_bn:
                self.b_bns = nn.ModuleList(self.b_bns)
            self.num_b_gcn_layers = len(self.b_gcns)

            # init the network to convert forward output info into backward input info
            self.fb_conversion_list = []
            # concat the embedding all cell groups, and then do the f-b conversion
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
            if self.updateopemb_method in {"concat_ofb", "concat_ofb_message"}:
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1] \
                         + self.op_embedding_dim
            # elif self.updateopemb_method == "concat_ofb_message":
            # # elif self.updateopemb_method == "concat_fb":
            # #     in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1]
            # # elif self.updateopemb_method == "concat_b":
            # #     in_dim = self.backward_gcn_out_dims[-1]
            else:
                raise NotImplementedError()

            self.updateop_embedder = []
            for embedder_dim in self.updateopemb_dims:
                self.updateop_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.updateop_embedder.append(nn.ReLU(inplace=False))
                in_dim = embedder_dim
            self.updateop_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.updateop_embedder = nn.Sequential(*self.updateop_embedder)

    def embed_and_transform_arch(self, archs, input_node_emb):
        adjs = self.op_emb.weight.new(archs).long()
        op_embs = self.op_emb(adjs)  # (batch_size, vertices, vertices, op_emb_dim)
        b_size = op_embs.shape[0]
        if self.init_input_node_emb:
            node_embs = torch.cat(
                (
                    self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1]),
                ),
                dim=1,
            )
        else:
            node_embs = torch.cat(
                (
                    input_node_emb.unsqueeze(-2),
                    self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1]),
                ),
                dim=1,
            )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs

    # def forward(self, archs, zs_as_l=None, zs_as_p=None, input_node_emb=None):
    def forward(self, archs, input_node_emb=None):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        if isinstance(archs, tuple):
            if len(archs) == 2:
                archs, zs_as_l = archs
                zs_as_p = None
            elif len(archs) == 3:
                archs, zs_as_l, zs_as_p = archs
            else:
                raise Exception()
        else:
            zs_as_l = zs_as_p = None

        adjs, x, op_embs = self.embed_and_transform_arch(archs, input_node_emb)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_embs).normal_() * 0.1
            op_embs = op_embs + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = self.op_emb.weight.new(zs_as_p)
        elif self.symmetry_breaking_method == "param_zs_add":
            zs_as_p = self.op_emb.weight.new(zs_as_p)
            op_embs = op_embs + self.param_zs_add_coeff * self.param_zs_embedder(zs_as_p)

        if self.num_time_steps > 1:
            # calculate op mask
            opemb_update_mask = F.embedding(adjs, self._parametrized_op_emb)
        else:
            opemb_update_mask = None

        for t in range(self.num_time_steps):
            # concat zeroshot onto the op embedding for forward and backward
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_embs = torch.cat((op_embs, zs_as_p), dim=-1)
            else:
                auged_op_embs = op_embs

            y, message = self._forward_pass(x, adjs, auged_op_embs)

            if t == self.num_time_steps - 1:
                break

            b_y, b_message = self._backward_pass(y, adjs, zs_as_l, auged_op_embs) 
            op_embs = self._update_op_emb(y, b_y, op_embs, message, b_message, opemb_update_mask)
        
        y = self._final_process(y)
        return y

    def _backward_pass(self, y, adjs, zs_as_l, auged_op_embs) -> Tensor:
        # --- backward pass ---
        b_info = y[:, -1:, :]
        if self.detach_vinfo:
            b_info = b_info.detach()
        if self.concat_arch_zs_as_l_dimension:
            # process before concat l
            b_info = self.fb_conversion[:self._num_before_concat_l](b_info)
            # concat l
            b_info = torch.cat((b_info, zs_as_l.unsqueeze(-2)), dim=-1)
            if not self.concat_l_layer == len(self.fb_converseion_list) - 1:
                # process after concat l
                b_info = self.fb_conversion[self._num_before_concat_l:](b_info)
        else:
            b_info = self.fb_conversion(b_info)
        b_info = torch.cat(
            (
                torch.zeros([y.shape[0], self.vertices - 1, b_info.shape[-1]], device=y.device),
                b_info
            ), dim=1
        )

        # start backward flow
        b_message = None
        b_adjs = adjs.transpose(1, 2)
        b_y = b_info
        b_op_embs = auged_op_embs.transpose(1, 2)
        for i_layer, gcn in enumerate(self.b_gcns):
            if self.use_message:
                b_y, b_message = gcn(b_y, b_adjs, b_op_embs, self_op_emb=self.self_op_emb)
            else:
                b_y = gcn(b_y, b_adjs, b_op_embs, self_op_emb=self.self_op_emb)
            if self.use_bn:
                shape_y = b_y.shape
                b_y = self.bns[i_layer](b_y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                b_y = F.relu(b_y)
            b_y = F.dropout(b_y, self.dropout, training=self.training)
        return b_y, b_message


    def _forward_pass(self, x, adjs, auged_op_embs) -> Tensor:
        y = x
        message = None
        for i_layer, gcn in enumerate(self.gcns):
            if self.use_message:
                y, message = gcn(y, adjs, auged_op_embs, self_op_emb = self.self_op_emb)
            else:
                y = gcn(y, adjs, auged_op_embs, self_op_emb = self.self_op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training = self.training)
        
        return y, message
    
    def _update_op_emb(self, y: Tensor, b_y: Tensor, op_embs: Tensor, message: Tensor, b_message: Tensor, opemb_update_mask: Tensor) -> Tensor:
        # --- UpdateOpEmb ---
        if self.updateopemb_method == "concat_ofb":
            unsqueeze_y = y.unsqueeze(-2).repeat([1, 1, self.vertices, 1])
            unsqueeze_b_y = b_y.unsqueeze(-3).repeat([1, self.vertices, 1, 1])
            in_embedding = torch.cat(
                    (
                        op_embs.detach() if self.updateopemb_detach_opemb else op_embs,
                        unsqueeze_y.detach() if self.updateopemb_detach_finfo else unsqueeze_y,
                        unsqueeze_b_y
                    ), dim = -1
            )
        elif self.updateopemb_method == "concat_ofb_message": # use_message==True
            in_embedding = torch.cat(
                    (
                        op_embs.detach() if self.updateopemb_detach_opemb else op_embs,
                        message.detach() if self.updateopemb_detach_finfo else message,
                        b_message.transpose(1, 2)
                    ), dim = -1
            )
        else:
            raise Exception()
        
        update = self.updateop_embedder(in_embedding)
        update = update * opemb_update_mask
        op_embs = op_embs + self.updateopemb_scale * update
        return op_embs

    def _final_process(self, y: Tensor) -> Tensor:
        # ---- final output ----
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding
            y = torch.mean(y, dim=1)  # average across nodes (bs, god)
        return y


class NB201FBFlowAnyTimeArchEmbedder(NB201FBFlowArchEmbedder):
    NAME = "nb201-fbflow-anytime"

    def forward(self, archs, input_node_emb = None, any_time: bool = False):
        if not any_time:
            return super(NB201FBFlowAnyTimeArchEmbedder, self).forward(archs, input_node_emb)

        if isinstance(archs, tuple):
            if len(archs) == 2:
                archs, zs_as_l = archs
                zs_as_p = None
            elif len(archs) == 3:
                archs, zs_as_l, zs_as_p = archs
            else:
                raise Exception()
        else:
            zs_as_l = zs_as_p = None

        adjs, x, op_embs = self.embed_and_transform_arch(archs, input_node_emb)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_embs).normal_() * 0.1
            op_embs = op_embs + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = self.op_emb.weight.new(zs_as_p)

        if self.num_time_steps > 1:
            # calculate op mask
            opemb_update_mask = F.embedding(adjs, self._parametrized_op_emb)
        else:
            opemb_update_mask = None

        y_cache = []
        for t in range(self.num_time_steps):
            # concat zeroshot onto the op embedding for forward and backward
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_embs = torch.cat((op_embs, zs_as_p), dim=-1)
            else:
                auged_op_embs = op_embs

            y, message = self._forward_pass(x, adjs, auged_op_embs)
            y_cache.append(self._final_process(y))

            if t == self.num_time_steps - 1:
                break

            b_y, b_message = self._backward_pass(y, adjs, zs_as_l, auged_op_embs) 
            op_embs = self._update_op_emb(y, b_y, op_embs, message, b_message, opemb_update_mask)
    
        return y_cache
