# -*- coding: utf-8 -*-
import os
import sys
import torch
import pickle
from itertools import repeat
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops


# DISCLAIMER
# This code file is derived from
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/io/planetoid.html#read_planetoid_data

class ARGADataset(InMemoryDataset):

    def __init__(self, root, embedding_type, dataset, graph_type="directed",
                 transform=None, pre_transform=None):
        self.embedding_type = embedding_type
        self.dataset = dataset
        self.graph_type = graph_type
        super(ARGADataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.graph_type == "directed":
            names = ['x', 'y', 'allx', 'ally', 'graph_directed']
        else:
            names = ['x', 'y', 'allx', 'ally', 'graph']
        return ["ind.{}.{}".format(self.dataset, name) for name in names]

    @property
    def processed_file_names(self):
        return self.dataset + "_" + self.graph_type + "_arga_data.pt"

    def download(self):
        pass

    def process(self):
        if self.graph_type == "directed":
            names = ['x', 'y', 'allx', 'ally', 'graph_directed']
        else:
            names = ['x', 'y', 'allx', 'ally', 'graph']
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "..", "data", "interim", "gat",
                            self.embedding_type, self.dataset)
        items = [self.read_file(path, name) for name in names]
        x, y, allx, ally, graph = items

        train_index = torch.arange(y.size(0), dtype=torch.long)
        val_index = torch.arange(y.size(0), ally.size(0), dtype=torch.long)

        y = ally.max(dim=1)[1]
        train_mask = self.index_to_mask(train_index, size=y.size(0))
        val_mask = self.index_to_mask(val_index, size=y.size(0))
        edge_index = self.edge_index_from_dict(graph, num_nodes=y.size(0))

        data = Data(x=allx, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def index_to_mask(self, index, size):
        mask = torch.zeros((size, ), dtype=torch.bool)
        mask[index] = 1
        return mask

    def edge_index_from_dict(self, graph_dict, num_nodes=None):
        row, col = [], []
        for key, value in graph_dict.items():
            row += repeat(key, len(value))
            col += value
        edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
        return edge_index

    def read_file(self, folder, name):
        path = os.path.join(folder, "ind.{}.{}".format(self.dataset, name))
        with open(path, "rb") as f:
            out = pickle.load(f, encoding="latin1")
        if name == "graph" or name == "graph_directed":
            return out
        out = out.todense() if hasattr(out, "todense") else out
        out = torch.Tensor(out)
        return out

    def __repr__(self):
        return "{}()".format(self.dataset)
