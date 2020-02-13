# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import pickle
import random
import numpy as np
import pandas as pd
from itertools import repeat, product
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "gat"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer
from AbstractClasses import AbstractModel
from preprocess_data import Processor
from gat_preprocess_data import Processor as GATProcessor
from DataLoader import DataLoader
from ffnn import FFNNModel


class SciBERT_ARGAModel(AbstractModel):

    def __init__(self, embedding_type, dataset, arga_model_name, n_latent=16,
                 learning_rate=0.001, weight_decay=0, dropout=0,
                 dis_loss_para=1, reg_loss_para=1, epochs=200, gpu=None,
                 ffnn_hidden_dim=100):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.embedding_type = embedding_type
        self.dataset = dataset

        self.gat_preprocessor = GATProcessor(self.embedding_type, self.dataset,
                                             "undirected", threshold=2,
                                             gpu=gpu)
        self.processor = Processor(
                self.embedding_type, self.dataset, arga_model_name, "test",
                n_latent, learning_rate, weight_decay, dropout, dis_loss_para,
                reg_loss_para, epochs, gpu)
        self.ffnn_model = FFNNModel(embedding_type, dataset, arga_model_name,
                                    n_latent, learning_rate, weight_decay,
                                    dropout, dis_loss_para, reg_loss_para,
                                    epochs, gpu, ffnn_hidden_dim)

        self.training_data = self._load_training_data()

        if not self._load_label_encoder():
            print("The label encoder does not exist.")

    def query_single(self, query):
        """Queries the model and returns a list of recommendations.

        Args:
            query (list): The query as needed by the model, is in the form
            [chapter_title, chapter_abstract, list(chapter_citations)].

        Returns
            list: ids of the conferences
            double: confidence scores
        """
        if self.dataset == "citations":
            if len(query) < 3:
                raise ValueError("The input does not contain enough data; " +
                                 "chapter title, chapter abstract, and " +
                                 "chapter citations are required.")
            return self.query_batch(query)
#            return self.query_batch([(query[0], query[1], query[2])])
        elif self.dataset == "citations_authors_het_edges":
            if len(query) < 4:
                raise ValueError("The input does not contain enough data; " +
                                 "chapter title, chapter abstract, chapter " +
                                 "citations, and chapter authors are required."
                                 )
            query_id = "new_node_id:" + "-".join(
                    [str(i) for i in random.sample(range(0, 10000), 5)])
            authors_df = pd.DataFrame({"author_name": query[3],
                                      "chapter": [query_id]*len(query[3])})
            return self.query_batch((
                    [(query_id, query[0], query[1], query[2])], authors_df))
        else:
            raise ValueError("Dataset not recognised.")

    def query_batch(self, batch):
        """Queries the model and returns a lis of recommendations.

        Args:
            batch (list of ntuples): The list of queries as needed
            by the model. The ntuples are in the form (chapter_id,
            chapter_title, chapter_abstract, list(chapter_citations)).

        Returns
            list: ids of the conferences
            double: confidence scores
        """
        if self.dataset == "citations":
            if len(batch) == 3:
                batch = [batch]
                df_test = pd.DataFrame(batch,
                                       columns=["chapter_title",
                                                "chapter_abstract",
                                                "chapter_citations"])
            else:
                df_test_extended = pd.DataFrame(batch,
                                                columns=["chapter",
                                                         "chapter_title",
                                                         "chapter_abstract",
                                                         "chapter_citations"])
                df_test = df_test_extended[["chapter_title",
                                            "chapter_abstract",
                                            "chapter_citations"]]
            authors_df = None
        elif self.dataset == "citations_authors_het_edges":
            if len(batch) == 4:
                df_test = pd.DataFrame(batch[0],
                                       columns=["chapter_title",
                                                "chapter_abstract",
                                                "chapter_citations"])
            else:
                df_test_extended = pd.DataFrame(batch[0],
                                                columns=["chapter",
                                                         "chapter_title",
                                                         "chapter_abstract",
                                                         "chapter_citations"])
                df_test = df_test_extended[["chapter_title",
                                            "chapter_abstract",
                                            "chapter_citations"]]
            authors_df = batch[1]
        else:
            raise ValueError("Dataset not recognised.")

        # Preprocess data
        test_data = self._preprocess_test_data(df_test)
        allx = test_data[2]
        arga_test_data = self._create_arga_test_dataset(test_data)[0]

        # Compute ARGA and SciBERT embeddings
        arga_embeddings = self.processor.arga_model.test(arga_test_data)
        arga_test_embeddings = arga_embeddings[allx.shape[0]:]
        scibert_test_embeddings = self.processor._scibert_embeddings(df_test)

        # Inference on test data
        predictions = self.ffnn_model.test(scibert_test_embeddings,
                                           arga_test_embeddings)

        # Compute predictions
        sorted_predictions = (-predictions).argsort(axis=1)
        conferences = list()
        confidences = list()

        for i in range(len(predictions)):
            one_hot_preds = np.zeros((self.recs, len(predictions[0])),
                                     dtype=int)
            for j in range(self.recs):
                one_hot_preds[j][sorted_predictions[i, j]] = 1
            conferences.append(list(self.label_encoder.inverse_transform(
                    one_hot_preds).flatten()))
            confidences.append(list(
                    predictions[i, sorted_predictions[:, :self.recs][i]]))

        results = [conferences, confidences]
        return results

    def _preprocess_test_data(self, df_test):
        # Load training data in GAT format
        x, y, allx, ally, graph = self.training_data

        # Reindex test dataframe such that indices follow those from the
        # train data
        df_test.index = range(allx.shape[0], allx.shape[0] + len(df_test))

        # Load training and validation data
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        train_val_data = pd.concat((df_train, df_validation),
                                   axis=0).reset_index(drop=True)

        print("Preprocessing data...")
        # Create the indices of test instances in graph (as a list object)
        test_index = list(df_test.index)

        # Create "fake" temporary labels for test data
        ty = np.zeros((len(df_test), len(ally[0])), dtype=int)

        # Update graph with test data
        print("Updating graph information...")
        if self.dataset == "citations":
            graph = self.gat_preprocessor._update_undirected_graph(
                    graph, train_val_data, df_test)
        if self.dataset == "citations_authors_het_edges":
            data_authors = authors_df.groupby("author_name")["chapter"].agg(
                    list).reset_index()
            graph = self.self.gat_preprocessor._update_heterogeneous_directed_graph(
                        graph, train_val_data, df_test, data_authors)
        print("Updated.")

        # Create feature vectors of test instances
        print("Creating features for test data...")
        tx = self.gat_preprocessor._create_features(df_test)
        print("Created.")

        x = self._to_tensor(x)
        tx = self._to_tensor(tx)
        allx = self._to_tensor(allx)
        y = self._to_tensor(y)
        ty = self._to_tensor(ty)
        ally = self._to_tensor(ally)
        test_index = torch.tensor(test_index, dtype=torch.long)
        print("Finished preprocessing data.")

        return x, tx, allx, y, ty, ally, graph, test_index

    def _to_tensor(self, x):
        x = x.todense() if hasattr(x, "todense") else x
        return torch.Tensor(x)

    def _create_arga_test_dataset(self, test_data):
        x, tx, allx, y, ty, ally, graph, test_index = test_data

        train_index = torch.arange(y.size(0), dtype=torch.long)
        val_index = torch.arange(y.size(0), ally.size(0), dtype=torch.long)
        sorted_test_index = test_index.sort()[0]

        x = torch.cat([allx, tx], dim=0)
        y = torch.cat([ally, ty], dim=0).max(dim=1)[1]

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]

        train_mask = self._index_to_mask(train_index, size=y.size(0))
        val_mask = self._index_to_mask(val_index, size=y.size(0))
        test_mask = self._index_to_mask(test_index, size=y.size(0))

        edge_index = self._edge_index_from_dict(graph, num_nodes=y.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return self.collate([data])

    def _index_to_mask(self, index, size):
        mask = torch.zeros((size, ), dtype=torch.bool)
        mask[index] = 1
        return mask

    def _edge_index_from_dict(self, graph_dict, num_nodes=None):
        row, col = [], []
        for key, value in graph_dict.items():
            row += repeat(key, len(value))
            col += value
        edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
        return edge_index

    def _load_label_encoder(self):
        label_encoder_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "gat",
                self.embedding_type, self.dataset, "label_encoder.pkl")
        if os.path.isfile(label_encoder_file):
            with open(label_encoder_file, "rb") as f:
                print("Loading label encoder.")
                self.label_encoder = pickle.load(f)
            print("Loaded.")
            return True
        return False

    def _load_training_data(self):
        print("Loading training data.")
        path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "gat",
                self.embedding_type, self.dataset)
        names = ['x', 'y', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(path_persistent + "/ind.{}.{}".format(
                    self.dataset, names[i]), 'rb') as f:
                objects.append(pickle.load(f, encoding='latin1'))
        x, y, allx, ally, graph = tuple(objects)
        print("Loaded.")
        return x, y, allx, ally, graph

    # DISCLAIMER
    # Function forked from
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/in_memory_dataset.html#InMemoryDataset.collate
    def collate(self, data_list):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(
                    item.__cat_dim__(key, item[key]))
            else:
                s = slices[key][-1] + 1
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            item = data_list[0][key]
            if torch.is_tensor(item):
                data[key] = torch.cat(data[key],
                                      dim=data.__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                data[key] = torch.tensor(data[key])

            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices

    def train(self):
        pass
