# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from utils import *

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
from TimerCounter import Timer
from DataLoader import DataLoader
from SciBERTEmbeddingsParser import EmbeddingsParser


class Processor:

    def __init__(self, embedding_type, dataset, gpu=0):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.embeddings_parser = EmbeddingsParser(gpu)
        self.timer = Timer()
        self.path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "gat",
                self.embedding_type, self.dataset)
        if not os.path.exists(self.path_persistent):
            os.makedirs(self.path_persistent)

    def training_data(self):
        self.timer.tic()
        print("Creating training files.\n")

        # Load training and validation data
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        train_val_data = pd.concat((df_train, df_validation),
                                   axis=0).reset_index(drop=True)

        # Create file with feature vectors for both training and validation
        # data (as a scipy.sparse.csr.csr_matrix object)
        print("Creating feature vectors for training and validation data.")
        train_val_features = self._create_features(train_val_data)
        print("Created.")
        print("Saving to disk...")
        allx_file = os.path.join(self.path_persistent,
                                 "ind." + self.dataset + ".allx")
        with open(allx_file, "wb") as f:
            pickle.dump(train_val_features, f)
        print("Saved.\n")

        # Create file with feature vectors only for training data
        # (as a scipy.sparse.csr.csr_matrix object)
        print("Creating feature vectors for training data.")
        train_features = train_val_features[:len(df_train)]
        print("Created.")
        print("Saving to disk...")
        x_file = os.path.join(self.path_persistent,
                              "ind." + self.dataset + ".x")
        with open(x_file, "wb") as f:
            pickle.dump(train_features, f)
        print("Saved.\n")

        # Create file with the labels for the training and validation data
        # (as a numpy.ndarray object)
        print("Creating labels for training and validation data.")
        self._train_label_encoder(train_val_data)
        train_val_labels = self.label_encoder.transform(
                np.array(train_val_data.conferenceseries).reshape(-1, 1))
        print("Created")
        print("Saving to disk...")
        ally_file = os.path.join(self.path_persistent,
                                 "ind." + self.dataset + ".ally")
        with open(ally_file, "wb") as f:
            pickle.dump(train_val_labels, f)
        print("Saved.\n")

        # Create file with the labels for the training data
        # (as a numpy.ndarray object)
        print("Creating labels for training data.")
        train_labels = train_val_labels[:len(df_train)]
        print("Created.")
        print("Saving to disk...")
        y_file = os.path.join(self.path_persistent,
                              "ind." + self.dataset + ".y")
        with open(y_file, "wb") as f:
            pickle.dump(train_labels, f)
        print("Saved.\n")

        # Create a dict in the format {index: [index_of_neighbor_nodes]}
        # (as a collections.defaultdict object)
        print("Creating dictionary of neighbours.")
        graph = defaultdict(list)
        with tqdm(desc="Adding neighbours: ",
                  total=len(train_val_data)) as pbar:
            for idx in range(len(train_val_data)):
                citations_indices = [train_val_data[
                        train_val_data.chapter == citation].index.tolist() for
                        citation in train_val_data.chapter_citations.iloc[idx]]
                neighbours = [c[0] for c in citations_indices if c]
                graph[idx].extend(neighbours)
                for node in neighbours:
                    graph[node].append(idx)
                pbar.update(1)
        with tqdm(desc="Removing duplicates: ", total=len(graph.keys())
                  ) as pbar:
            for idx in range(len(graph.keys())):
                graph[idx] = list(set(graph[idx]))
                pbar.update(1)
        print("Saving to disk...")
        graph_file = os.path.join(self.path_persistent,
                                  "ind." + self.dataset + ".graph")
        with open(graph_file, "wb") as f:
            pickle.dump(graph, f)
        print("Saved.\n")

        print("Statistics")
        print("\tTraining data features: {}.".format(train_features.shape))
        print("\tTraining data labels: {}.".format(len(train_labels)))
        print("\tTraining and validation data features: {}.".format(
                train_val_features.shape))
        print("\tTraining and validation data labels: {}.".format(
                len(train_val_labels)))
        print("\tGraph size: {}.".format(len(graph)))

    def _create_features(self, data):
        features = []
        with tqdm(desc="Creating features: ", total=len(data)) as pbar:
            for idx in range(len(data)):
                features.append(np.concatenate((
                        self.embeddings_parser.embed_sequence(
                                data.chapter_title.iloc[idx],
                                self.embedding_type),
                        self.embeddings_parser.embed_sequence(
                                data.chapter_abstract.iloc[idx],
                                self.embedding_type)),
                        axis=0).tolist())
                pbar.update(1)
        return sp.csr.csr_matrix(np.array(features))

    def _train_label_encoder(self, data):
        self.label_encoder = OneHotEncoder(handle_unknown='ignore',
                                           sparse=False, dtype=np.int)
        labels = data.conferenceseries.unique()
        labels = labels.reshape(-1, 1)
        self.label_encoder.fit(labels)
        with open(os.path.join(self.path_persistent, "label_encoder.pkl"),
                  "wb") as f:
            pickle.dump(self.label_encoder, f)

    def test_data(self, df_test, train_features, train_labels,
                  train_val_features, train_val_labels, graph):
        print("Preprocessing data...")
        # Load training and validation data
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        train_val_data = pd.concat((df_train, df_validation),
                                   axis=0).reset_index(drop=True)

        # Create the indices of test instances in graph (as a list object)
        test_indices = list(df_test.index)

        # Create "fake" temporary labels for test data
        test_labels = np.zeros(
                (len(df_test), len(train_val_labels[0])), dtype=int)

        # Update graph with test data
        print("Updating graph information...")
        with tqdm(desc="Adding neighbours: ", total=len(df_test)) as pbar:
            for idx in list(df_test.index):
                citations_indices = [train_val_data[
                        train_val_data.chapter == citation].index.tolist() for
                        citation in df_test.chapter_citations.loc[idx]]
                neighbours = [c[0] for c in citations_indices if c]
                graph[idx].extend(neighbours)
                for node in neighbours:
                    graph[node].append(idx)
                pbar.update(1)
        with tqdm(desc="Removing duplicates: ", total=len(graph.keys())
                  ) as pbar:
            for idx in range(len(graph.keys())):
                graph[idx] = list(set(graph[idx]))
                pbar.update(1)
        print("Updated.")

        # Create feature vectors of test instances
        print("Creating features for test data...")
        test_features = self._create_features(df_test)
        print("Created.")

        max_degree = len(max(graph.values(), key=len))
        test_idx_range = np.sort(test_indices)
        features = sp.vstack((train_val_features, test_features)).tolil()
        features[test_indices, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((train_val_labels, test_labels))
        labels[test_indices, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(train_labels))
        idx_val = range(len(train_labels), len(train_val_labels))

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
        print("Finished preprocessing data.\n")

        print("Adjacency matrix shape: {}.".format(adj.shape))
        print("Features matrix shape: {}.".format(features.shape))
        print("Graph size: {}.".format(len(graph)))
        print("Max degree: {}.\n".format(max_degree))

        dataset = [adj, features, y_train, y_test, train_mask, test_mask]
        prepared_test_data = self._prepare_test_data(dataset, max_degree)
        return prepared_test_data, max_degree

    def _prepare_test_data(self, dataset, max_degree):
        print("Preparing test data...")
        adj, features, y_train, y_test, train_mask, test_mask = dataset
        train_index = np.where(train_mask)[0]
        adj_train = adj[train_index, :][:, train_index]
        y_train = y_train[train_index]
        test_index = np.where(test_mask)[0]
        y_test = y_test[test_index]

        num_train = adj_train.shape[0]
        input_dim = features.shape[1]

        features = nontuple_preprocess_features(features).todense()
        train_features = features[train_index]

        norm_adj_train = nontuple_preprocess_adj(adj_train)
        norm_adj = nontuple_preprocess_adj(adj)

        adj_train, adj_val_train = compute_adjlist(norm_adj_train, max_degree)
        train_features = np.concatenate((train_features,
                                         np.zeros((1, input_dim))))
        print("Prepared.\n")
        return norm_adj, adj_train, adj_val_train, features, train_features, y_train, y_test, test_index

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for data preprocessing.')
        parser.add_argument('embedding_type',
                            choices=["AVG_L", "AVG_2L", "AVG_SUM_L4",
                                     "AVG_SUM_ALL", "MAX_2L",
                                     "CONC_AVG_MAX_2L", "CONC_AVG_MAX_SUM_L4",
                                     "SUM_L", "SUM_2L"
                                     ],
                            help="Type of embedding.")
        parser.add_argument('dataset',
                            help='Name of the object file that stores the '
                            + 'training data.')
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        args = parser.parse_args()
        print("Starting...")
        from preprocess_data import Processor
        processor = Processor(args.embedding_type, args.dataset, args.gpu)
        processor.training_data()
        print("Finished.")

    if __name__ == "__main__":
        main()
