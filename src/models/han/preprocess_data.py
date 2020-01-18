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
from process import sample_mask
from itertools import combinations

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
from TimerCounter import Timer
from DataLoader import DataLoader
from SciBERTEmbeddingsParser import EmbeddingsParser


class Processor:

    def __init__(self, embedding_type, gpu=0):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.embedding_type = embedding_type
        self.embeddings_parser = EmbeddingsParser(gpu)
        self.timer = Timer()
        self.path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "han",
                self.embedding_type)
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

        print("Creating index files for training and validation data.")
        train_idx = np.asarray(list(train_val_data.index))[:len(df_train)]
        train_idx = np.asarray([train_idx])
        val_idx = np.asarray(list(train_val_data.index))[len(df_train):]
        val_idx = np.asarray([val_idx])
        print("Created.")
        print("Saving to disk...")
        train_idx_file = os.path.join(self.path_persistent, "train_idx.pkl")
        val_idx_file = os.path.join(self.path_persistent, "val_idx.pkl")
        with open(train_idx_file, "wb") as f:
            pickle.dump(train_idx, f)
        with open(val_idx_file, "wb") as f:
            pickle.dump(val_idx, f)
        print("Saved.")

        print("Creating labels for training and validation data.")
        self._train_label_encoder(train_val_data)
        train_val_labels = self.label_encoder.transform(
                np.array(train_val_data.conferenceseries).reshape(-1, 1))
        print("Created")
        print("Saving to disk...")
        labels_file = os.path.join(self.path_persistent, "labels.pkl")
        with open(labels_file, "wb") as f:
            pickle.dump(train_val_labels, f)
        print("Saved.\n")

        print("Creating feature vectors for training and validation data.")
        train_val_features = self._create_features(train_val_data)
        print("Created.")
        print("Saving to disk...")
        features_file = os.path.join(self.path_persistent, "features.pkl")
        with open(features_file, "wb") as f:
            pickle.dump(train_val_features, f)
        print("Saved.\n")

        df_train_authors = d_train.author_names().data
        df_val_authors = d_val.author_names().data
        train_val_authors_data = pd.concat(
                (df_train_authors, df_val_authors), axis=0).reset_index(
                        drop=True)
        data_authors = train_val_authors_data.groupby(
                "author_name")["chapter"].agg(list).reset_index()

        print("Creating adjacency matrices...")
        PCP = self._create_PCP_adjacency(train_val_data)
        PAP = self._create_PAP_adjacency(train_val_data, data_authors)
        print("Created.")

        print("Finished creating training files.\n")

        print("Statistics")
        print("\tTraining and validation data features: {}.".format(
                train_val_features.shape))
        print("\tTraining and validation data labels: {}.".format(
                train_val_labels.shape))
        print("\tPCP graph size: {}.".format(len(PCP)))
        print("\tMax node degree: {}.".format(len(max(PCP.values(),
              key=len))))
        print("\tPAP graph size: {}.".format(len(PAP)))
        print("\tMax node degree: {}.".format(len(max(PAP.values(),
              key=len))))

    def test_data(self):
        pass

    def _create_PCP_adjacency(self, data):
        print("Creating paper-citation-paper adjacency lists.")
        graph = defaultdict(list)
        with tqdm(desc="Adding neighbours: ", total=len(data)) as pbar:
            for idx in range(len(data)):
                citations_indices = [data[
                        data.chapter == citation].index.tolist() for
                        citation in data.chapter_citations.iloc[idx]]
                graph[idx] = list(set([i[0] for i in citations_indices if i]))
                pbar.update(1)
        print("Created.")
        print("Saving to disk...")
        graph_file = os.path.join(self.path_persistent, "PCP.pkl")
        with open(graph_file, "wb") as f:
            pickle.dump(graph, f)
        print("Saved.\n")
        return graph

    def _create_PAP_adjacency(self, data, data_authors):
        print("Creating paper-author-paper adjacency lists.")
        graph = defaultdict()
        for idx in data.index:
            graph[idx] = []
        # Add edges between papers if they share an author
        with tqdm(desc="Adding neighbours: ", total=len(data_authors)) as pbar:
            for idx in range(len(data_authors)):
                authors_indices = [data[data.chapter == paper].index.tolist()
                                   for paper in data_authors.chapter.iloc[idx]]
                authors_indices = [i[0] for i in authors_indices if i]
                edges = [i for i in combinations(authors_indices, 2)]
                for edge in edges:
                    graph[edge[0]].append(edge[1])
                pbar.update(1)
        print("Created.")
        print("Saving to disk...")
        graph_file = os.path.join(self.path_persistent, "PAP.pkl")
        with open(graph_file, "wb") as f:
            pickle.dump(graph, f)
        print("Saved.\n")
        return graph

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
        return np.asarray(features)

    def _train_label_encoder(self, data):
        self.label_encoder = OneHotEncoder(handle_unknown='ignore',
                                           sparse=False, dtype=np.int)
        labels = data.conferenceseries.unique()
        labels = labels.reshape(-1, 1)
        self.label_encoder.fit(labels)
        with open(os.path.join(self.path_persistent, "label_encoder.pkl"),
                  "wb") as f:
            pickle.dump(self.label_encoder, f)

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
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        args = parser.parse_args()
        print("Starting...")
        from preprocess_data import Processor
        processor = Processor(args.embedding_type, args.gpu)
        processor.training_data()
        print("Finished.")

    if __name__ == "__main__":
        main()
