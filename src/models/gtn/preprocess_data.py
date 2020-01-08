# -*- coding: utf-8 -*-
import os
import sys
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
from DataLoader import DataLoader
from SciBERTEmbeddingsParser import EmbeddingsParser


class Processor:

    def __init__(self, embedding_type, dataset, gpu=0):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.embeddings_parser = EmbeddingsParser(gpu)
        self.path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "gtn",
                self.embedding_type, self.dataset)
        if not os.path.exists(self.path_persistent):
            os.makedirs(self.path_persistent)

    def training_data(self):
        print("Creating training files.\n")

        # Load training and validation data
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        train_val_papers = pd.concat((df_train, df_validation),
                                     axis=0).reset_index(drop=True)

        df_train_authors = d_train.author_names().data
        df_val_authors = d_val.author_names().data
        train_val_authors = pd.concat((df_train_authors, df_val_authors),
                                      axis=0).reset_index(drop=True)

        # Create labels for training data
        self._create_labels(train_val_papers, df_train)

        # Create adjacency matrices
        print("Creating adjacency matrices...")
        papers = train_val_papers[["chapter"]]
        authors = train_val_authors.groupby("author_name")["chapter"].agg(
                list).reset_index()
        authors.index = range(papers.shape[0], papers.shape[0]+len(authors))
        conferences = train_val_authors.groupby("conferenceseries")[
                "chapter"].agg(list).reset_index()
        conferences.index = range(
                papers.shape[0]+authors.shape[0],
                papers.shape[0]+authors.shape[0]+len(conferences))
        count_nodes = len(papers) + len(authors) + len(conferences)
        print("\tNumber of papers: {}".format(len(papers)))
        print("\tNumber of authors: {}".format(len(authors)))
        print("\tNumber of conferences: {}".format(len(conferences)))
        print("\tTotal number of nodes: {}".format(count_nodes))
        indices_pa = self._papers_authors_adjacency(papers, authors)
        indices_pc = self._papers_conferences_adjacency(papers, conferences)

        print("Transforming adjacency lists to matrices...")
        A_pa = self._adj_list_to_matrix(indices_pa, count_nodes)
        A_ap = A_pa.T
        A_pc = self._adj_list_to_matrix(indices_pc, count_nodes)
        A_cp = A_pc.T
        edges = [A_pa[:count_nodes, :count_nodes],
                 A_ap[:count_nodes, :count_nodes],
                 A_pc[:count_nodes, :count_nodes],
                 A_cp[:count_nodes, :count_nodes]]
        print("Edges: ", edges)
        print("Transformed.")
        print("Saving edges to disk...")
        edges_file = os.path.join(self.path_persistent, "edges.pkl")
        with open(edges_file, "wb") as f:
            pickle.dump(edges, f)
        print("Saved.\n")

        # Create features
        print("Creating node features...")
        paper_features = self._create_paper_features(train_val_papers)
        data_authors = train_val_authors.groupby("author_name")[
                "chapter_abstract"].agg(list).reset_index()
        author_features = self._create_authors_features(data_authors)
        data_conferences = train_val_papers.groupby("conferenceseries")[
                "chapter_abstract"].agg(list).reset_index()
        conference_features = self._create_conference_features(
                data_conferences)
        node_features = np.concatenate(
                [paper_features, author_features, conference_features], axis=0)
        print("\tCreated a total of {} node features.".format(
                len(node_features)))
        print("Created.")
        print("Saving to disk...")
        node_features_file = os.path.join(self.path_persistent,
                                          "node_features.pkl")
        with open(node_features_file, "wb") as f:
            pickle.dump(node_features, f)
        print("Saved.")

        print("Created")

    def _create_paper_features(self, data):
        paper_features = []
        with tqdm(desc="Creating paper features: ", total=len(data)) as pbar:
            for idx in range(len(data)):
                paper_features.append(np.concatenate((
                        self.embeddings_parser.embed_sequence(
                                data.chapter_abstract.iloc[idx],
                                self.embedding_type),
                        self.embeddings_parser.embed_sequence(
                                data.chapter_title.iloc[idx],
                                self.embedding_type)),
                        axis=0).tolist())
                pbar.update(1)
        print("\tCreated {} paper features.".format(len(paper_features)))
        return np.asarray(paper_features)

    def _create_authors_features(self, data):
        author_features = []
        with tqdm(desc="Creating author features: ", total=len(data)) as pbar:
            for idx in range(len(data)):
                author_features.append(np.concatenate((
                        np.mean([self.embeddings_parser.embed_sequence(
                                data.iloc[idx].chapter_abstract[i],
                                self.embedding_type) for i in range(
                                        len(data.iloc[idx].chapter_abstract))],
                                axis=0),
                        self.embeddings_parser.embed_sequence(
                                data.iloc[idx].author_name,
                                self.embedding_type)),
                        axis=0).tolist())
                pbar.update(1)
        print("\tCreated {} author features.".format(len(author_features)))
        return np.asarray(author_features)

    def _create_conference_features(self, data):
        conference_features = []
        with tqdm(desc="Creating conference features: ",
                  total=len(data)) as pbar:
            for idx in range(len(data)):
                conference_features.append(np.concatenate((
                        np.mean([self.embeddings_parser.embed_sequence(
                                data.iloc[idx].chapter_abstract[i],
                                self.embedding_type) for i in range(
                                        len(data.iloc[idx].chapter_abstract))],
                                axis=0),
                        self.embeddings_parser.embed_sequence(
                                data.iloc[idx].conferenceseries,
                                self.embedding_type)),
                        axis=0).tolist())
                pbar.update(1)
        print("\tCreated {} conference features.".format(len(
                conference_features)))
        return np.asarray(conference_features)

    def _adj_list_to_matrix(self, indices, count_nodes):
        row = np.array(indices)[:, 0]
        col = np.array(indices)[:, 1]
        data = np.ones_like(row)
        matrix = csr_matrix((data, (row, col)),
                            shape=(count_nodes, count_nodes))
        return matrix

    def _papers_authors_adjacency(self, papers, authors):
        indices_pa = []
        with tqdm(desc="Creating papers-author adjacency matrix",
                  total=len(authors)) as pbar:
            for idx in list(authors.index):
                paper_indices = [papers[papers.chapter == paper].index.tolist()
                                 for paper in authors.chapter.loc[idx]]
                paper_indices = [i[0] for i in paper_indices if i]
                indices_pa.extend([[i, idx] for i in paper_indices])
                pbar.update(1)
        print("Paper-author edges: {}".format(len(indices_pa)))
        return indices_pa

    def _papers_conferences_adjacency(self, papers, conferences):
        indices_pc = []
        with tqdm(desc="Creating paper-conference adjacency matrix",
                  total=len(conferences)) as pbar:
            for idx in list(conferences.index):
                paper_indices = [papers[papers.chapter == paper].index.tolist()
                                 for paper in conferences.chapter.loc[idx]]
                paper_indices = [i[0] for i in paper_indices if i]
                indices_pc.extend([[i, idx] for i in paper_indices])
                pbar.update(1)
        print("Paper-conference edges: {}".format(len(indices_pc)))
        return indices_pc

    def _create_labels(self, train_val_data, train_data):
        print("Creating labels...")
        label_encoder = LabelEncoder()
        conferences = train_val_data.conferenceseries.unique()
        label_encoder.fit(conferences)
        labels = label_encoder.transform(train_val_data.conferenceseries)
        label_indices = [[idx, labels[idx]] for idx in range(len(
                        train_val_data))]
        labels_list = [label_indices[:len(train_data)],
                       label_indices[len(train_data):]]
        print("Created.")
        print("Saving to disk...")
        labels_file = os.path.join(self.path_persistent, "labels.pkl")
        encoder_file = os.path.join(self.path_persistent, "label_encoder.pkl")
        with open(labels_file, "wb") as f:
            pickle.dump(labels_list, f)
        with open(encoder_file, "wb") as f:
            pickle.dump(label_encoder, f)
        print("Saved.\n")

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
        print("Starting...\n")
        from preprocess_data import Processor
        processor = Processor(args.embedding_type, args.dataset, args.gpu)
        processor.training_data()
        print("Finished.\n")

    if __name__ == "__main__":
        main()