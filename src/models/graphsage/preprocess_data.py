# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler

from utils import run_random_walks

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
from TimerCounter import Timer
from DataLoader import DataLoader
from SciBERTEmbeddingsParser import EmbeddingsParser


class Processor():

    def __init__(self, embedding_type, graph_type, gpu=None):
        self.embedding_type = embedding_type
        self.graph_type = graph_type
        self.embeddings_parser = EmbeddingsParser(gpu)
        self.timer = Timer()
        self.path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, self.graph_type)
        if not os.path.isdir(self.path_persistent):
            os.mkdir(self.path_persistent)

    def training_data(self, num_walks=50):
        self.prefix = "train_val"
        self.timer.tic()
        print("Creating training files.")

        # Load training and validation data
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        # Create and save graph
        self.G = nx.Graph()

        # Add nodes and edges
        print("Adding training nodes.")
        self._add_nodes(df_train, test=False, val=False)
        print("Adding training edges.")
        if self.graph_type == "authors":
            df_train = d_train.author_names().data
        self._add_edges(df_train)
        print("Adding validation nodes.")
        self._add_nodes(df_validation, test=False, val=True)
        print("Adding validation edges.")
        if self.graph_type == "authors":
            df_validation = d_val.author_names().data
        self._add_edges(df_validation)
        print("Removing nodes without features.")
        for node in list(self.G.nodes()):
            if "feature" not in self.G.nodes[node].keys():
                self.G.remove_node(node)
        print("Nodes in graph: {}, edges in graph: {}.\n".format(
                self.G.number_of_nodes(), self.G.number_of_edges()))

        print("Saving graph to disk.")
        G_data = json_graph.node_link_data(self.G)
        with open(os.path.join(self.path_persistent, self.prefix + "-G.json"),
                  "w") as f:
            f.write(json.dumps(G_data))

        # Create and save id map
        self._create_id_map()

        # Create and save features
        self._create_features()

        # Perform and save random walks
        nodes = [n for n in list(self.G.nodes()) if not self.G.node[n]["val"]
                 and not self.G.node[n]["test"]]
        subgraph = self.G.subgraph(nodes)
        self._run_random_walks(subgraph, nodes, num_walks)

        print("Finished creating training files.")
        self.timer.toc()

        # print some statistics
        self._get_stats()

        # Plot degree histogram
        self._degree_histogram()

    def test_data(self, df_test, G_train, authors_df=None, normalize=True):
        # TO DO: Add case for authors
        self.prefix = "test"
        print("Preprocessing data...")
        self.G = G_train
        print("Training graph has {} nodes and {} edges.\n".format(
                self.G.number_of_nodes(), self.G.number_of_edges()))

        # Add nodes and edges
        print("Adding test nodes.")
        self._add_nodes(df_test, test=True, val=False)
        print("Adding test edges.")
        if self.graph_type == "authors":
            if authors_df is not None:
                df_test = pd.merge(df_test, authors_df, how="left",
                                   on=["chapter", "chapter"])
            else:
                raise ValueError("Paper authors are missing.")
        self._add_edges(df_test)
        print("Removing nodes without features.")
        for node in list(self.G.nodes()):
            if "feature" not in self.G.nodes[node].keys():
                self.G.remove_node(node)
        print("Nodes in graph: {}, edges in graph: {}.\n".format(
                self.G.number_of_nodes(), self.G.number_of_edges()))

        # Remove all nodes that do not have val/test annotations
        broken_count = 0
        for node in self.G.nodes():
            if 'val' not in self.G.node[node] or 'test' not in self.G.node[
                    node]:
                self.G.remove_node(node)
                broken_count += 1
        print("Removed {} nodes that lacked proper annotations due to networkx versioning issues.".format(
                broken_count))

        # Make sure the graph has edge train_removed annotations
        for edge in self.G.edges():
            if (self.G.node[edge[0]]['val'] or self.G.node[edge[1]]['val'] or
               self.G.node[edge[0]]['test'] or self.G.node[edge[1]]['test']):
                self.G[edge[0]][edge[1]]['train_removed'] = True
            else:
                self.G[edge[0]][edge[1]]['train_removed'] = False

        # Create and process id map
        id_map = self._create_id_map()

        if isinstance(list(self.G.nodes)[0], int):
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n
        id_map = {conversion(k): int(v) for k, v in id_map.items()}

        # Create and process features
        features = self._create_features()

        if normalize:
            train_ids = np.array([id_map[n] for n in self.G.nodes() if not
                                  self.G.node[n]['val'] and not
                                  self.G.node[n]['test']])
            train_feats = features[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            features = scaler.transform(features)
        print("Finished preprocessing data.")

        # print some statistics
        self._get_stats()

        # Plot degree histogram
        self._degree_histogram()

        return self.G, features, id_map

    def _add_nodes(self, data, test=False, val=False):
        with tqdm(desc="Adding nodes: ", total=len(data), unit="node") as pbar:
            for idx in range(len(data)):
                self.G.add_node(
                        data.chapter.iloc[idx],
                        test=test,
                        feature=np.concatenate((
                                self.embeddings_parser.embed_sequence(
                                        data.chapter_title.iloc[idx],
                                        self.embedding_type),
                                self.embeddings_parser.embed_sequence(
                                        data.chapter_abstract.iloc[idx],
                                        self.embedding_type)),
                                axis=0).tolist(),
                        val=val)
                pbar.update(1)
        print("Nodes in graph: {}.\n".format(self.G.number_of_nodes()))

    def _add_edges(self, data):
        if self.graph_type == "citations":
            self._add_edges_citations(data)
        elif self.graph_type == "conferenceseries":
            self._add_edges_conferenceseries(data)
        elif self.graph_type == "authors":
            self._add_edges_authors(data)
        else:
            raise KeyError("Graph type unknown.")

    def _add_edges_citations(self, data):
        """Adds edges between papers that share a citation.
        """
        with tqdm(desc="Adding edges: ", total=len(data), unit="edge") as pbar:
            for idx in range(len(data)):
                self.G.add_edges_from(
                        [(data.chapter.iloc[idx],
                          data.chapter_citations.iloc[idx][i])
                         for i in range(
                                len(data.chapter_citations.iloc[idx]))])
                pbar.update(1)
        print("Edges in graph: {}.\n".format(self.G.number_of_edges()))

    def _add_edges_conferenceseries(self, data):
        """Adds edges between papers published at the same conferenceseries.
        """
        data_grouped = data.groupby("conferenceseries")["chapter"].agg(
                list).reset_index()
        with tqdm(desc="Adding edges: ", total=len(data_grouped),
                  unit="edge") as pbar:
            for idx in range(len(data_grouped)):
                self.G.add_edges_from(combinations(
                        data_grouped.iloc[idx].chapter, 2))
                pbar.update(1)
        print("Edges in graph: {}.\n".format(self.G.number_of_edges()))

    def _add_edges_authors(self, data):
        """Adds edges between papers sharing an author.
        """
        data_grouped = data.groupby("author_name")["chapter"].agg(
                list).reset_index()
        with tqdm(desc="Adding edges: ", total=len(data_grouped),
                  unit="edge") as pbar:
            for idx in range(len(data_grouped)):
                self.G.add_edges_from(combinations(
                        data_grouped.iloc[idx].chapter, 2))
                pbar.update(1)
        print("Edges in graph: {}.\n".format(self.G.number_of_edges()))

    def _create_id_map(self):
        if self.prefix == "train_val":
            print("Creating id map.")

        nodes = list(self.G.nodes)
        id_map = {nodes[i]: i for i in range(len(nodes))}

        if self.prefix == "test":
            return id_map
        else:
            print("Saving id map to disk.")
            with open(os.path.join(
                    self.path_persistent, self.prefix + "-id_map.json"),
                    "w") as f:
                f.write(json.dumps(id_map))

    def _create_features(self):
        if self.prefix == "train_val":
            print("Creating features.")

        features = np.array([self.G.nodes[node]["feature"] for node in
                             list(self.G.nodes)])

        if self.prefix == "test":
            return features
        else:
            print("Saving features to disk.")
            np.save(os.path.join(self.path_persistent, self.prefix +
                                 "-feats.npy"),
                    features)

    def _run_random_walks(self, graph, nodes, num_walks):
        print("Running random walks.")
        walks = run_random_walks(graph, nodes, num_walks=num_walks)
        print("Saving random walks to disk.")
        with open(os.path.join(
                self.path_persistent, self.prefix + "-walks.txt"), "w") as fp:
            fp.write("\n".join([str(w[0]) + "\t" + str(w[1]) for w in walks]))

    def _get_stats(self):
        degree_sequence = sorted([d for n, d in self.G.degree()], reverse=True)
        degree_count = Counter(degree_sequence)

        with open(os.path.join(
                self.path_persistent, self.prefix + "-stats.txt"), "w") as fp:
            self._print("Number of nodes in the graph: {}\n".format(
                    self.G.number_of_nodes()), fp)
            self._print("Number of edges in the graph: {}\n".format(
                    self.G.number_of_edges()), fp)
            self._print("The graph is connected: {}\n".format(
                    nx.is_connected(self.G)), fp)
            self._print("Number of connected components: {}\n".format(
                    nx.number_connected_components(self.G)), fp)
            self._print("Number of self-loops: {}\n".format(
                    nx.number_of_selfloops(self.G)), fp)
            self._print("Maximum degree: {}\n".format(max(degree_count)), fp)
            self._print("Minimum degree: {}\n".format(min(degree_count)), fp)
            self._print("Average degree: {}\n".format(
                    sum(degree_sequence)/len(self.G)), fp)

    def _degree_histogram(self):
        # Plot degree histogram
        degree_sequence = sorted([d for n, d in self.G.degree()], reverse=True)
        degree_count = Counter(degree_sequence)
        deg, cnt = zip(*degree_count.items())

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color='b')
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)

        plt.savefig(os.path.join(
                self.path_persistent, self.prefix + "-degree_histogram.png"),
            bbox_inches="tight")

    def _print(self, text, f):
        print(text)
        f.write(text)
