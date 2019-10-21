# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph

from utils import run_random_walks

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
from TimerCounter import Timer
from DataLoader import DataLoader
from SciBERTEmbeddingsParser import EmbeddingsParser


class Processor():

    def __init__(self, embedding_type, gpu=None):
        self.embedding_type = embedding_type
        self.embeddings_parser = EmbeddingsParser(gpu)
        self.timer = Timer()
        self.path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type)
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
        self._add_edges(df_train)
        print("Adding validation nodes.")
        self._add_nodes(df_validation, test=False, val=True)
        print("Adding validation edges.")
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

        # print some statistics
        self._get_stats()

        # Plot degree histogram
        self._degree_histogram()

        print("Finished creating training files.")
        self.timer.toc()

#    def test_data(self):
#        self.prefix = "test"
#        pass

    def _add_nodes(self, data, test=False, val=False):
        with tqdm(desc="Adding training nodes: ", total=len(data),
                  unit="node") as pbar:
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
        with tqdm(desc="Adding training edges: ", total=len(data),
                  unit="edge") as pbar:
            for idx in range(len(data)):
                self.G.add_edges_from(
                        [(data.chapter.iloc[idx],
                          data.chapter_citations.iloc[idx][i])
                         for i in range(
                                len(data.chapter_citations.iloc[idx]))])
                pbar.update(1)
        print("Edges in graph: {}.\n".format(self.G.number_of_edges()))

    def _create_id_map(self):
        print("Creating id map.")
        nodes = list(self.G.nodes)
        id_map = {nodes[i]: i for i in range(len(nodes))}
        print("Saving id map to disk.")
        with open(os.path.join(
                self.path_persistent, self.prefix + "-id_map.json"), "w") as f:
            f.write(json.dumps(id_map))

    def _create_features(self):
        print("Creating features.")
        features = np.array([self.G.nodes[node]["feature"] for node in
                             list(self.G.nodes)])
        print("Saving features to disk.")
        np.save(os.path.join(self.path_persistent, self.prefix + "-feats.npy"),
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
            self._print("Number of nodes in the graph: {}".format(
                    self.G.number_of_nodes()), fp)
            self._print("Number of edges in the graph: {}".format(
                    self.G.number_of_edges()), fp)
            self._print("The graph is connected: {}".format(
                    nx.is_connected(self.G)), fp)
            self._print("Number of connected components: {}.".format(
                    nx.number_connected_components(self.G)), fp)
            self._print("Number of self-loops: {}".format(
                    nx.number_of_selfloops(self.G)), fp)
            self._print("Maximum degree: {}".format(max(degree_count)), fp)
            self._print("Minimum degree: {}".format(min(degree_count)), fp)
            self._print("Average degree: {}.".format(
                    sum(degree_sequence)/len(G)), fp)

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

