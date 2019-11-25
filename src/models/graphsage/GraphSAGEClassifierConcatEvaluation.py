# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

from networkx.readwrite import json_graph
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "evaluations"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))

from TimerCounter import Timer
from DataLoader import DataLoader
from AbstractClasses import AbstractModel
from unsupervised_model import UnsupervisedModel
from EvaluationContainer import EvaluationContainer


class GraphSAGEClassifierConcatEvaluation():

    def __init__(self, classifier_name, embedding_type, model_name, model_size,
                 learning_rate, gpu, recs=10):
        self.classifier = self._choose_classifier(classifier_name)
        self.embedding_type = embedding_type
        self.gpu = gpu
        self.recs = recs

        # Classifier file location
        classifier_dir = os.path.join(os.path.dirname(
                         os.path.realpath(__file__)), "..", "..", "..", "data",
                         "processed", "graphsage", self.embedding_type,
                         "citations_authors")
        classifier_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
                model=model_name, model_size=model_size, lr=learning_rate)
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)
        self.classifier_file = os.path.join(classifier_dir,
                                            self.classifier.__class__.__name__
                                            + ".pkl")

    def compute_predictions(self, test_embeddings):
        predictions = self.classifier.predict_proba(test_embeddings)
        sorted_predictions = np.argsort(-np.array(predictions))

        conferenceseries = list()
        confidences = list()

        for index, order in enumerate(sorted_predictions):
            conferences = list()
            scores = list()
            i = 0
            while len(conferences) < self.recs:
                conf = self.label_encoder.inverse_transform(
                        [order[i]]).tolist()[0]
                if conf not in conferences:
                    conferences.append(conf)
                    scores.append(predictions[index][order][i])
                i += 1
            conferenceseries.append(conferences)
            confidences.append(scores)

        results = [conferenceseries, confidences]
        return results

    def infer_embeddings(self, query, query_authors, graph_type, model,
                         model_checkpoint, queue):
        df_test = pd.DataFrame(query, columns=["chapter", "chapter_title",
                                               "chapter_abstract",
                                               "chapter_citations"])

        # Load the training graph
        print("Loading {} training graph...".format(graph_type))
        G_train = self._load_training_graph(graph_type)
        print("Loaded.")

        # Load the training walks
        print("Loading {} training walks...".format(graph_type))
        walks = self._load_training_walks(graph_type, G_train)
        print("Loaded.")

        print("Preprocessing {} test data...".format(graph_type))
        from preprocess_data import Processor
        preprocessor = Processor(self.embedding_type, graph_type, self.gpu)
        graph, features, id_map = preprocessor.test_data(
                df_test, G_train, authors_df = query_authors)
        print("Preprocessed.")

        print("Inferring embeddings...")
        embeddings = model.predict([graph, features, id_map, walks],
                                   model_checkpoint)[1]
        print("Inferred.")

        queue.put(embeddings)

    def train(self, model_citations, model_authors):
        d_train = DataLoader()
        training_data = d_train.training_data_with_abstracts_citations().data

        # Load trained embeddings
        print("Loading the citations training embeddings...")
        pretrained_embeddings_citations, pretrained_embeddings_id_map_citations = \
            self._load_train_embeddings(model_citations)
        print("Loaded.")

        print("Loading the authors training embeddings...")
        pretrained_embeddings_authors, pretrained_embeddings_id_map_authors = \
            self._load_train_embeddings(model_authors)
        print("Loaded.")

        training_ids = list(training_data.chapter)
        training_embeddings_citations = pretrained_embeddings_citations[[
                pretrained_embeddings_id_map_citations[id] for id in
                training_ids]]
        training_embeddings_authors = pretrained_embeddings_authors[[
                pretrained_embeddings_id_map_authors[id] for id in
                training_ids]]

        # Concatenate embeddings
        training_embeddings = np.concatenate((
                training_embeddings_citations, training_embeddings_authors
                ), axis=1)

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(
                data.conferenceseries)
        self.classifier.fit(training_embeddings, self.labels)
        self._save_model_classifier()

        print("Training finished.")
        timer.toc()

    def load_data(self):
        d_test = DataLoader()
        d_test_authors = DataLoader()

        # Load test data
        query_test, truth = d_test.evaluation_data_with_abstracts_citations()
        query_test_authors = d_test_authors.test_data_with_abstracts_citations(
                ).author_names().data[["author_name", "chapter"]]
        return query_test, query_test_authors, truth

    def _choose_classifier(self, classifier_name):
        if classifier_name == "KNN":
            classifier = KNeighborsClassifier(n_neighbors=30, n_jobs=10)
        elif classifier_name == "MLP":
            classifier = MLPClassifier(random_state=0, verbose=True)
        elif classifier_name == "MultinomialLogisticRegression":
            classifier = LogisticRegression(penalty="l2",
                                            random_state=0,
                                            solver="saga",
                                            multi_class="multinomial",
                                            verbose=1,
                                            n_jobs=10)
        else:
            raise ValueError("Classifier name not recognised.")
        return classifier

    def _load_training_graph(self, graph_type):
        graph_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, graph_type + "/train_val-G.json")
        if os.path.isfile(graph_file):
            with open(graph_file) as f:
                G_train = json_graph.node_link_graph(json.load(f))
            return G_train
        else:
            raise ValueError("The {} training graph does not exist.".format(
                    graph_type))

    def _load_training_walks(self, graph_type, G_train):
        walks_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, graph_type + "/train_val-walks.txt")
        walks = []
        if isinstance(list(G_train.nodes)[0], int):
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n

        if os.path.isfile(walks_file):
            with open(walks_file) as f:
                for line in f:
                    walks.append(map(conversion, line.split()))
            return walks
        else:
            raise ValueError("The {} walks do not exist.".format(graph_type))

    def _load_train_embeddings(self, model):
        embeddings_file = os.path.join(model._log_dir(), "embeddings.npy")
        embeddings_ids_file = os.path.join(model._log_dir(),
                                           "embeddings_ids.txt")
        if os.path.isfile(embeddings_file) and os.path.isfile(
                embeddings_ids_file):
            pretrained_embeddings = np.load(embeddings_file)
            pretrained_embeddings_id_map = {}
            with open(embeddings_ids_file) as f:
                for i, line in enumerate(f):
                    pretrained_embeddings_id_map[line.strip()] = i
            return pretrained_embeddings, pretrained_embeddings_id_map
        else:
            raise ValueError(
                    "The pretrained citations embeddings are missing.")

    def _load_model_classifier(self):
        if os.path.isfile(self.classifier_file):
            print("Loading classifier...")
            with open(self.classifier_file, "rb") as f:
                self.label_encoder, self.labels, self.classifier = pickle.load(
                        f)
                print("Loaded.")
                return True
        return False

    def _save_model_classifier(self):
        with open(self.classifier_file, "wb") as f:
            pickle.dump([self.label_encoder, self.labels, self.classifier],
                        f, protocol=4)

    def _has_persistent_model(self):
        if os.path.isfile(self.classifier_file):
            return True
        return False

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for unsupervised GraphSAGE model.')
        parser.add_argument("classifier_name",
                            choices=["KNN", "MLP",
                                     "MultinomialLogisticRegression"],
                            help="The name of the classifier.")
        parser.add_argument('embedding_type',
                            choices=["AVG_L", "AVG_2L", "AVG_SUM_L4",
                                     "AVG_SUM_ALL", "MAX_2L",
                                     "CONC_AVG_MAX_2L", "CONC_AVG_MAX_SUM_L4",
                                     "SUM_L", "SUM_2L"
                                     ],
                            help="Type of embedding.")
        parser.add_argument('model_checkpoint_citations',
                            help='Name of the GraphSAGE model checkpoint ' +
                            'for the citations graph.')
        parser.add_argument('model_checkpoint_authors',
                            help='Name of the GraphSAGE model checkpoint ' +
                            'for the authors graph.')
        parser.add_argument('train_prefix_citations',
                            help='Name of the object file that stores the '
                            + 'citations training data.')
        parser.add_argument('train_prefix_authors',
                            help='Name of the object file that stores the '
                            + 'authors training data.')
        parser.add_argument('model_name',
                            choices=["graphsage_mean", "gcn", "graphsage_seq",
                                     "graphsage_maxpool", "graphsage_meanpool"
                                     ],
                            help="Model names.")
        parser.add_argument('--model_size',
                            choices=["small", "big"],
                            default="small",
                            help="Can be big or small; model specific def'ns")
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.00001,
                            help='Initial learning rate.')
        parser.add_argument('--epochs',
                            type=int,
                            default=10,
                            help='Number of epochs to train.')
        parser.add_argument('--dropout',
                            type=float,
                            default=0.0,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=0.0,
                            help='Weight for l2 loss on embedding matrix.')
        parser.add_argument('--max_degree',
                            type=int,
                            default=100,
                            help='Maximum node degree.')
        parser.add_argument('--samples_1',
                            type=int,
                            default=25,
                            help='Number of samples in layer 1.')
        parser.add_argument('--samples_2',
                            type=int,
                            default=10,
                            help='Number of users samples in layer 2.')
        parser.add_argument('--dim_1',
                            type=int,
                            default=128,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--dim_2',
                            type=int,
                            default=128,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--random_context',
                            action="store_false",
                            default=True,
                            help='Whether to use random context or direct ' +
                            'edges.')
        parser.add_argument('--neg_sample_size',
                            type=int,
                            default=20,
                            help='Number of negative samples.')
        parser.add_argument('--batch_size',
                            type=int,
                            default=512,
                            help='Minibatch size.')
        parser.add_argument('--identity_dim',
                            type=int,
                            default=0,
                            help='Set to positive value to use identity ' +
                            'embedding features of that dimension.')
        parser.add_argument('--save_embeddings',
                            action="store_true",
                            default=False,
                            help='Whether to save embeddings for all nodes ' +
                            'after training')
        parser.add_argument('--base_log_dir',
                            default='../../../data/processed/graphsage/',
                            help='Base directory for logging and saving ' +
                            'embeddings')
        parser.add_argument('--validate_iter',
                            type=int,
                            default=5000,
                            help='How often to run a validation minibatch.')
        parser.add_argument('--validate_batch_size',
                            type=int,
                            default=256,
                            help='How many nodes per validation sample.')
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        parser.add_argument('--print_every',
                            type=int,
                            default=50,
                            help='How often to print training info.')
        parser.add_argument('--max_total_steps',
                            type=int,
                            default=10**10,
                            help='Maximum total number of iterations.')
        parser.add_argument('--log_device_placement',
                            action="store_true",
                            default=False,
                            help='Whether to log device placement.')
        parser.add_argument('--recs',
                            type=int,
                            default=10,
                            help='Number of recommendations.')
        args = parser.parse_args()

        print("Starting evaluation...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print("Using GPU {}.".format(str(args.gpu)))

        from GraphSAGEClassifierConcatEvaluation import GraphSAGEClassifierConcatEvaluation
        evaluation_model = GraphSAGEClassifierConcatEvaluation(
                args.classifier_name, args.embedding_type, args.model_name,
                args.model_size, args.learning_rate, args.gpu, args.recs)

        # Initialize GraphSAGE models
        graphsage_model_citations = UnsupervisedModel(
                args.train_prefix_citations, args.model_name, args.model_size,
                args.learning_rate, args.epochs, args.dropout,
                args.weight_decay, args.max_degree, args.samples_1,
                args.samples_2, args.dim_1, args.dim_2, args.random_context,
                args.neg_sample_size, args.batch_size, args.identity_dim,
                args.save_embeddings, args.base_log_dir, args.validate_iter,
                args.validate_batch_size, args.gpu, args.print_every,
                args.max_total_steps, args.log_device_placement)
        graphsage_model_authors = UnsupervisedModel(
                args.train_prefix_authors, args.model_name, args.model_size,
                args.learning_rate, args.epochs, args.dropout,
                args.weight_decay, args.max_degree, args.samples_1,
                args.samples_2, args.dim_1, args.dim_2, args.random_context,
                args.neg_sample_size, args.batch_size, args.identity_dim,
                args.save_embeddings, args.base_log_dir, args.validate_iter,
                args.validate_batch_size, args.gpu, args.print_every,
                args.max_total_steps, args.log_device_placement)

        # Train model if needed:
        if not evaluation_model._has_persistent_model():
            print("Classifier not trained yet. Training now...")
            timer = Timer()
            timer.tic()
            evaluation_model.train(graphsage_model_citations,
                                   graphsage_model_authors)
            print("Training finished.")
            timer.toc()
        else:
            evaluation_model._load_model_classifier()

        # Load test data
        print("Loading test data...")
        query_test, query_test_authors, truth = evaluation_model.load_data()
        print("Loaded.")

        # Infer embeddings
        print("Inferring embeddings for citations graph.")
        queue_citations = mp.Queue()
        process_citations = mp.Process(
                target=evaluation_model.infer_embeddings,
                args=(query_test, None, "citations", graphsage_model_citations,
                      args.model_checkpoint_citations, queue_citations))
        process_citations.start()
        embeddings_citations = queue_citations.get()
        process_citations.join()
        process_citations.terminate()

        print("Inferring embeddings for authors graphs.")
        queue_authors = mp.Queue()
        process_authors = mp.Process(
                target=evaluation_model.infer_embeddings,
                args=(query_test, query_test_authors, "authors",
                      graphsage_model_authors, args.model_checkpoint_authors,
                      queue_authors))
        process_authors.start()
        embeddings_authors = queue_authors.get()
        process_authors.join()
        process_authors.terminate()

        # Concatenate embeddings
        test_embeddings = np.concatenate(
                (embeddings_citations, embeddings_authors), axis=1)

        print("Computing predictions...")
        recommendation = evaluation_model.compute_predictions(test_embeddings)
        print("Predictions computed.")

        # Evaluate
        print("Evaluating...")
        evaluation = EvaluationContainer()
        evaluation.evaluate(recommendation, truth)
        print("Finsihed.")

    if __name__ == "__main__":
        main()
