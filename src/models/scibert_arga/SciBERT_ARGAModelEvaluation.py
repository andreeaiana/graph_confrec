# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.join(os.getcwd(), "..", "evaluations"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))

from DataLoader import DataLoader
from EvaluationContainer import EvaluationContainer
from SciBERT_ARGAModel import SciBERT_ARGAModel


class SciBERT_ARGAModelEvaluation:

    def __init__(self, embedding_type, dataset, arga_model_name,
                 graph_type="directed", n_latent=16, learning_rate=0.001,
                 weight_decay=0, dropout=0, dis_loss_para=1, reg_loss_para=1,
                 epochs=200, gpu=None, ffnn_hidden_dim=100, recs=10):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.dataset = dataset

        self.d = DataLoader()
        if self.dataset == "citations_authors_het_edges":
            self.d_authors = DataLoader()

        self.model = SciBERT_ARGAModel(
                embedding_type, dataset, arga_model_name, graph_type, n_latent,
                learning_rate, weight_decay, dropout, dis_loss_para,
                reg_loss_para, epochs, gpu, ffnn_hidden_dim, recs)

    def evaluate(self):
        if self.dataset == "citations":
            # Load test data
            query_test, truth = self.d.evaluation_data_with_abstracts_citations()

            # Retrieve predictions
            recommendation = self.model.query_batch(query_test)
        elif self.dataset == "citations_authors_het_edges":
            # Load test data
            query_test, truth = self.d.evaluation_data_with_abstracts_citations()
            query_test_authors = self.d_authors.test_data_with_abstracts_citations(
                    ).author_names().data[["author_name", "chapter"]]
            # Retrieve predictions
            recommendation = self.model.query_batch((query_test,
                                                     query_test_authors))
        else:
            raise ValueError("Dataset not recognised.")

        # Evaluate
        print("Evaluating...")
        evaluation = EvaluationContainer()
        evaluation.evaluate(recommendation, truth)

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
        parser.add_argument('model_name',
                            choices=["ARGA", "ARGVA"],
                            help="Type of model.")
        parser.add_argument('--graph_type',
                            choices=["directed", "undirected"],
                            default="directed",
                            help='The type of graph used ' +
                            '(directed vs. undirected).')
        parser.add_argument("--n_latent",
                            type=int,
                            default=16,
                            help="Number of units in hidden layer.")
        parser.add_argument("--learning_rate",
                            type=float,
                            default=0.001,
                            help="Initial learning rate.")
        parser.add_argument("--weight_decay",
                            type=float,
                            default=0,
                            help="Weight for L2 loss on embedding matrix.")
        parser.add_argument("--dropout",
                            type=float,
                            default=0,
                            help="Dropout rate (1 - keep probability).")
        parser.add_argument("--dis_loss_para",
                            type=float,
                            default=1)
        parser.add_argument("--reg_loss_para",
                            type=float,
                            default=1)
        parser.add_argument("--epochs",
                            type=int,
                            default=200,
                            help="Number of epochs.")
        parser.add_argument('--gpu',
                            type=int,
                            help='Which gpu to use.')
        parser.add_argument("--ffnn_hidden_dim",
                            type=int,
                            default=100,
                            help="Number of units in hidden layer of the FFNN."
                            )
        parser.add_argument('--recs',
                            type=int,
                            default=10,
                            help='Number of recommendations.')
        args = parser.parse_args()
        print("Starting...")
        from SciBERT_ARGAModelEvaluation import SciBERT_ARGAModelEvaluation
        model = SciBERT_ARGAModelEvaluation(
                args.embedding_type, args.dataset, args.model_name,
                args.graph_type,  args.n_latent, args.learning_rate,
                args.weight_decay, args.dropout, args.dis_loss_para,
                args.reg_loss_para, args.epochs, args.gpu, args.ffnn_hidden_dim
                )
        model.evaluate()
        print("Finished.")

    if __name__ == "__main__":
        main()
