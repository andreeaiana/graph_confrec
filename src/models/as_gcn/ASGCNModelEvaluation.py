# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.getcwd(), "..", "evaluations"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))

from DataLoader import DataLoader
from EvaluationContainer import EvaluationContainer
from ASGCNModel import ASGCNModel


class ASGCNModelEvaluation:

    def __init__(self, embedding_type, dataset, model_name, max_degree=696,
                 learning_rate=0.001, weight_decay=5e-4, dropout=0.0,
                 epochs=300, early_stopping=30, hidden1=16, rank=128, skip=0,
                 var=0.5, sampler_device="cpu", gpu=None, recs=10):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.d = DataLoader()
        self.model = ASGCNModel(embedding_type, dataset, model_name,
                                max_degree, learning_rate, weight_decay,
                                dropout, epochs, early_stopping, hidden1,
                                rank, skip, var, sampler_device, gpu, recs)

    def evaluate(self):
        # Load test data
        query_test, truth = self.d.evaluation_data_with_abstracts_citations()

        # Retrieve predictions
        recommendation = self.model.query_batch(query_test)

        # Evaluate
        print("Evaluating...")
        evaluation = EvaluationContainer()
        evaluation.evaluate(recommendation, truth)

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for ASGCN model evaluation.')
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
        parser.add_argument("model_name",
                            choices=["gcn_adapt", "gcn_adapt_mix"],
                            help="Model names.")
        parser.add_argument('--max_degree',
                            type=int,
                            default=696,
                            help='Maximum degree for constructing the ' +
                                 'adjacent matrix.')
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.001,
                            help='Learning rate.')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=5e-4,
                            help='Weight decay.')
        parser.add_argument('--dropout',
                            type=float,
                            default=0.0,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--epochs',
                            type=int,
                            default=300,
                            help='Number of epochs to train.')
        parser.add_argument('--early_stopping',
                            type=int,
                            default=30,
                            help='Tolerance for early stopping (# of epochs).')
        parser.add_argument("--hidden1",
                            type=int,
                            default=16,
                            help="Number of units in hidden layer 1.")
        parser.add_argument("--rank",
                            type=int,
                            default=128,
                            help="The number of nodes per layer.")
        parser.add_argument('--skip',
                            type=float,
                            default=0,
                            help='If use skip connection.')
        parser.add_argument('--var',
                            type=float,
                            default=0.5,
                            help='If use variance reduction.')
        parser.add_argument("--sampler_device",
                            choices=["gpu", "cpu"],
                            default="cpu",
                            help="The device for sampling: cpu or gpu.")
        parser.add_argument('--gpu',
                            type=int,
                            help='Which gpu to use.')
        parser.add_argument('--recs',
                            type=int,
                            default=10,
                            help='Number of recommendations.')
        args = parser.parse_args()

        from ASGCNModelEvaluation import ASGCNModelEvaluation
        print("Starting...")
        model = ASGCNModelEvaluation(
                args.embedding_type, args.dataset, args.model_name,
                args.max_degree, args.learning_rate, args.weight_decay,
                args.dropout, args.epochs, args.early_stopping, args.hidden1,
                args.rank, args.skip, args.var, args.sampler_device, args.gpu,
                args.recs)
        model.evaluate()
        print("Finished.")

    if __name__ == "__main__":
        main()
