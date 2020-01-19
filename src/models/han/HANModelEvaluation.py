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
from HANModel import HANModel


class HANModelEvaluation:

    def __init__(self, model, embedding_type, hid_units=[64], n_heads=[8, 1],
                 learning_rate=0.005, weight_decay=0, epochs=200, batch_size=1,
                 patience=100, residual=False, nonlinearity=tf.nn.elu,
                 ffd_drop=0.5, attn_drop=0.5, gpu=0, recs=10):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.d = DataLoader()
        self.d_authors = DataLoader()

        self.model = HANModel(model, embedding_type, hid_units, n_heads,
                              learning_rate, weight_decay, epochs, batch_size,
                              patience, residual, nonlinearity, ffd_drop,
                              attn_drop, gpu, recs)

    def evaluate(self):
        # Load test data
        query_test, truth = self.d.evaluation_data_with_abstracts_citations()
        query_test_authors = self.d_authors.test_data_with_abstracts_citations(
                ).author_names().data[["author_name", "chapter"]]
        # Retrieve predictions
        recommendation = self.model.query_batch((query_test,
                                                 query_test_authors))

        # Evaluate
        print("Evaluating...")
        evaluation = EvaluationContainer()
        evaluation.evaluate(recommendation, truth)

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for HAN model evaluation.')
        parser.add_argument("model",
                            choices=["HeteGAT", "HeteGAT_multi"],
                            help="The type of model used.")
        parser.add_argument('embedding_type',
                            choices=["AVG_L", "AVG_2L", "AVG_SUM_L4",
                                     "AVG_SUM_ALL", "MAX_2L",
                                     "CONC_AVG_MAX_2L", "CONC_AVG_MAX_SUM_L4",
                                     "SUM_L", "SUM_2L"
                                     ],
                            help="Type of embedding.")
        parser.add_argument("--hid_units",
                            type=int,
                            nargs="+",
                            default=[64],
                            help="Number of hidden units per each attention "
                            + "head in each layer.")
        parser.add_argument('--n_heads',
                            type=int,
                            nargs="+",
                            default=[8, 1],
                            help='Additional entry for the output layer.')
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.005,
                            help='Learning rate.')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=0,
                            help='Weight decay.')
        parser.add_argument('--epochs',
                            type=int,
                            default=200,
                            help='Number of epochs to train.')
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='Batch size.')
        parser.add_argument('--patience',
                            type=int,
                            default=100)
        parser.add_argument('--residual',
                            action="store_true",
                            default=False)
        parser.add_argument('--nonlinearity',
                            help="Type of activation used")
        parser.add_argument('--ffd_drop',
                            type=float,
                            default=0.5)
        parser.add_argument('--attn_drop',
                            type=float,
                            default=0.5)
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        parser.add_argument('--recs',
                            type=int,
                            default=10,
                            help='Number of recommendations.')
        args = parser.parse_args()

        from HANModelEvaluation import HANModelEvaluation
        print("Starting...")
        model = HANModelEvaluation(
                args.model, args.embedding_type, args.hid_units, args.n_heads,
                args.learning_rate, args.weight_decay, args.epochs,
                args.batch_size, args.patience, args.residual,
                args.nonlinearity, args.ffd_drop, args.attn_drop, args.gpu,
                args.recs)
        model.evaluate()
        print("Finished.")

    if __name__ == "__main__":
        main()
