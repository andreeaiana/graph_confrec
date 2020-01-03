# -*- coding: utf-8 -*-
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
from GraphSAGERLModel import GraphSAGERLModel


class GraphSAGERLModelEvaluation():

    def __init__(self, embedding_type, graph_type, train_prefix, model_name,
                 nonlinear_sampler=True, fast_ver=False, allhop_rewards=False,
                 model_size="small", learning_rate=0.001, epochs=10,
                 dropout=0.0, weight_decay=0.0, max_degree=100, samples_1=25,
                 samples_2=10, samples_3=0, dim_1=512, dim_2=512, dim_3=0,
                 batch_size=128, sigmoid=False, identity_dim=0,
                 base_log_dir='../../../data/processed/graphsage_rl/',
                 validate_iter=5000, validate_batch_size=128, gpu=0,
                 print_every=5, max_total_steps=10**10,
                 log_device_placement=False, recs=10, threshold=2):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.graph_type = graph_type

        self.d = DataLoader()
        if self.graph_type == "citations_authors_het_edges":
            self.d_authors = DataLoader()

        self.model = GraphSAGERLModel(
                embedding_type, graph_type, train_prefix, model_name,
                nonlinear_sampler, fast_ver, allhop_rewards, model_size,
                learning_rate, epochs, dropout, weight_decay, max_degree,
                samples_1, samples_2, samples_3, dim_1, dim_2, dim_3,
                batch_size, sigmoid, identity_dim, base_log_dir,
                validate_iter, validate_batch_size, gpu, print_every,
                max_total_steps, log_device_placement, recs, threshold)

    def evaluate(self):
        if self.graph_type == "citations":
            # Load test data
            query_test, truth = self.d.evaluation_data_with_abstracts_citations()
            # Retrieve predictions
            recommendation = self.model.query_batch(query_test)

        if self.graph_type == "citations_authors_het_edges":
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
                description='Arguments for supervised GraphSAGE_RL model ' +
                'evaluation.')
        parser.add_argument('embedding_type',
                            choices=["AVG_L", "AVG_2L", "AVG_SUM_L4",
                                     "AVG_SUM_ALL", "MAX_2L",
                                     "CONC_AVG_MAX_2L", "CONC_AVG_MAX_SUM_L4",
                                     "SUM_L", "SUM_2L"
                                     ],
                            help="Type of embedding.")
        parser.add_argument('graph_type',
                            choices=["citations", "authors",
                                     "citations_authors_het_edges"],
                            help="The type of graph used.")
        parser.add_argument('train_prefix',
                            help='Name of the object file that stores the '
                            + 'training data.')
        parser.add_argument("model_name",
                            choices=["mean_concat", "mean_add", "gcn",
                                     "graphsage_seq", "graphsage_maxpool",
                                     "graphsage_meanpool"
                                     ],
                            help="Model names.")
        parser.add_argument("--nonlinear_sampler",
                            action="store_false",
                            default=True,
                            help="Where to use nonlinear sampler o.w. " +
                            "linear sampler"
                            )
        parser.add_argument("--fast_ver",
                            action="store_true",
                            default=False,
                            help="Whether to use a fast version of the " +
                            "nonlinear sampler"
                            )
        parser.add_argument("--allhop_rewards",
                            action="store_true",
                            default=False,
                            help="Whether to use a all-hop rewards or " +
                            "last-hop reward for training the nonlinear " +
                            "sampler"
                            )
        parser.add_argument('--model_size',
                            choices=["small", "big"],
                            default="small",
                            help="Can be big or small; model specific def'ns")
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.001,
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
        parser.add_argument('--samples_3',
                            type=int,
                            default=0,
                            help='Number of users samples in layer 3. ' +
                            '(Only for mean model)')
        parser.add_argument('--dim_1',
                            type=int,
                            default=512,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--dim_2',
                            type=int,
                            default=512,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--dim_3',
                            type=int,
                            default=0,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--batch_size',
                            type=int,
                            default=128,
                            help='Minibatch size.')
        parser.add_argument('--sigmoid',
                            action="store_true",
                            default=False,
                            help='Whether to use sigmoid loss ')
        parser.add_argument('--identity_dim',
                            type=int,
                            default=0,
                            help='Set to positive value to use identity ' +
                            'embedding features of that dimension.')
        parser.add_argument('--base_log_dir',
                            default='../../../data/processed/graphsage_rl/',
                            help='Base directory for logging and saving ' +
                            'embeddings')
        parser.add_argument('--validate_iter',
                            type=int,
                            default=5000,
                            help='How often to run a validation minibatch.')
        parser.add_argument('--validate_batch_size',
                            type=int,
                            default=128,
                            help='How many nodes per validation sample.')
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        parser.add_argument('--print_every',
                            type=int,
                            default=5,
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
        parser.add_argument('--threshold',
                            type=int,
                            default=2,
                            help='Threshold for edge weights in ' +
                            'heterogeneous graph.')
        args = parser.parse_args()

        from GraphSAGERLModelEvaluation import GraphSAGERLModelEvaluation
        print("Starting...")
        model = GraphSAGERLModelEvaluation(
                args.embedding_type, args.graph_type, args.train_prefix,
                args.model_name, args.nonlinear_sampler, args.fast_ver,
                args.allhop_rewards, args.model_size, args.learning_rate,
                args.epochs, args.dropout, args.weight_decay, args.max_degree,
                args.samples_1, args.samples_2, args.samples_3, args.dim_1,
                args.dim_2, args.dim_3, args.batch_size, args.sigmoid,
                args.identity_dim, args.base_log_dir, args.validate_iter,
                args.validate_batch_size, args.gpu, args.print_every,
                args.max_total_steps, args.log_device_placement, args.recs)
        model.evaluate()
        print("Finished.")

    if __name__ == "__main__":
        main()
