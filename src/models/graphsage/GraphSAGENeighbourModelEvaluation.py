# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "..", "evaluations"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))

from DataLoader import DataLoader
from EvaluationContainer import EvaluationContainer
from GraphSAGENeighbourModel import GraphSAGENeighbourModel


class GraphSAGENeighbourModelEvaluation():

    def __init__(self, embedding_type, model_checkpoint, train_prefix,
                 model_name, model_size="small", learning_rate=0.00001,
                 epochs=10, dropout=0.0, weight_decay=0.0, max_degree=100,
                 samples_1=25, samples_2=10, dim_1=128, dim_2=128,
                 random_context=True, neg_sample_size=20, batch_size=512,
                 identity_dim=0, save_embeddings=False,
                 base_log_dir='../../../data/processed/graphsage/',
                 validate_iter=5000, validate_batch_size=256, gpu=0,
                 print_every=50, max_total_steps=10**10,
                 log_device_placement=False, recs=10):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        self.model = GraphSAGENeighbourModel(
                embedding_type, model_checkpoint, train_prefix, model_name,
                model_size, learning_rate, epochs, dropout, weight_decay,
                max_degree, samples_1, samples_2, dim_1, dim_2, random_context,
                neg_sample_size, batch_size, identity_dim, save_embeddings,
                base_log_dir, validate_iter, validate_batch_size, gpu,
                print_every, max_total_steps, log_device_placement, recs)

    def evaluate(self):
        # Load test data
        d_test = DataLoader()
        query_test, truth = evaluation_data_with_abstracts_citations()

        # Retrieve predictions
        recommendation = self.model.query_batch(query_test)

        # Evaluate
        print("Evaluating...")
        evaluation = EvaluationContainer()
        evaluation.evaluate(recommendation, truth)

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for unsupervised GraphSAGE model.')
        parser.add_argument("embedding_type",
                            choices=["AVG_L", "AVG_2L", "AVG_SUM_L4",
                                     "AVG_SUM_ALL", "MAX_2L",
                                     "CONC_AVG_MAX_2L", "CONC_AVG_MAX_SUM_L4",
                                     "SUM_L", "SUM_2L"
                                     ],
                            help="Type of embedding.")
        parser.add_argument('model_checkpoint',
                            help='Name of the GraphSAGE model checkpoint.')
        parser.add_argument('train_prefix',
                            help='Name of the object file that stores the '
                            + 'training data.')
        parser.add_argument("model_type",
                            choices=["graphsage_mean", "gcn", "graphsage_seq",
                                     "graphsage_maxpool", "graphsage_meanpool"
                                     ],
                            help="Model names.")
        parser.add_argument('--model_size',
                            choices=["small", "big"],
                            default="small",
                            help="Can be big or small; model specific def'ns")
        parser.add_argument('--learning_rate',
                            type=int,
                            default=0.00001,
                            help='Initial learning rate.')
        parser.add_argument('--epochs',
                            type=int,
                            default=10,
                            help='Number of epochs to train.')
        parser.add_argument('--dropout',
                            type=int,
                            default=0.0,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--weight_decay',
                            type=int,
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
                            action="store_false",
                            default=True,
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

        from GraphSAGENeighbourModelEvaluation import GraphSAGENeighbourModelEvaluation
        print("Starting...")
        model = UnsupervisedModel(args.embedding_type, args.model_checkpoint,
                                  args.train_prefix, args.model_name,
                                  args.model_size, args.learning_rate,
                                  args.epochs, args.dropout, args.weight_decay,
                                  args.max_degree, args.samples_1,
                                  args.samples_2, args.dim_1, args.dim_2,
                                  args.random_context, args.neg_sample_size,
                                  args.batch_size, args.identity_dim,
                                  args.save_embeddings, args.base_log_dir,
                                  args.validate_iter, args.validate_batch_size,
                                  args.gpu, args.print_every,
                                  args.max_total_steps,
                                  args.log_device_placement, args.recs)
        model.evaluate()
        print("Finished.")

    if __name__ == "__main__":
        main()
