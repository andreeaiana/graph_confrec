# -*- coding: utf-8 -*-
import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
from DataLoader import DataLoader
from SciBERTEmbeddingsParser import EmbeddingsParser
from arga import ARGAModel


class Processor:

    def __init__(self, embedding_type, dataset, arga_model_name,
                 graph_type="directed", mode="train", n_latent=16,
                 learning_rate=0.001, weight_decay=0, dropout=0,
                 dis_loss_para=1, reg_loss_para=1, epochs=200, gpu=None):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.embedding_type = embedding_type
        self.embeddings_parser = EmbeddingsParser(gpu)
        self.arga_model = ARGAModel(
                self.embedding_type, dataset, arga_model_name, graph_type,
                mode, n_latent, learning_rate, weight_decay, dropout,
                dis_loss_para, reg_loss_para, epochs, gpu)

        self.path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "scibert_arga",
                self.embedding_type)
        if not os.path.exists(self.path_persistent):
            os.makedirs(self.path_persistent)

    def training_data_arga(self):
        print("Creating ARGA training files.\n")
        arga_data = self.arga_model.data
        arga_embeddings = self.arga_model.test(arga_data)
        self.arga_model.save_embeddings(arga_embeddings)
        print("ARGA training files created.\n")

    def training_data_scibert(self):
        print("Creating SciBERT training files.\n")

        # Load training and validation data
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        train_val_data = pd.concat((df_train, df_validation),
                                   axis=0).reset_index(drop=True)
        scibert_embeddings = self._scibert_embeddings(train_val_data)
        print("Saving SciBERT embeddings to disk...")
        scibert_embeddings_file = os.path.join(self.path_persistent,
                                               "scibert_embeddings.pkl")
        with open(scibert_embeddings_file, "wb") as f:
            pickle.dump(scibert_embeddings, f)
        print("Saved.\n")
        print("SciBERT training files created.")

    def _scibert_embeddings(self, data):
        features = []
        with tqdm(desc="Computing SciBERT embeddings: ",
                  total=len(data)) as pbar:
            for idx in range(len(data)):
                features.append(self.embeddings_parser.embed_sequence(
                                data.chapter_abstract.iloc[idx],
                                self.embedding_type))
                pbar.update(1)
        print("Computed.")
        return np.array(features)

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
        parser.add_argument('arga_model_name',
                            choices=["ARGA", "ARGVA"],
                            help="Type of model.")
        parser.add_argument('--graph_type',
                            choices=["directed", "undirected"],
                            default="directed",
                            help='The type of graph used ' +
                            '(directed vs. undirected).')
        parser.add_argument('--mode',
                            choices=["train", "test"],
                            default="train",
                            help="Whether to set the ARGA net to " +
                            "training mode.")
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
        args = parser.parse_args()
        print("Starting...")
        from preprocess_data_scibert_arga import Processor
        processor = Processor(
                args.embedding_type, args.dataset, args.arga_model_name,
                args.graph_type, args.mode, args.n_latent, args.learning_rate,
                args.weight_decay, args.dropout, args.dis_loss_para,
                args.reg_loss_para, args.epochs, args.gpu)
        processor.training_data_scibert()
        processor.training_data_arga()
        print("Finished.")

    if __name__ == "__main__":
        main()
