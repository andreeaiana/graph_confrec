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

class Processor:

    def __init__(self, embedding_type, gpu=0):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.embedding_type = embedding_type
        self.embeddings_parser = EmbeddingsParser(gpu)
        self.path_persistent_scibert = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "scibert_arga",
                self.embedding_type)
        if not os.path.exists(self.path_persistent_scibert):
            os.makedirs(self.path_persistent_scibert)

    def training_data_scibert(self):
        print("Creating training files.\n")

        # Load training and validation data
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        train_val_data = pd.concat((df_train, df_validation),
                                   axis=0).reset_index(drop=True)
        scibert_embeddings = self._scibert_embeddings(train_val_data)
        print("Saving embeddings to disk...")
        scibert_embeddings_file = os.path.join(self.path_persistent_scibert,
                                               "scibert_embeddings.pkl")
        with open(scibert_embeddings_file, "wb") as f:
            pickle.dump(scibert_embeddings, f)
        print("Saved.")

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
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        args = parser.parse_args()
        print("Starting...")
        from preprocess_data import Processor
        processor = Processor(args.embedding_type, args.gpu)
        processor.training_data_scibert()
        print("Finished.")

    if __name__ == "__main__":
        main()


