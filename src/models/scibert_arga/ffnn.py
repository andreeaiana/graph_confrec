# -*- coding: utf-8 -*-
import os
import sys
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from DataLoader import DataLoader
from TimerCounter import Timer


class FeedfowardNeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedfowardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FFNNModel:

    def __init__(self, embedding_type, dataset, arga_model_name, n_latent=16,
                 learning_rate=0.001, weight_decay=0, dropout=0,
                 dis_loss_para=1, reg_loss_para=1, epochs=200, gpu=None,
                 ffnn_hidden_dim=100):

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.arga_model_name = arga_model_name
        self.n_latent = n_latent
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.dis_loss_para = dis_loss_para
        self.reg_loss_para = reg_loss_para
        self.epochs = epochs

        # Set device
        if torch.cuda.is_available() and gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print("Using GPU device: {}.".format(str(gpu)))
            self.device = torch.device("cuda:" + str(gpu))
        else:
            self.device = "cpu"

        self.model_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "processed", "scibert_arga",
                self.embedding_type, self.dataset)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_file = f'SciBERT_{arga_model_name}_{n_latent}_{learning_rate}_{weight_decay}_{dropout}.pt'
        self.model_path = os.path.join(self.model_dir, model_file)

        # Load training data
        self.training_data, self.validation_data, self.training_labels, self.validation_labels = self._load_training_data()
        self.training_data = self.training_data.to(self.device)
        self.validation_data = self.validation_data.to(self.device)
        self.training_labels = self.training_labels.to(self.device)
        self.validation_labels = self.validation_labels.to(self.device)

        # Initialize model
        self.model = FeedfowardNeuralNetwork(
                input_dim=self.training_data.shape[1],
                hidden_dim=ffnn_hidden_dim,
                output_dim=len(range(max(self.training_labels))) + 1)
        if self.device is not "cpu":
            self.model.to(self.device)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

    def train(self):
        # Make the datasets iterable
        batch_size = 10000

        train_data_loader = torch.utils.data.DataLoader(
                dataset=self.training_data, batch_size=batch_size)
        validation_data_loader = torch.utils.data.DataLoader(
                dataset=self.validation_data, batch_size=batch_size)
        train_labels_loader = torch.utils.data.DataLoader(
                dataset=self.training_labels, batch_size=batch_size)
        validation_labels_loader = torch.utils.data.DataLoader(
                dataset=self.validation_labels, batch_size=batch_size)

        # Train the model
        timer = Timer()
        timer.tic()

        mean_train_losses = []
        mean_validation_losses = []

        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch + 1))
            train_losses = []
            validation_losses = []
            self.model.train()

            for i, (train_data, train_labels) in enumerate(zip(
                    train_data_loader, train_labels_loader)):
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(train_data)
                loss = self.cross_entropy_loss(outputs.squeeze(), train_labels)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # Compute validation loss
                self.model.eval()
                with torch.no_grad():
                    for _, (val_data, val_labels) in enumerate(zip(
                            validation_data_loader, validation_labels_loader)):
                        val_pred = self.model(val_data)
                        val_loss = self.cross_entropy_loss(
                                val_pred.squeeze(), val_labels)
                        validation_losses.append(val_loss.item())

            print("\tTrain loss: {}, validation loss: {}".format(
                    np.mean(train_losses), np.mean(validation_losses)))
            mean_train_losses.append(np.mean(train_losses))
            mean_validation_losses.append(np.mean(validation_losses))
            if mean_validation_losses[-1] == min(mean_validation_losses):
                print("\tSaving model...")
                torch.save(self.model.state_dict(),
                           self.model_path)
                print("\tSaved.")

        print("Finished training.")
        training_time = timer.toc()
        self._plot_losses(mean_train_losses, mean_validation_losses)
        self._print_stats(mean_train_losses, mean_validation_losses,
                          training_time)

    def test(self, scibert_emb, arga_emb):
        # Preprocessing test data
        scibert_emb = torch.FloatTensor(scibert_emb)
        arga_emb = torch.FloatTensor(arga_emb)
        test_data = torch.cat((scibert_emb, arga_emb), dim=1)

        print("Loading model...")
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print("Loaded.\n")
        except Exception as e:
            print("Could not load model from {} ({})".format(
                    self.model_path, e))

        self.model.eval()

        print("Computing predictions...")
        test_data = test_data.to(self.device)
        with torch.no_grad():
            predictions = self.model.forward(test_data)
            predictions = predictions.cpu().detach().numpy()
        print("Computed.")

        return predictions

    def _load_training_data(self):
        # load data
        print("Loading training data...")
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        scibert_embeddings = self._load_scibert_embeddings()
        arga_embeddings = self._load_arga_embeddings()
        print("Loaded.")

        print("Preprocessing training data...")
        # Process labels
        train_val_data = pd.concat((df_train, df_validation),
                                   axis=0).reset_index(drop=True)
        labels = self._get_training_labels(train_val_data)

        # Transform data to tensors
        scibert_embeddings = torch.FloatTensor(scibert_embeddings)
        arga_embeddings = torch.FloatTensor(arga_embeddings)
        labels = torch.LongTensor(labels)

        # Concatenate embeddings
        data = torch.cat((scibert_embeddings, arga_embeddings), dim=1)
        training_data = data[:len(df_train)]
        validation_data = data[len(df_train):]
        training_labels = labels[:len(df_train)]
        validation_labels = labels[len(df_train):]
        training_labels = torch.max(training_labels, 1)[1]
        validation_labels = torch.max(validation_labels, 1)[1]
        print("Preprocessed.")

        return training_data, validation_data, training_labels, validation_labels

    def _load_scibert_embeddings(self):
        scibert_embeddings_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "scibert_arga",
                self.embedding_type, "scibert_embeddings.pkl")
        if os.path.isfile(scibert_embeddings_file):
            print("Loading SciBERT embeddings...")
            with open(scibert_embeddings_file, "rb") as f:
                scibert_embeddings = pickle.load(f)
            print("Loaded.")
            return scibert_embeddings

    def _load_arga_embeddings(self):
        filename = f'{self.dataset}_{self.arga_model_name}_embeddings_{self.n_latent}_{self.learning_rate}_{self.weight_decay}_{self.dropout}.pkl'
        embeddings_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "scibert_arga",
                self.embedding_type, filename)
        if os.path.isfile(embeddings_file):
            print("Loading ARGA embeddings...")
            with open(embeddings_file, "rb") as f:
                arga_embeddings = pickle.load(f)
                print("Loaded.")
                return arga_embeddings
        else:
            print("Not a file: ", embeddings_file)

    def _get_training_labels(self, data):
        self._load_label_encoder()
        labels = self.label_encoder.transform(
                np.array(data.conferenceseries).reshape(-1, 1))
        return labels

    def _load_label_encoder(self):
        # load label encoder
        encoder_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "gat",
                self.embedding_type, self.dataset, "label_encoder.pkl")
        if os.path.isfile(encoder_file):
            print("Loading label encoder...")
            with open(encoder_file, "rb") as f:
                self.label_encoder = pickle.load(f)
            print("Loaded.")
            return True
        return False

    def _plot_losses(self, train_losses, validation_losses):
        # Plot the training and validation losses
        ymax = max(max(train_losses), max(validation_losses))
        ymin = min(min(train_losses), min(validation_losses))
        plt.plot(train_losses, color='tab:blue')
        plt.plot(validation_losses, color='tab:orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend(["train", "validation"], loc=3)
        plt.ylim(ymin=ymin-0.5, ymax=ymax+0.5)
        plot_file = f'scibert_{self.arga_model_name}_losses.png'
        plt.savefig(os.path.join(self.model_dir, plot_file),
                    bbox_inches="tight")

    def _print_stats(self, train_losses, validation_losses, training_time):
        epochs = len(train_losses)
        time_per_epoch = training_time/epochs
        epoch_min_val = validation_losses.index(min(validation_losses))
        file = f'scibert_{self.arga_model_name}_stats.txt'
        stats_file = os.path.join(self.model_dir, file)
        with open(stats_file, "w") as f:
            self._print("Total number of epochs trained: {}, average time per epoch: {} minutes.\n".format(
                    epochs, round(time_per_epoch/60, 4)), f)
            self._print("Total time trained: {} minutes.\n".format(
                    round(training_time/60, 4)), f)
            self._print("Lowest validation loss at epoch {} = {}.\n".format(
                    epoch_min_val, validation_losses[epoch_min_val]), f)
            f.write("\n\n")
            for epoch in range(epochs):
                f.write('Epoch: %.f | Training: loss = %.5f | Val: loss = %.5f\n' %
                        (epoch, train_losses[epoch], validation_losses[epoch]))

    def _print(self, text, f):
        print(text)
        f.write(text)

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
        parser.add_argument("--n_latent",
                            type=int,
                            default=16,
                            help="Number of units in hidden layer of ARGA.")
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
        args = parser.parse_args()
        print("Starting...")
        from ffnn import FFNNModel
        model = FFNNModel(
                args.embedding_type, args.dataset, args.model_name,
                args.n_latent, args.learning_rate, args.weight_decay,
                args.dropout, args.dis_loss_para, args.reg_loss_para,
                args.epochs, args.gpu, args.ffnn_hidden_dim)
        model.train()
        print("Finished.")

    if __name__ == "__main__":
        main()
