# -*- coding: utf-8 -*-
import os
import sys
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import to_undirected
from torch_geometric.nn import ARGA, ARGVA, GCNConv
from arga_dataset import ARGADataset

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, model_name, dropout):
        self.model_name = model_name
        self.dropout = dropout
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        if self.model_name == "ARGA":
            self.conv2 = GCNConv(2 * out_channels, out_channels)
        elif self.model_name == "ARGVA":
            self.conv_mu = GCNConv(2 * out_channels, out_channels)
            self.conv_logvar = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.model_name == "ARGA":
            x = self.conv2(x, edge_index)
            return x
        elif self.model_name == "ARGVA":
            mu = self.conv_mu(x, edge_index)
            logvar = self.conv_logvar(x, edge_index)
            return mu, logvar


class Discriminator(nn.Module):
    def __init__(self, n_input):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(n_input, 4 * n_input),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4 * n_input, 2 * n_input),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(2 * n_input, 1), nn.Sigmoid())

    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        out = self.fc3(z)
        return out


class ARGAModel:
    def __init__(self, embedding_type, dataset, model_name,
                 graph_type="directed", mode="train", n_latent=16,
                 learning_rate=0.001, weight_decay=0, dropout=0,
                 dis_loss_para=1, reg_loss_para=1, epochs=200, gpu=None):

        # Set device
        if torch.cuda.is_available() and gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print("Using GPU device: {}.".format(str(gpu)))
            self.device = torch.device("cuda:" + str(gpu))
        else:
            self.device = "cpu"

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.model_name = model_name
        self.graph_type = graph_type
        self.n_latent = n_latent
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.dis_loss_para = dis_loss_para
        self.reg_loss_para = reg_loss_para
        self.epochs = epochs
        self.mode = mode

        # Load training data
        path_data_raw = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..", "..", "..",
                "data", "interim", "scibert_arga", self.embedding_type)
        self.data = ARGADataset(path_data_raw, self.embedding_type, dataset,
                                self.graph_type)[0]
        n_total_features = self.data.num_features

        # Initialize encoder and discriminator
        encoder = Encoder(n_total_features, self.n_latent, self.model_name,
                          self.dropout)
        discriminator = Discriminator(self.n_latent)
        if self.device is not "cpu":
            encoder.to(self.device)
            discriminator.to(self.device)

        # Choose and initialize model
        if self.model_name == "ARGA":
            self.model = ARGA(encoder=encoder, discriminator=discriminator,
                              decoder=None)
        else:
            self.model = ARGVA(encoder=encoder, discriminator=discriminator,
                               decoder=None)
        if self.device is not "cpu":
            self.model.to(self.device)

        if self.mode == "train":
            print("Preprocessing data...")
            self.data = self.split_edges(self.data)
            print("Data preprocessed.\n")

        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay)

        # Set model file
        self.model_dir = self._model_dir()
        self.model_file = f'{self.model_name}_{self.n_latent}_{self.learning_rate}_{self.weight_decay}_{self.dropout}.pt'

        print('Model: ' + self.model_name)
        print("\tEmbedding: {}, Dataset: {}, Graph type: {}".format(
                self.embedding_type, self.dataset, self.graph_type))
        print("\tHidden units: {}".format(self.n_latent))
        print("\tLearning rate: {}".format(self.learning_rate))
        print("\tWeight decay: {}".format(self.weight_decay))
        print("\tDropout: {}\n".format(self.dropout))
        print("\tEpochs: {}\n".format(self.epochs))

    def train(self):
        train_losses = []
        val_losses = []
        model_path = os.path.join(self.model_dir, self.model_file)

        print("Training model...\n")
        timer = Timer()
        timer.tic()

        x = self.data.x.to(self.device)
        train_pos_edge_index = self.data.train_pos_edge_index.to(self.device)

        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch + 1))
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model.encode(x, train_pos_edge_index)
            loss = self.model.recon_loss(z, train_pos_edge_index)
            if self.model_name == "ARGVA":
                loss = loss + (1 / self.data.num_nodes) * self.model.kl_loss()
            loss += self.dis_loss_para * self.model.discriminator_loss(z) + \
                self.reg_loss_para * self.model.reg_loss(z)
            loss.backward()
            self.optimizer.step()

            # Evaluate on validation data
            self.model.eval()
            with torch.no_grad():
                train_losses.append(loss.cpu().detach().numpy())

                # Compute validation statistics
                val_pos_edge_index = self.data.val_pos_edge_index.to(
                        self.device)
                val_neg_edge_index = self.data.val_neg_edge_index.to(
                        self.device)
                z = self.model.encode(x, train_pos_edge_index)
                val_loss = self.model.recon_loss(z, train_pos_edge_index)
                if self.model_name == "ARGVA":
                    val_loss += (1 / self.data.num_nodes) * self.model.kl_loss(
                            )
                val_loss += self.dis_loss_para * self.model.discriminator_loss(
                        z) + self.reg_loss_para * self.model.reg_loss(z)
                val_losses.append(val_loss.cpu().detach().numpy())
                if val_losses[-1] == min(val_losses):
                    print("\tSaving model...")
                    torch.save(self.model.state_dict(), model_path)
                    print("\tSaved.")
                print("\ttrain_loss=", "{:.5f}".format(loss),
                      "val_loss=", "{:.5f}".format(val_loss))

        print("Finished training.\n")
        training_time = timer.toc()
        self._plot_losses(train_losses, val_losses)
        self._print_stats(train_losses, val_losses, training_time)

    def test(self, test_data):
        if self.mode == "test":
            print("Splitting edges...")
            data = self._split_edges_test(test_data)
            print("Finished splitting edges.")
        else:
            data = test_data

        print("Loading model...")
        model_path = os.path.join(self.model_dir, self.model_file)
        try:
            self.model.load_state_dict(torch.load(model_path))
            print("Loaded.\n")
        except Exception as e:
            print("Could not load model from {} ({})".format(model_path, e))

        self.model.eval()

        print("Computing embeddings...")
        x = data.x.to(self.device)
        train_pos_edge_index = data.train_pos_edge_index.to(self.device)
        with torch.no_grad():
            z = self.model.encode(x, train_pos_edge_index)
            z = z.cpu().detach().numpy()
        print("Computed.\n")

        return z

    def _filter_labels(self, data, labels):
        mapping = torch.ones(data.size()).byte()
        for label in labels:
            mapping = mapping & (~ data.eq(label)).byte()
        return mapping

    # This method implementation is based on
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html#split_edges
    def split_edges(self, data):
        assert "batch" not in data

        row, col = data.edge_index
        data.edge_index = None

        # Return upper triangular portion
        mask = row < col
        row, col = row[mask], col[mask]

        n_train = len(np.where(data.train_mask == True)[0])
        n_val = len(np.where(data.val_mask == True)[0])

        train_labels_mask = range(n_train)
        val_labels_mask = range(n_train, n_train + n_val)

        # Positive edges
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        row_map = self._filter_labels(row, val_labels_mask)
        col_map = self._filter_labels(col, val_labels_mask)
        map_intersection = row_map * col_map
        r = row[map_intersection]
        c = col[map_intersection]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)

        row_map = self._filter_labels(row, train_labels_mask)
        col_map = self._filter_labels(col, train_labels_mask)
        map_intersection = row_map * col_map
        r = row[map_intersection]
        c = col[map_intersection]

        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        perm = random.sample(range(neg_row.size(0)),
                             min(n_val, neg_row.size(0)))

        perm = torch.tensor(perm)
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        neg_row_map = self._filter_labels(neg_row, val_labels_mask)
        neg_col_map = self._filter_labels(neg_col, val_labels_mask)
        map_intersection = neg_row_map * neg_col_map
        row = neg_row_map[map_intersection]
        col = neg_col_map[map_intersection]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def _split_edges_test(self, data):
        assert "batch" not in data

        row, col = data.edge_index
        data.edge_index = None

        # Return upper triangular portion
        mask = row < col
        if len(np.where(mask == True)[0]) == 0:
            mask = col < row
        row, col = row[mask], col[mask]

        n_train = len(np.where(data.train_mask == True)[0])
        n_val = len(np.where(data.val_mask == True)[0])
        n_test = len(np.where(data.test_mask == True)[0])

        train_labels_mask = range(n_train)
        val_labels_mask = range(n_train, n_train + n_val)
        test_labels_mask = range(n_train + n_val, n_train + n_val + n_test)

        # Positive edges
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        row_map = self._filter_labels(row, val_labels_mask)
        col_map = self._filter_labels(col, val_labels_mask)
        map_intersection = row_map * col_map
        r = row[map_intersection]
        c = col[map_intersection]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)

        row_map = self._filter_labels(row, test_labels_mask)
        col_map = self._filter_labels(col, test_labels_mask)
        map_intersection = row_map * col_map
        r = row[map_intersection]
        c = col[map_intersection]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        row_map = self._filter_labels(row, train_labels_mask)
        col_map = self._filter_labels(col, train_labels_mask)
        map_intersection = row_map * col_map
        r = row[map_intersection]
        c = col[map_intersection]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        perm = random.sample(range(neg_row.size(0)),
                             min(n_val+n_test, neg_row.size(0)))
        perm = torch.tensor(perm)
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        neg_row_map = self._filter_labels(neg_row, val_labels_mask)
        neg_col_map = self._filter_labels(neg_col, val_labels_mask)
        map_intersection = neg_row_map * neg_col_map
        row = neg_row_map[map_intersection]
        col = neg_col_map[map_intersection]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        neg_row_map = self._filter_labels(neg_row, test_labels_mask)
        neg_col_map = self._filter_labels(neg_col, test_labels_mask)
        map_intersection = neg_row_map * neg_col_map
        row = neg_row_map[map_intersection]
        col = neg_col_map[map_intersection]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def _embeddings_file(self):
        file = f'{self.dataset}_{self.graph_type}_{self.model_name}_embeddings_{self.n_latent}_{self.learning_rate}_{self.weight_decay}_{self.dropout}.pkl'
        path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "scibert_arga",
                self.embedding_type, file)
        return path

    def save_embeddings(self, embeddings):
        print("Saving embeddings to disk...")
        file_embeddings = self._embeddings_file()
        with open(file_embeddings, "wb") as f:
            pickle.dump(embeddings, f)
        print("Saved.")

    def load_embeddings(self):
        file_embeddings = self._embeddings_file()
        with open(file_embeddings, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings

    def _model_dir(self):
        model_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "processed", "scibert_arga",
                self.embedding_type, self.dataset, self.graph_type)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

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
        plt.savefig(os.path.join(self.model_dir, "losses.png"),
                    bbox_inches="tight")

    def _print_stats(self, train_losses, validation_losses, training_time):
        epochs = len(train_losses)
        time_per_epoch = training_time/epochs
        epoch_min_val = validation_losses.index(min(validation_losses))

        stats_file = os.path.join(self.model_dir, "stats.txt")
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
                description='Arguments for ARGA model.')
        parser.add_argument('embedding_type',
                            choices=["AVG_L", "AVG_2L", "AVG_SUM_L4",
                                     "AVG_SUM_ALL", "MAX_2L",
                                     "CONC_AVG_MAX_2L", "CONC_AVG_MAX_SUM_L4",
                                     "SUM_L", "SUM_2L", "temp"],
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
        parser.add_argument('--mode',
                            choices=["train", "test"],
                            default="train",
                            help="Whether to set the net to training mode.")
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

        print("Starting...\n")
        from arga import ARGAModel
        model = ARGAModel(args.embedding_type, args.dataset, args.model_name,
                          args.graph_type, args.mode, args.n_latent,
                          args.learning_rate, args.weight_decay, args.dropout,
                          args.dis_loss_para, args.reg_loss_para, args.epochs,
                          args.gpu)
        model.train()
        print("Finished.\n")

    if __name__ == "__main__":
        main()
