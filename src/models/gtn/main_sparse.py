import os
import torch
import pickle
import argparse
import numpy as np
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from torch_geometric.utils import dense_to_sparse, f1_score, accuracy
from torch_geometric.data import Data
from model_sparse import GTN
from utils import f1_score

# DISCLAIMER:
# This code file is derived from
# https://github.com/seongjunyun/Graph_Transformer_Networks


class Model:

    def __init__(self, embedding_type, dataset, epochs=40, node_dim=64,
                 num_channels=2, learning_rate=0.005, weight_decay=0.001,
                 num_layers=3, norm=True, adaptive_lr=False, gpu=0):

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.epochs = epochs
        self.node_dim = node_dim
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.norm = norm
        self.adaptive_lr = adaptive_lr

        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print("Using GPU device: {}.\n".format(str(gpu)))

        print("Embedding type: {}".format(self.embedding_type))
        print("Dataset: {}".format(dataset))
        print("----- Hyperparameters -----")
        print("\tEpochs: {}".format(self.epochs))
        print("\tNode dimension: {}".format(self.node_dim))
        print("\tNumber of channels: {}".format(self.num_channels))
        print("\tNumber of layers: {}".format(self.num_layers))
        print("\tLearning rate: {}".format(self.learning_rate))
        print("\tWeight decay: {}".format(self.weight_decay))
        print("\tNormalization: {}".format(self.norm))
        print("\tAdaptive learning rate: {}\n".format(self.adaptive_lr))

    def train(self):
        print("Loading data...")
        node_features, edges, labels = self._load_data()
        print("Loaded.\n")

        print("Processing data...")
        num_nodes = edges[0].shape[0]
        A = []

        for i, edge in enumerate(edges):
            edge_tmp = torch.from_numpy(
                    np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(
                            torch.cuda.LongTensor)
            value_tmp = torch.ones(edge_tmp.shape[1]).type(
                    torch.cuda.FloatTensor)
            A.append((edge_tmp, value_tmp))
        edge_tmp = torch.stack(
                (torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(
                        torch.cuda.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
        A.append((edge_tmp, value_tmp))

        node_features = torch.from_numpy(node_features).type(
                torch.cuda.FloatTensor)
        train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(
                torch.cuda.LongTensor)
        train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(
                torch.cuda.LongTensor)
        valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(
                torch.cuda.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(
                torch.cuda.LongTensor)

        num_classes = torch.max(train_target).item()+1
        print("\tTraining nodes: {}".format(train_node.shape))
        print("\tValidation nodes: {}".format(valid_node.shape))
        print("Processed.\n")

        final_f1 = 0
        model_file = self._get_model_file()

        best_val_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        model = GTN(num_edge=len(A),
                    num_channels=self.num_channels,
                    w_in=node_features.shape[1],
                    w_out=self.node_dim,
                    num_class=num_classes,
                    num_nodes=node_features.shape[0],
                    num_layers=self.num_layers)
        model.cuda()
        if self.adaptive_lr is False:
            optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                    [{'params': model.gcn.parameters()},
                     {'params': model.linear1.parameters()},
                     {'params': model.linear2.parameters()},
                     {"params": model.layers.parameters(), "lr": 0.5}],
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay)
        loss = nn.CrossEntropyLoss()
        Ws = []

        print("Training model...")
        for i in range(self.epochs):
            print('Epoch: ', i+1)
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            model.train()
            model.zero_grad()
            loss, y_train, _ = model(A, node_features, train_node,
                                     train_target)
            loss.backward()
            optimizer.step()
            train_f1 = torch.mean(f1_score(
                    torch.argmax(y_train, dim=1), train_target, num_classes=3)
                    ).cpu().numpy()
            print('Train - Loss: {}, Macro_F1: {}'.format(
                    loss.detach().cpu().numpy(), train_f1))
            model.eval()

            # Valid
            with torch.no_grad():
                val_loss, y_valid, _ = model.forward(A, node_features,
                                                     valid_node, valid_target)
                val_f1 = torch.mean(f1_score(
                        torch.argmax(y_valid, dim=1),
                        valid_target, num_classes=3)).cpu().numpy()
                print('Valid - Loss: {}, Macro_F1: {}'.format(
                        val_loss.detach().cpu().numpy(), val_f1))

                if val_f1 > best_val_f1:
                    best_val_loss = val_loss.detach().cpu().numpy()
                    best_train_loss = loss.detach().cpu().numpy()
                    best_train_f1 = train_f1
                    best_val_f1 = val_f1
                    print("Saving model at epoch {} (train loss: {}, " +
                          "validation loss: {})".format(
                                  i+1, best_train_loss, best_val_loss))
                    torch.save(model.state_dict(), model_file)
                    print("Saved.\n")
            torch.cuda.empty_cache()

        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(
                best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(
                best_val_loss, best_val_f1))

    def inference(self):
        print("Loading data...")
        node_features, edges, labels = self._load_data()
        print("Loaded.\n")

        print("Processing data...")
        num_nodes = edges[0].shape[0]
        A = []

        for i, edge in enumerate(edges):
            edge_tmp = torch.from_numpy(
                    np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(
                            torch.cuda.LongTensor)
            value_tmp = torch.ones(edge_tmp.shape[1]).type(
                    torch.cuda.FloatTensor)
            A.append((edge_tmp, value_tmp))
        edge_tmp = torch.stack(
                (torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(
                        torch.cuda.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
        A.append((edge_tmp, value_tmp))

        node_features = torch.from_numpy(node_features).type(
                torch.cuda.FloatTensor)
        train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(
                torch.cuda.LongTensor)
        test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(
                torch.cuda.LongTensor)
        test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(
                torch.cuda.LongTensor)
        num_classes = torch.max(train_target).item() + 1
        print("Processed.\n")

        model = GTN(num_edge=len(A),
                    num_channels=self.num_channels,
                    w_in=node_features.shape[1],
                    w_out=self.node_dim,
                    num_class=num_classes,
                    num_nodes=node_features.shape[0],
                    num_layers=self.num_layers)
        model.cuda()

        print("Loading saved model...")
        model_file = self._get_model_file()
        if os.path.isfile(model_file):
            try:
                model.load_state_dict(torch.load(model_file))
                model.eval()
                print("Loaded.\n")
            except Exception as e:
                raise ValueError("Could not load saved model: {}".format(e))
        else:
            raise FileNotFoundError("Model file does not exist.")

        print("Inference...")
        with torch.no_grad():
            test_loss, y_test, W = model.forward(A, node_features, test_node,
                                                 test_target)
            test_f1 = torch.mean(f1_score(
                    torch.argmax(y_test, dim=1), test_target,
                    num_classes=3)).cpu().numpy()
            test_acc = accuracy(torch.argmax(y_test, dim=1), test_target)
            print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(
                    test_loss.detach().cpu().numpy(), test_f1, test_acc))
            print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(
                    test_loss.detach().cpu().numpy(), test_f1, test_acc))
        print("Finished.")
        print(y_test.cpu().numpy(), y_test.cpu().numpy().shape)
        return y_test.cpu().numpy()

    def _get_model_file(self):
        save_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "processed", "gtn",
                self.embedding_type, self.dataset)
        if self.adaptive_lr:
            adapt_lr = "adaptive_lr"
        else:
            adapt_lr = "fixed_lr"
        save_dir += "/{node_dim:s}_{num_channels:s}_{num_layers:s}_{lr:0.5f}_{wd:0.5f}_{alr:s}_sparse".format(
                node_dim=str(self.node_dim),
                num_channels=str(self.num_channels),
                num_layers=str(self.num_layers),
                lr=self.learning_rate,
                wd=self.weight_decay,
                alr=adapt_lr)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path_persistent_model = os.path.join(save_dir, "model.pth")
        return path_persistent_model

    def _load_data(self):
        path_persistent_raw = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "gtn",
                self.embedding_type, self.dataset)
        node_features_file = os.path.join(
                path_persistent_raw, "node_features.pkl")
        edges_file = os.path.join(
                path_persistent_raw, "edges.pkl")
        labels_file = os.path.join(
                path_persistent_raw, "labels.pkl")
        if os.path.isfile(node_features_file):
            with open(node_features_file, "rb") as f:
                node_features = pickle.load(f)
        else:
            raise FileNotFoundError(
                    "The node features file could not be found.")

        if os.path.isfile(edges_file):
            with open(edges_file, "rb") as f:
                edges = pickle.load(f)
        else:
            raise FileNotFoundError(
                    "The edges file could not be found.")

        if os.path.isfile(labels_file):
            with open(labels_file, "rb") as f:
                labels = pickle.load(f)
        else:
            raise FileNotFoundError(
                    "The labels file could not be found.")

        return node_features, edges, labels

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for GTN model.')
        parser.add_argument('embedding_type',
                            choices=["AVG_L", "AVG_2L", "AVG_SUM_L4",
                                     "AVG_SUM_ALL", "MAX_2L",
                                     "CONC_AVG_MAX_2L", "CONC_AVG_MAX_SUM_L4",
                                     "SUM_L", "SUM_2L"
                                     ],
                            help="Type of embedding.")
        parser.add_argument('dataset',
                            choices=["AP", "APT"],
                            help='Name of the object file that stores the '
                            + 'training data.')
        parser.add_argument('--epochs',
                            type=int,
                            default=40,
                            help='Training Epochs')
        parser.add_argument('--node_dim',
                            type=int,
                            default=64,
                            help='Node dimension')
        parser.add_argument('--num_channels',
                            type=int,
                            default=2,
                            help='Number of channels')
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.005,
                            help='Learning rate')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=0.001,
                            help='L2 reg')
        parser.add_argument('--num_layers',
                            type=int,
                            default=3,
                            help='Number of layers')
        parser.add_argument('--norm',
                            default=True,
                            action='store_false',
                            help='Whether to use normalization.')
        parser.add_argument('--adaptive_lr',
                            default=False,
                            action='store_true',
                            help='Whether to use an adaptive learning rate.')
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        args = parser.parse_args()

        print("Starting...")
        from main_sparse import Model
        model = Model(args.embedding_type, args.dataset, args.epochs,
                      args.node_dim,  args.num_channels, args.learning_rate,
                      args.weight_decay, args.num_layers, args.norm,
                      args.adaptive_lr, args.gpu)
        model.train()
        print("Finished.")

    if __name__ == '__main__':
        main()
