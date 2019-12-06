# # Fraud Detection using Graph Convolutional Networks

import argparse
import os
import random
import time

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# set random seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
dgl.random.seed(seed)


class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        loss = loss_fcn(logits, labels)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        p, r, f, _ = precision_recall_fscore_support(labels, indices)
        if not args.nowandb:
            wandb.log(
                {
                    "confusion_matrix": wandb.Table(
                        data=list(
                            np.hstack(
                                (
                                    [["true_neg"], ["true_pos"]],
                                    confusion_matrix(labels, indices),
                                )
                            )
                        ),
                        columns=["", "pred_neg", "pred_pos"],
                    )
                }, commit=False
            )
        return loss, correct.item() * 1.0 / len(labels), p[1], r[1], f[1]


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="Provide index of GPU to use."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay of L2 regularization",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of data to use for training",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate for training",
    )
    parser.add_argument(
        "--nhidden", type=int, default=100, help="Number of hidden units per layer",
    )
    parser.add_argument(
        "--nlayer", type=int, default=2, help="Number of layers",
    )
    parser.add_argument(
        "--onlylocal",
        default=False,
        action="store_true",
        help="Use only local features for training",
    )
    parser.add_argument(
        "--posweight",
        type=float,
        default=0.7,
        help="Loss weight given to positive samples",
    )
    parser.add_argument(
        "--noselfloop",
        default=False,
        action="store_true",
        help="If true, then do not add selfloop to nodes",
    )
    parser.add_argument(
        "--save",
        default=False,
        dest="save",
        action="store_true",
        help="If true, then final weights will be saved",
    )
    parser.add_argument(
        "--nowandb",
        default=False,
        action="store_true",
        help="If true, do not use wandb for this run",
    )
    args = parser.parse_args()
    return args


# %%
if __name__ == "__main__":

    args = _parse_args()

    if not args.nowandb:
        wandb.init(config={"framework": "torch"}, project="bitcoin-transaction-graph3")
        wandb.config.update(args)

    # load data
    path = os.path.dirname(os.path.realpath(__file__))
    df_edges = pd.read_csv(path + "/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
    df_classes = pd.read_csv(
        path + "/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
    )
    df_features = pd.read_csv(
        path + "/elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None
    )

    # rename the classes to ints that can be handled by pytorch as labels
    df_classes["label"] = (
        df_classes["class"].replace({"unknown": -1, "2": 0}).astype(int)
    )

    # create networkx graph from the pandas dataframes
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(
        zip(df_classes["txId"], [{"label": v} for v in df_classes["label"]])
    )
    g_nx.add_edges_from(zip(df_edges["txId1"], df_edges["txId2"]))

    # create DGL graph
    g = dgl.DGLGraph()
    g.from_networkx(g_nx)
    g.ndata["label"] = torch.tensor(
        df_classes.set_index("txId").loc[sorted(g_nx.nodes()), "label"].values
    )
    g.ndata["features_matrix"] = torch.tensor(
        df_features.set_index(0).loc[sorted(g_nx.nodes()), :].values
    )

    # add self loop
    if not args.noselfloop:
        g.add_edges(g.nodes(), g.nodes())

    print(g)

    # get features and labels for training
    if args.onlylocal:
        features = g.ndata["features_matrix"][:, 0:94].float()
    else:
        features = g.ndata["features_matrix"].float()
    labels = g.ndata["label"].long()  # cross entropy loss
    in_feats = features.shape[1]
    n_classes = 2  # df_classes['label'].nunique()
    n_edges = g.number_of_edges()

    dataset_size = df_classes["label"].notna().sum()
    train_ratio = args.train_ratio
    train_time_steps = round(len(np.unique(features[:, 0])) * train_ratio)
    train_indices = (
        ((features[:, 0] <= train_time_steps) & (labels != -1)).nonzero().view(-1)
    )
    val_indices = (
        ((features[:, 0] > train_time_steps) & (labels != -1)).nonzero().view(-1)
    )
    # test_indices = (
    #     ((features[:, 0] > train_time_steps) & (labels != -1)).nonzero().view(-1)
    # )

    print(
        f"""----Data statistics------
        #Edges {n_edges}
        #Classes {n_classes}
        #Train samples {len(train_indices)}
        #Val samples {len(val_indices)}
        #Input features {in_feats}"""
    )

    n_hidden = args.nhidden
    n_layers = args.nlayer
    dropout = args.dropout

    # create GCN model
    model = GCN(g, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout)
    if not args.nowandb:
        wandb.watch(model)

    loss_fcn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1 - args.posweight, args.posweight])
    )

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    dur = []
    losses = {"train": [], "val": []}
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward pass
        logits = model(features)
        loss = loss_fcn(logits[train_indices], labels[train_indices])
        losses["train"].append(loss)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # duration
        if epoch >= 3:
            dur.append(time.time() - t0)
        # evaluate on validation set
        val_loss, acc, precision, recall, f1_score = evaluate(
            model, features, labels, val_indices
        )
        losses["val"].append(val_loss)
        # log to wandb
        if not args.nowandb:
            wandb.log(
                {
                    "train_loss": loss,
                    "val_loss": val_loss,
                    "val_accuracy": acc,
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1_score": f1_score,
                }
            )
        print(
            f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.2f} | Loss {loss.item():.4f} "
            f"| Precision {precision:.4f} | Recall {recall:.4f} | Acc {acc:.4f} "
        )

    # %%
    # print()
    # _, acc, precision, recall, f1_score = evaluate(
    #     model, features, labels, val_indices
    # )
    # print(
    #     f"Val Accuracy {acc:.4f} | Precision {precision:.4f} | Recall {recall:.4f} | "
    #     f"F1 score {f1_score:.4f}"
    # )
