# code modified from https://github.com/Diego999/pyGAT

import argparse
import logging
import os
import sys
import random
import time

import rich
import torch
import torch.nn.functional as F

from easydist import easydist_setup, mdconfig
from easydist.torch.experimental.api import easydist_compile

sys.path.append(os.path.abspath(__file__))
from gat import GAT
from data import load_data

random.seed(42)
torch.manual_seed(42)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--nb_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--dataset', type=str, choices=["wiki-cs", "cora"], default="wiki-cs")
    parser.add_argument('--epochs', type=int, default=800)

    args = parser.parse_args()
    rich.print(f"Training config: {args}")

    return args


def main():

    # setting up easydist and torch.distributed
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda")

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    args = get_args()

    adj, features, labels, idx_train, idx_val, _, train_mask = load_data(args.dataset)

    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha).cuda()

    adj, features, labels, idx_train = adj.cuda(), features.cuda(), labels.cuda(), idx_train.cuda()
    train_mask = train_mask.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 foreach=True,
                                 capturable=True)

    loss_scale = len(idx_train) / len(labels)

    @easydist_compile()
    def train_step(model, optimizer, adj, features, labels, mask):
        output = model(features, adj)
        loss_train = F.nll_loss(output * mask[:, None], labels * mask) / loss_scale
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        return output, loss_train

    for epoch in range(args.epochs):

        start_t = time.perf_counter()

        output, loss_train = train_step(model, optimizer, adj, features, labels, train_mask)

        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        rich.print('epoch={:04d} | '.format(epoch + 1),
                   'loss_train={:.6f}'.format(loss_train.data.item()),
                   'acc_train={:.6f}'.format(acc_train.data.item()),
                   'loss_val={:.6f}'.format(loss_val.data.item()),
                   'acc_val={:.6f}'.format(acc_val.data.item()),
                   'time={:.3f}'.format(time.perf_counter() - start_t))


if __name__ == "__main__":
    main()
