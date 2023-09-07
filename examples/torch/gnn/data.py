# code modified from https://github.com/Diego999/pyGAT

import dgl


def load_data(dataset="cora"):

    if dataset == "cora":
        dataset = dgl.data.CoraGraphDataset()
        adj = dataset[0].adj().to_dense()

        features = dataset[0].ndata["feat"]
        labels = dataset[0].ndata["label"]

        train_mask = dataset[0].ndata["train_mask"]
        val_mask = dataset[0].ndata["val_mask"]
        test_mask = dataset[0].ndata["test_mask"]

        idx_train = (train_mask == True).nonzero().flatten()
        idx_val = (val_mask == True).nonzero().flatten()
        idx_test = (test_mask == True).nonzero().flatten()

        return adj, features, labels, idx_train, idx_val, idx_test, train_mask

    elif dataset == "wiki-cs":
        dataset = dgl.data.WikiCSDataset()
        adj = dataset[0].adj().to_dense()

        features = dataset[0].ndata["feat"]
        labels = dataset[0].ndata["label"]

        train_mask = dataset[0].ndata["train_mask"][:, 0]
        val_mask = dataset[0].ndata["val_mask"][:, 0]
        test_mask = dataset[0].ndata["test_mask"]

        idx_train = (train_mask == True).nonzero().flatten()
        idx_val = (val_mask == True).nonzero().flatten()
        idx_test = (test_mask == True).nonzero().flatten()

        return adj, features, labels, idx_train, idx_val, idx_test, train_mask
