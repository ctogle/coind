import collections
import numpy as np
import pandas as pd
import math
import os
import pytest
import matplotlib.pyplot as plt
from coind.data import dataloaders
from coind.data.tickers import TickerDataset, TickerDataLoader
from coind.model import Oracle
from coind.training import train_epoch
from coind.validation import validate

import torch.optim as optim


@pytest.fixture
def sample_class():
    return collections.namedtuple('Sample', ('time', 'X', 'Y', 'edge'))


@pytest.fixture
def periodic_stream(sample_class):
    def g(N, SNR):
        T = np.arange(N)
        R = np.random.randn(N) / SNR
        A = (0.5 + 0.2 * np.sin(T / 5)) * np.sin(T / 30) + np.random.randn(N) / SNR
        B = (0.2 + 0.5 * np.sin(T / 100)) * np.sin(T / 40) + np.random.randn(N) / SNR
        C = (-0.6 + 0.2 * np.sin(T / 60)) * np.sin(30 + T / 60) + np.random.randn(N) / SNR
        D = (0.5 + 0.2 * np.sin(T / 5)) * A + 0.1 * (B + C)
        for t, a, b, c, d in zip(T, A, B, C, D):
            X = np.array([a, b, c])
            Y = np.array([d])
            yield sample_class(t, X, Y, False)
    return g(8000, 50)


def test_periodic_stream(periodic_stream):
    t, X, Y = zip(*[(t.time, t.X, t.Y) for t in periodic_stream])
    X = np.array(X).transpose(0, 1)
    Y = np.array(Y).transpose(0, 1)
    print(X.shape, X[0].shape)
    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    for j, x in enumerate(X.transpose(1, 0)):
        ax.plot(t, x)
    for j, y in enumerate(Y.transpose(1, 0)):
        ax.plot(t, y)
    ax.set_xlim((0, min(500, X.shape[0])))
    plt.show()


def test_periodic(periodic_stream, sample_class):
    df = pd.DataFrame.from_records(periodic_stream, columns=sample_class._fields)
    edge = math.ceil(0.8 * len(df))
    train_df = df[:edge]
    val_df = df[edge:]

    window, stride = 24, 12
    train_set = TickerDataset(train_df, window, stride)
    val_set = TickerDataset(val_df, window, stride)

    products = ['z']
    dl_kws = dict(num_workers=8, batch_size=32, products=products)
    train_dl = TickerDataLoader(train_set, resample=False, **dl_kws)
    val_dl = TickerDataLoader(val_set, **dl_kws)

    n_features = 3
    d_hidden = 200
    n_layers = 1
    n_products = 1
    n_classes = 2
    model = Oracle(n_features, d_hidden, n_layers, n_products, n_classes)

    savedir, epochs, lr = './', 10, 0.01
    trainer = Trainer(savedir, epochs, lr)
    trainer.train(model, train_dl, val_dl, products)
    return


class Trainer:

    #def __init__(self, savedir, batch_size, num_workers, lr):
    def __init__(self, datastream, savedir, epochs, lr):
        self.savedir = savedir
        self.epochs = epochs
        #self.batch_size = batch_size
        #self.num_workers = num_workers
        self.lr = lr

        df = pd.DataFrame.from_records(datastream, columns=sample_class._fields)
        edge = math.ceil(0.8 * len(df))
        train_df = df[:edge]
        val_df = df[edge:]

        window, stride = 24, 12
        train_set = TickerDataset(train_df, window, stride)
        val_set = TickerDataset(val_df, window, stride)

        products = ['z']
        dl_kws = dict(num_workers=8, batch_size=32, products=products)
        train_dl = TickerDataLoader(train_set, resample=False, **dl_kws)
        val_dl = TickerDataLoader(val_set, **dl_kws)

    def train(self, model, train_dl, val_dl, products):
        train_losses, val_losses, accuracies = [], [], []
        #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        val_metrics = validate(model, val_dl, products)
        val_losses.append(np.mean(val_metrics['loss']))
        for e in range(self.epochs):
            train_metrics = train_epoch(e, self.savedir, model, train_dl, optimizer)
            train_losses.extend(train_metrics['loss'])
            val_metrics = validate(model, val_dl, products)
            val_losses.append(np.mean(val_metrics['loss']))
            accuracies.append(val_metrics['mean_accuracy'])
            #torch.save(model, model_path)

        f, ax = plt.subplots(1, 1, figsize=(10, 4))
        for y in (train_losses, val_losses, accuracies):
            y = np.array(y) / max(y)
            x = np.linspace(0, self.epochs + 1, len(y))
            ax.plot(x, y)
        plt.savefig(os.path.join(self.savedir, 'loss_accuracy.png'))


