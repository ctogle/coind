import argparse
import matplotlib.pyplot as plt
import os
import collections
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler

import coind.tickers as tickers
from .tickers import ticker_feature_keys
from .model import Oracle
from .validation import validate


class TickerDataset(Dataset):

    def __init__(self, df, products, window=6, stride=6):
        self.df = df
        self.products = products
        self.window = window
        self.stride = stride


    def __getitem__(self, index):
        snapshot = self.df[index:index + self.window + self.stride]
        frames = []
        prices = [[] for product in self.products]
        for i, snap in enumerate(snapshot.itertuples(index=False)):
            if i < self.window:
                fs = snap.features
                frames.append((fs - fs.mean()) / fs.std())
            if i >= self.window - 1:
                for j, product in enumerate(self.products):
                    prices[j].append(snap.prices[product])
        frames, prices = np.array(frames), np.array(prices)

        #bull = 1 * (prices[:, -1] > prices[:, 0])

        price_grads = np.gradient(prices, axis=1)
        bull = 1 * (price_grads.mean(axis=1) > 0)
        #bull = 1 * (price_grads.min(axis=1) > 0)

        assert len(frames) == self.window
        assert len(bull) == len(self.products)
        return frames, bull


    def __len__(self):
        return self.df.shape[0] - self.window - self.stride


    @staticmethod
    @torch.no_grad()
    def _collate(samples):
        frames, bulls = zip(*samples)
        batched_frames = torch.FloatTensor(frames).transpose(0, 1)
        batched_bulls = torch.LongTensor(bulls).transpose(0, 1)
        return batched_frames, batched_bulls


    @classmethod
    def from_log(cls, ticker_log, window=3, stride=3, stream_window=600):
        products = tickers.streamable(ticker_log, window=stream_window)
        df = tickers.df(ticker_log, products, window=stream_window)
        return cls(df, products, window, stride)


    @classmethod
    def splits(cls, *ticker_logs, window=3, stride=3, stream_window=600, products=None):
        if products is None:
            products = tickers.streamable(*ticker_logs, window=stream_window)
        df = tickers.df(*ticker_logs, products=products, window=stream_window)
        many = df.shape[0]
        assert many > 3 * (window + stride),\
               f'insufficient data... (only {many} samples)'
        tiny = int(0.02 * many)
        test_df = df.iloc[-tiny:]
        val_df = df.iloc[-tiny * 2:-tiny]
        index = np.random.RandomState(0).permutation(np.arange(many - 2 * tiny))
        train_df = df.iloc[index]
        print(train_df.shape, val_df.shape, test_df.shape)
        return (cls(train_df, products, window, stride),
                cls(val_df, products, window, stride),
                cls(test_df, products, window, stride))


class ContiguousSampler(Sampler):

    def __init__(self, dataset, resample=False):
        self.ids = []
        self.bins = [[[] for x in range(2)] for product in dataset.products]
        print(len(dataset), dataset.df.shape)
        edgecount = 0
        for x in np.arange(len(dataset)):
            snapshot = dataset.df[x:x + dataset.window + dataset.stride]
            if not snapshot.edge.values.any():
                self.ids.append(x)
                inputs, targets = dataset[x]
                for y, target in enumerate(targets):
                    self.bins[y][target].append(x)
            else:
                edgecount += 1
        self.n_samples = len(self.ids)
        self._iteration = -1
        print(f'edges: {edgecount}')
        for product, bins in zip(dataset.products, self.bins):
            print(f'Product: {product}')
            for cls in range(len(bins)):
                print(f'cls: {cls}, cnt: {len(bins[cls])}')
        self.resample = resample
        self.btc_index = dataset.products.index('BTC-USD')


    def __iter__(self):
        self._iteration += 1
        rng = np.random.RandomState(self._iteration)
        if self.resample:
            n_samples_per_class = 10000
            bins = self.bins[self.btc_index]
            index = []
            for target_bin in bins:
                chunk = rng.permutation(target_bin)
                chunk = chunk[:min(len(chunk), n_samples_per_class)]
                index.extend(chunk)
        else:
            index = self.ids
        return iter(rng.permutation(index))


    def __len__(self):
        return self.n_samples


def plot_grad_flow(ax, named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    ax.bar(np.arange(len(max_grads)) + 0.5, max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)) + 0.5, ave_grads, alpha=0.1, lw=1, color="b")
    ax.set_xticks(np.arange(len(ave_grads)) + 0.5)
    ax.set_xticklabels(layers, rotation="vertical")
    ax.set_xlim((     0, len(ave_grads)))
    ax.set_ylim((-0.001, max(max_grads)))
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")


def train_epoch(epoch, savedir, model, dl, lr=0.1, momentum=0.9):
    model.train()
    metrics = collections.defaultdict(list)
    criterion = nn.NLLLoss()
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    with tqdm(total=len(dl)) as pbar:
        for j, (batch, targets) in enumerate(dl):
            metrics['positives'].append(targets.sum().item())
            optimizer.zero_grad()
            hypothesis = model(batch)
            losses = []
            for i, (prediction, target) in enumerate(zip(hypothesis, targets)):
                losses.append(criterion(prediction, target))
            loss = torch.sum(torch.stack(losses))
            loss.backward()
            plot_grad_flow(ax, model.named_parameters())
            optimizer.step()
            metrics['loss'].append(loss.item())
            pbar.update(1)
            desc = f'loss: {loss.item():.6f}'
            pbar.set_description(desc)
    plt.savefig(os.path.join(savedir, f'grads.{epoch}.png'))
    return metrics


class DeviceWrap:

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device


    def __iter__(self):
        for batch, targets in self.dl:
            yield batch.to(self.device), targets.to(self.device)


    def __len__(self):
        return len(self.dl)


def train(epochs, savedir, ticker_logs,
          window=3, stride=3, stream_window=300, products=None,
          d_hidden=20, n_layers=1, batch_size=4, num_workers=4):
    train_set, val_set, test_set = TickerDataset.splits(*ticker_logs,
                                                        window=window,
                                                        stride=stride,
                                                        stream_window=stream_window,
                                                        products=products)
    train_sampler = ContiguousSampler(train_set, resample=True)
    val_sampler = ContiguousSampler(val_set)
    test_sampler = ContiguousSampler(test_set)
    dl_kws = dict(num_workers=num_workers,
                  batch_size=batch_size,
                  collate_fn=TickerDataset._collate)
    train_dl = DataLoader(train_set, sampler=train_sampler, **dl_kws)
    val_dl = DataLoader(val_set, sampler=val_sampler, **dl_kws)
    test_dl = DataLoader(test_set, sampler=test_sampler, **dl_kws)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dl = DeviceWrap(train_dl, device)
    val_dl = DeviceWrap(val_dl, device)
    test_dl = DeviceWrap(test_dl, device)
    n_products = len(train_set.products)
    n_features = len(ticker_feature_keys) * n_products
    n_classes = 2
    model = Oracle(n_features, d_hidden, n_layers, n_products, n_classes)
    model.to(device)
    model.products = train_set.products
    model_path = os.path.join(savedir, 'oracle.pt')
    train_losses = []
    val_losses = []
    accuracies = []
    val_metrics = validate(model, val_dl)
    val_losses.append(np.mean(val_metrics['loss']))
    for e in range(epochs):
        train_metrics = train_epoch(e, savedir, model, train_dl)
        train_losses.extend(train_metrics['loss'])
        val_metrics = validate(model, val_dl)
        val_losses.append(np.mean(val_metrics['loss']))
        accuracies.append(val_metrics['mean_accuracy'])
        torch.save(model, model_path)
    return model, train_losses, val_losses, accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train predictive models')
    parser.add_argument('--savedir', type=str, default='./',
                        help='Path to store training related files')
    parser.add_argument('--inputs', nargs='+', default=[],
                        help='One or more ticker logs to use for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--products', default=None,
                        help='Path to list of targeted products')
    parser.add_argument('--window', type=int, default=9,
                        help='Number of input timesteps for predictive model')
    parser.add_argument('--stride', type=int, default=9,
                        help='Number of timesteps being predicted')
    parser.add_argument('--stream_window', type=int, default=900,
                        help='Number of seconds per averaged timestep')
    parser.add_argument('--d_hidden', type=int, default=100,
                        help='LSTM hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of LSTMS to stack')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of samples per batch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for loading data')
    args = parser.parse_args()

    if args.products is None:
        products = None
    else:
        with open(args.products, 'r') as f:
            products = [l.strip() for l in f.readlines() if not l.startswith('#')]
            products = [l for l in products if l]

    model, train_losses, val_losses, accuracies =\
        train(args.epochs, args.savedir, args.inputs,
              window=args.window, stride=args.stride,
              stream_window=args.stream_window,
              products=products, d_hidden=args.d_hidden, n_layers=args.n_layers,
              batch_size=args.batch_size, num_workers=args.num_workers)

    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    for y in (train_losses, val_losses, accuracies):
        y = np.array(y) / max(y)
        x = np.linspace(0, args.epochs + 1, len(y))
        ax.plot(x, y)
    plt.savefig(os.path.join(args.savedir, 'loss_accuracy.png'))
