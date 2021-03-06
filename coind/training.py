import argparse
import matplotlib.pyplot as plt
import os
import collections
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

from .data import Dataset, DataLoader
#from .data.tickers import tickers_df, ticker_feature_count
from .data.tickers import TickerStream
from .model import Oracle
from .validation import validate


class Trainer:

    def __init__(self, df, window, stride, products,
                 savedir, epochs, batch_size, num_workers, lr):
        self.savedir = savedir
        self.epochs = epochs
        self.lr = lr
        edge = math.ceil(0.8 * len(df))
        train = Dataset(df[:edge], window, stride)
        val = Dataset(df[edge:], window, stride)
        self.dl_kws = dict(num_workers=num_workers,
                           batch_size=batch_size,
                           products=products)
        self.train_dl = DataLoader(train, resample=False, **self.dl_kws)
        self.val_dl = DataLoader(val, **self.dl_kws)


    def train(self, model):
        train_losses, val_losses, accuracies = [], [], []
        #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        val_metrics = validate(model, self.val_dl, self.dl_kws['products'])
        val_losses.append(np.mean(val_metrics['loss']))
        for e in range(self.epochs):
            train_metrics = self.train_epoch(e, model, self.train_dl, optimizer)
            train_losses.extend(train_metrics['loss'])
            val_metrics = validate(model, self.val_dl, self.dl_kws['products'])
            val_losses.append(np.mean(val_metrics['loss']))
            accuracies.append(val_metrics['mean_accuracy'])
            model_path = os.path.join(self.savedir, 'oracle.pt')
            torch.save(model, model_path)
        f, ax = plt.subplots(1, 1, figsize=(10, 4))
        for y in (train_losses, val_losses, accuracies):
            y = np.array(y) / max(y)
            x = np.linspace(0, self.epochs + 1, len(y))
            ax.plot(x, y)
        plt.savefig(os.path.join(self.savedir, 'loss_accuracy.png'))


    def train_epoch(self, epoch, model, dl, optimizer):
        model.train()
        metrics = collections.defaultdict(list)
        criterion = nn.NLLLoss()
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
                self.plot_grad_flow(ax, model.named_parameters())
                optimizer.step()
                metrics['loss'].append(loss.item())
                pbar.update(1)
                desc = f'loss: {loss.item():.6f}'
                pbar.set_description(desc)
        plt.savefig(os.path.join(self.savedir, f'grads.{epoch}.png'))
        return metrics


    @staticmethod
    def plot_grad_flow(ax, named_parameters):
        '''Usage: Plug this function in Trainer class after loss.backwards() as
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
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of samples per batch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for loading data')
    args = parser.parse_args()

    if args.products is None:
        #products = None
        products = streamable(*args.inputs, window=stream_window)
    else:
        with open(args.products, 'r') as f:
            products = [l.strip() for l in f.readlines() if not l.startswith('#')]
            products = [l for l in products if l]

    stream = TickerStream(products, args.inputs, args.stream_window)
    df = stream.tickers_df()
    #df = tickers_df(*args.inputs, products=products, window=args.stream_window)

    n_products = len(products)
    n_features = stream.ticker_feature_count * n_products
    n_classes = 2
    model = Oracle(n_features, args.d_hidden, args.n_layers, n_products, n_classes)

    trainer = Trainer(df, args.window, args.stride, products,
                      args.savedir, args.epochs, args.batch_size, args.num_workers,
                      args.lr)
    trainer.train(model)
