import argparse
import matplotlib.pyplot as plt
import os
import collections
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import dataloaders
from .model import Oracle
from .validation import validate


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


def train_epoch(epoch, savedir, model, dl, optimizer):
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
            plot_grad_flow(ax, model.named_parameters())
            optimizer.step()
            metrics['loss'].append(loss.item())
            pbar.update(1)
            desc = f'loss: {loss.item():.6f}'
            pbar.set_description(desc)
    plt.savefig(os.path.join(savedir, f'grads.{epoch}.png'))
    return metrics


def train(epochs, savedir, ticker_logs,
          window=3, stride=3, stream_window=300, products=None,
          d_hidden=20, n_layers=1, batch_size=4, num_workers=4):

    feature_config = dict(window=window, stride=stride,
        stream_window=stream_window, products=products)

    products, n_features_per_product, train_dl, val_dl, test_dl =\
        dataloaders(*ticker_logs, **feature_config)

    n_products = len(products)
    n_features = n_features_per_product * n_products
    n_classes = 2
    model = Oracle(n_features, d_hidden, n_layers, n_products, n_classes)
    model.to(train_dl.device)
    model.products = products
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
