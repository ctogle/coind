import matplotlib.pyplot as plt
import os
import collections
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .tickers import ticker_feature_keys
from .data import TickerDataset, DataLoader
from .model import Oracle
from .validate import validate


def plot_grad_flow(named_parameters):
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
    plt.bar(np.arange(len(max_grads)) + 0.5, max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)) + 0.5, ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(np.arange(len(ave_grads)) + 0.5, layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=max(max_grads)) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")


def train_epoch(model, dl, lr=0.01, momentum=0.9):
    metrics = collections.defaultdict(list)
    criterion = nn.NLLLoss()
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
            plot_grad_flow(model.named_parameters())
            optimizer.step()
            metrics['loss'].append(loss.item())
            pbar.update(1)
            pbar.set_description('loss: {}'.format(loss.item()))
    return metrics


def train(epochs, save_dir, ticker_logs,
          window=6, stride=3, stream_window=600,
          d_hidden=20, n_layers=1, batch_size=4, num_workers=4):
    train_set, val_set, test_set = TickerDataset.splits(*ticker_logs,
                                                        window=window,
                                                        stride=stride,
                                                        stream_window=stream_window)
    dl_kws = dict(num_workers=num_workers,
                  batch_size=batch_size,
                  collate_fn=TickerDataset._collate)
    train_dl = DataLoader(train_set, **dl_kws)
    val_dl = DataLoader(val_set, **dl_kws)
    test_dl = DataLoader(test_set, **dl_kws)
    n_products = len(train_set.products)
    n_features = len(ticker_feature_keys) * n_products
    model = Oracle(n_features, d_hidden, n_layers, n_products)
    model.products = train_set.products
    train_losses = []
    val_losses = []
    accuracies = []
    model_path = os.path.join(save_dir, 'oracle.pt')
    os.makedirs(save_dir, exist_ok=True)
    val_metrics = validate(model, val_dl)
    val_losses.append(np.mean(val_metrics['loss']))
    for e in range(epochs):
        train_metrics = train_epoch(model, train_dl)
        train_losses.extend(train_metrics['loss'])
        val_metrics = validate(model, val_dl)
        val_losses.append(np.mean(val_metrics['loss']))
        accuracies.append(np.mean(val_metrics['accuracy']))
        torch.save(model, model_path)
    return model, train_losses, val_losses, accuracies


