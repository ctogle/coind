from datetime import datetime, timedelta
from multiprocessing import Queue
import collections
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
import coind.tickers as tickers


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
            else:
                for j, product in enumerate(self.products):
                    prices[j].append(snap.prices[product])
        frames, prices = np.array(frames), np.array(prices)
        price_grads = np.gradient(prices, axis=1)
        bull = 1 * (price_grads.min(axis=1) > 0)
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
    def splits(cls, *ticker_logs, window=3, stride=3, stream_window=600):
        products = tickers.streamable(*ticker_logs, window=stream_window)
        df = tickers.df(*ticker_logs, products=products, window=stream_window)
        many = df.shape[0]
        assert many > 3 * (window + stride),\
               f'insufficient data... (only {many} samples)'
        tiny = int(0.2 * many)
        test_df = df.iloc[-tiny:]
        val_df = df.iloc[-tiny * 2:-tiny]
        index = np.random.RandomState(0).permutation(np.arange(many - 2 * tiny))
        train_df = df.iloc[index]
        return (cls(train_df, products, window, stride),
                cls(val_df, products, window, stride),
                cls(test_df, products, window, stride))
