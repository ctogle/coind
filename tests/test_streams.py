import collections
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from coind.data.tickers import TickerStream


def plot_stream(stream):
    t, X, Y = zip(*[(t.time, t.X, t.Y) for t in stream])
    X = np.array(X).transpose(0, 1)
    Y = np.array(Y).transpose(0, 1)
    print(X.shape, X[0].shape)
    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    for j, x in enumerate(X.transpose(1, 0)):
        ax.plot(t, x)
    for j, y in enumerate(Y.transpose(1, 0)):
        ax.plot(t, y)
    #ax.set_xlim((0, min(500, X.shape[0])))
    plt.show()


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
    plot_stream(periodic_stream)


@pytest.fixture
def ticker_stream(sample_class):
    def g(products, stream_window, *paths):
        stream = TickerStream(products, paths, stream_window)
        df = stream.tickers_df()
        for state in df.itertuples(index=False):
            yield sample_class(state.time, state.X, state.Y, False)
    products = './products.txt'
    with open(products, 'r') as f:
        products = [l.strip() for l in f.readlines() if not l.startswith('#')]
        products = [l for l in products if l]
    return g(products, 300, './stream.0.log')


def test_ticker_stream(ticker_stream):
    plot_stream(ticker_stream)
