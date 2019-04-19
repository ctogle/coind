import collections
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from coind.model import Oracle
from coind.training import Trainer
from test_streams import periodic_stream, sample_class


def test_periodic_convergence(periodic_stream, sample_class):
    df = pd.DataFrame.from_records(periodic_stream, columns=sample_class._fields)
    window, stride, products = 24, 12, ['mu']
    n_features, d_hidden, n_layers, n_products, n_classes = 3, 200, 1, 1, 2
    model = Oracle(n_features, d_hidden, n_layers, n_products, n_classes)
    savedir, epochs, batch_size, num_workers, lr = './', 10, 32, 8, 0.01
    trainer = Trainer(df, window, stride, products,
                      savedir, epochs, batch_size, num_workers, lr)
    trainer.train(model)
