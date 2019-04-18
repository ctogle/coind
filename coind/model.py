import torch.nn as nn


class Oracle(nn.Module):

    def __init__(self, n_features, d_hidden, n_layers, n_products, n_classes):
        super().__init__()
        self.rnn = nn.LSTM(n_features, d_hidden, n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_hidden, n_products * n_classes, bias=False),
        )
        self.soft = nn.LogSoftmax(dim=-1)
        self.n_features = n_features
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_products = n_products
        self.n_classes = n_classes


    def forward(self, x):
        o, (h, c) = self.rnn(x)
        y = self.decoder(h[-1])
        y = y.view(self.n_products, x.shape[1], self.n_classes)
        return self.soft(y)
