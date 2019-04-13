import torch.nn as nn


class Oracle(nn.Module):


    #def _hidden(self, x):
    #    self.n_layers, x.shape[1], self.d_hidden


    def __init__(self, n_features, d_hidden, n_layers, n_coins):
        super().__init__()
        self.rnn = nn.LSTM(n_features, d_hidden, n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_hidden, n_coins * 2),
            #nn.Softshrink(0.1),
        )
        self.soft = nn.LogSoftmax(dim=-1)
        self.n_features = n_features
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_coins = n_coins


    def forward(self, x):
        o, (h, c) = self.rnn(x)
        y = self.decoder(h[-1])
        y = y.view(self.n_coins, x.shape[1], 2)
        return self.soft(y)
