import argparse
from datetime import datetime, timedelta
from multiprocessing import Queue
#from recordclass import recordclass
import collections
import pandas as pd
import numpy as np
import json
import cbpro
import time
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class TickerStream:

    def __init__(self, products, source=None, window=600):
        self.products = products
        if source is None:
            self.stream = self.stream_tickers_live()
        elif isinstance(source, str):
            self.stream = self.stream_tickers(source)
        else:
            stream = (self.stream_tickers(path) for path in source)
            self.stream = (i for it in stream for i in it)
        self.window = window


    def stream_tickers_live(self, doraise=False):
        wsClient = TickerClient(products=self.products)
        wsClient.start()
        print(wsClient.url, wsClient.products)
        try:
            while True:
                if wsClient.queue.empty():
                    time.sleep(1)
                else:
                    yield wsClient.queue.get()
        except KeyboardInterrupt:
            wsClient.close()
            if doraise:
                raise KeyboardInterrupt
        #if wsClient.error:


    def stream_tickers(self, path):
        try:
            with open(path, 'r') as f:
                for l in f:
                    msg = json.loads(l)
                    if msg['type'] == 'ticker' and 'time' in msg:
                        if self.products is None or msg['product_id'] in self.products:
                            yield self.process_ticker_message(msg)
        except:
            print(f'could not stream tickers! ("{path}")')
            raise


    ticker_meta_keys = ('product_id', 'time')
    ticker_sample = collections.namedtuple('TickerSample',
                                           ticker_meta_keys + ('X', ))
    #ticker_feature_keys = ('high_24h', 'low_24h',
    ticker_feature_keys = ('best_ask', 'best_bid', 'high_24h', 'low_24h',
                           'open_24h', 'price', 'volume_24h', 'volume_30d')
    #ticker_feature_keys = ('high_24h', 'low_24h', 'price', 'volume_24h')
    #ticker_feature_keys = ('price', )
    ticker_feature_count = len(ticker_feature_keys)
    ticker_price_index = ticker_feature_keys.index('price')
    ticker_date_format = "%Y-%m-%dT%H:%M:%S.%fZ"

    def process_ticker_message(self, msg):
        msg['time'] = datetime.strptime(msg['time'], self.ticker_date_format)
        features = np.array([float(msg[key]) for key in self.ticker_feature_keys])
        meta = tuple(msg[key] for key in self.ticker_meta_keys) + (features, )
        return self.ticker_sample(*meta)


    def __iter__(self):
        stream = self.stream_batches(self.stream, self.products, self.window)
        queue = []
        for sample in stream:
            queue.append(sample)
            if len(queue) == self.window:
                snapshot = pd.DataFrame.from_records(queue,
                    columns=tickers.batch_sample._fields)
                yield (self._collate(snapshot), snapshot.iloc[-1].prices)
                queue.pop(0)


    @staticmethod
    @torch.no_grad()
    def _collate(snapshot):
        frames = []
        for i, snap in enumerate(snapshot.itertuples(index=False)):
            fs = snap.features
            frames.append((fs - fs.mean()) / fs.std())
        frames = np.array([frames])
        frames = torch.FloatTensor(frames).transpose(0, 1)
        return frames


    batch_meta_keys = ('time', 'mass', 'purity', 'Y', 'edge')
    batch_sample = collections.namedtuple('Sample', batch_meta_keys + ('X', ))


    def tickers_df(self):
        return pd.DataFrame.from_records(self.stream_batches(),
                                         columns=self.batch_sample._fields)


    def stream_batches(self):
        zero = lambda : np.zeros(len(self.ticker_feature_keys))
        self._cached = collections.defaultdict(zero)
        for time, batch in self.stream_buffer():
            if batch:
                batch_products = set(s.product_id for s in batch)
                mass = len(batch)
                purity = (len(batch_products) / len(self.products))
                prices, features = self.collate_ticker_samples(batch)
                yield self.batch_sample(time, mass, purity, prices, False, features)
            else:
                yield self.batch_sample(time, 0.0, 0.0, None, True, None)


    def collate_ticker_samples(self, batch):
        by_product_id = collections.defaultdict(list)
        for sample in batch:
            by_product_id[sample.product_id].append(sample.X)
        for product in self.products:
            if not by_product_id[product]:
                by_product_id[product].append(self._cached[product])
        product_features = [np.array(by_product_id[p]) for p in sorted(by_product_id)]
        product_features = [np.mean(pfs, axis=0) for pfs in product_features]
        prices = []
        g = zip(sorted(self.products), product_features)
        for j, (product, features) in enumerate(g):
            self._cached[product] = features
            prices.append(features[self.ticker_price_index])
        features = np.concatenate(product_features)
        return prices, features


    def stream_buffer(self):
        start = next(self.stream).time
        start = start - timedelta(
                            seconds=start.second,
                            microseconds=start.microsecond)
        buffer = []
        checkpoint = start + timedelta(seconds=self.window)
        for ticker in self.stream:
            if ticker.time > checkpoint:
                # TODO: do raise during inference and to loop restarting?
                #assert buffer
                yield checkpoint, buffer
                buffer = []
                checkpoint += timedelta(seconds=self.window)
            else:
                buffer.append(ticker)


class TickerClient(cbpro.WebsocketClient):

    def on_open(self):
        self.url = "wss://ws-feed.pro.coinbase.com/"
        self.channels = ['ticker']
        self.message_count = 0
        self.queue = Queue()


    def on_message(self, msg):
        self.message_count += 1
        self.queue.put(json.dumps(msg, sort_keys=True))


    def on_close(self):
        print("MessageCount = %i" % self.message_count)
        print("-- Goodbye! --")


    def on_error(self, *ags, **kws):
        super().on_error(*ags, **kws)
        self.stop = False


class TickerDataset(Dataset):
    """Given an input time-series of feature vectors,
    approximate the gradient of a different output time-series of feature vectors.

    In context, the input time-series contains available live ticker data for a set
    of products (e.g. BTC-USD, ETH-USDC). The output time-series consists of price
    data for the same set of products. Thus this class is for approximating the
    direction of future price changes for a set of products given a recent history
    of ticker data about those products.

    Args:
        df (pd.DataFrame): Time-series where each entry has fields {time, X, Y}.
        window (int): How many recent frames (X) are used as the input time-series.
        stride (int): How many future frames (Y) are used as the output time-series.

    """

    epsilon = 0.00001

    def __init__(self, df, window=6, stride=6):
        self.df = df
        self.window = window
        self.stride = stride


    def __getitem__(self, index):
        snapshot = self.df[index:index + self.window + self.stride]
        frames, targets = [], []
        for i, snap in enumerate(snapshot.itertuples(index=False)):
            if i < self.window:
                frames.append(snap.X)
            if i >= self.window - 1:
                targets.append(snap.Y)
        frames, targets = np.array(frames), np.array(targets)
        f_mean, f_std = frames.mean(axis=0), frames.std(axis=0)
        frames = np.array([((f - f_mean) / (f_std + self.epsilon)) for f in frames])
        #targets = 1 * (np.gradient(targets, axis=0).mean(axis=0) > 0)
        targets = 1 * (targets[-1] > targets[0])
        return frames, targets


    @torch.no_grad()
    def _collate(self, samples):
        frames, targets = zip(*samples)
        frames = torch.FloatTensor(frames).transpose(0, 1)
        targets = torch.LongTensor(targets).transpose(0, 1)
        return frames, targets


    def __len__(self):
        return self.df.shape[0] - self.window - self.stride


class TickerSampler(Sampler):

    def __init__(self, dataset, resample=False, products=None):
        self.ids = []
        self.bins = [[[] for x in range(2)] for product in products]
        print(len(dataset), dataset.df.shape)
        edgecount = 0
        for x in np.arange(len(dataset)):
            snapshot = dataset.df[x:x + dataset.window + dataset.stride]
            if not snapshot.edge.values.any():
                self.ids.append(x)
                inputs, targets = dataset[x]
                for y, target in enumerate(targets):
                    self.bins[y][target.item()].append(x)
            else:
                edgecount += 1
        self.n_samples = len(self.ids)
        self._iteration = -1
        print(f'edges: {edgecount}')
        for product, bins in zip(products, self.bins):
            print(f'Product: {product}')
            for cls in range(len(bins)):
                print(f'cls: {cls}, cnt: {len(bins[cls])}')
        self.resample = resample


    def __iter__(self):
        self._iteration += 1
        rng = np.random.RandomState(self._iteration)
        if self.resample:
            n_samples_per_class = 10000
            pivot = 0
            bins = self.bins[pivot]
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


class TickerDataLoader(DataLoader):

    def __init__(self, dataset, device=None, resample=False, products=None, **kws):
        sampler = TickerSampler(dataset, resample=resample, products=products)
        super().__init__(dataset,
                         sampler=sampler,
                         collate_fn=dataset._collate,
                         **kws)
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device


    def __iter__(self):
        for batch, targets in super().__iter__():
            yield batch.to(self.device), targets.to(self.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
		description='Utility for aggregating ticker data')
    parser.add_argument('--output', default='stream.log',
                        help='Path to store ticker data')
    parser.add_argument('--products', default='products.txt',
                        help='Path to list of targeted products')
    args = parser.parse_args()

    with open(args.products, 'r') as f:
        products = [l.strip() for l in f.readlines() if not l.startswith('#')]
        products = [l for l in products if l]

    with open(args.output, 'w') as f:
        for j, ticker in enumerate(stream_tickers_live(products=products)):
            f.write(f'{ticker}\n')
            print(f'tickers: {j}', end='\r')
