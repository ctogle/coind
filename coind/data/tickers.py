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


batch_meta_keys = ('time', 'mass', 'purity', 'Y', 'edge')
batch_sample = collections.namedtuple('Sample', batch_meta_keys + ('X', ))


def tickers_df(*paths, products=None, window=600):
    streams = (stream_tickers(path, products) for path in paths)
    stream = (i for it in streams for i in it)
    batches = stream_batches(stream, products, window)
    df = pd.DataFrame.from_records(batches, columns=batch_sample._fields)
    return df


def stream_batches(stream, products=None, window=600):
    global _cached
    zero = lambda : np.zeros(len(ticker_feature_keys))
    _cached = collections.defaultdict(zero)
    for time, batch in stream_buffer(stream, window):
        if batch:
            batch_products = set(s.product_id for s in batch)
            mass = len(batch)
            purity = (len(batch_products) / len(products))
            prices, features = collate_ticker_samples(batch, products)
            yield batch_sample(time, mass, purity, prices, False, features)
        else:
            yield batch_sample(time, 0.0, 0.0, None, True, None)


def streamable(*paths, products=None, window=600):
    streams = []
    for path in paths:
        streams.append(stream_tickers(path, products))
    stream = (i for it in streams for i in it)
    ticker_times = collections.defaultdict(list)
    for ticker in stream:
        ticker_times[ticker.product_id].append(ticker.time)
    streamable = []
    for ticker, times in ticker_times.items():
        if len(times) > 1:
            departure_times = [(times[j] - times[j - 1]) for j in range(1, len(times))]
            departure_times = [dt.total_seconds() for dt in departure_times]
            if max(departure_times) < window:
                streamable.append(ticker)
    assert streamable, 'found no streamable products!'
    return streamable


def collate_ticker_samples(batch, products):
    global _cached
    by_product_id = collections.defaultdict(list)
    for sample in batch:
        by_product_id[sample.product_id].append(sample.X)
    for product in products:
        if not by_product_id[product]:
            by_product_id[product].append(_cached[product])
    product_features = [np.array(by_product_id[p]) for p in sorted(by_product_id)]
    product_features = [np.mean(pfs, axis=0) for pfs in product_features]
    prices = [[] for product in products]
    for j, (product, features) in enumerate(zip(sorted(products), product_features)):
        _cached[product] = features
        prices[j].append(features[ticker_price_index])
    features = np.concatenate(product_features)
    return prices, features


def stream_buffer(stream, window):
    start = next(stream).time
    start = start - timedelta(
                        seconds=start.second,
                        microseconds=start.microsecond)
    buffer = []
    checkpoint = start + timedelta(seconds=window)
    for ticker in stream:
        if ticker.time > checkpoint:
            # TODO: do raise during inference and to loop restarting?
            #assert buffer
            yield checkpoint, buffer
            buffer = []
            checkpoint += timedelta(seconds=window)
        else:
            buffer.append(ticker)


def stream_tickers(path, products=None):
    try:
        with open(path, 'r') as f:
            for l in f:
                msg = json.loads(l)
                if msg['type'] == 'ticker' and 'time' in msg:
                    if products is None or msg['product_id'] in products:
                        yield process_ticker_message(msg)
    except:
        print(f'could not stream tickers! ("{path}")')
        raise


def stream_tickers_live(products=None, doraise=False):
    wsClient = TickerClient(products=products)
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


def stream_tickers_log(path, products=None):
    with open(path, 'w') as f:
        for ticker in stream_tickers_live(products):
            f.write(f'{json.dumps(ticker)}\n')


ticker_meta_keys = ('product_id', 'time')
ticker_sample = collections.namedtuple('TickerSample',
                                       ticker_meta_keys + ('X', ))
#ticker_feature_keys = ('best_ask', 'best_bid', 'high_24h', 'low_24h',
#ticker_feature_keys = ('high_24h', 'low_24h',
#                       'open_24h', 'price', 'volume_24h', 'volume_30d')
ticker_feature_keys = ('high_24h', 'low_24h', 'price', 'volume_24h')
#ticker_feature_keys = ('price', )
ticker_feature_count = len(ticker_feature_keys)
ticker_price_index = ticker_feature_keys.index('price')
ticker_date_format = "%Y-%m-%dT%H:%M:%S.%fZ"


def process_ticker_message(msg):
    msg['time'] = datetime.strptime(msg['time'], ticker_date_format)
    features = np.array([float(msg[key]) for key in ticker_feature_keys])
    meta = tuple(msg[key] for key in ticker_meta_keys) + (features, )
    return ticker_sample(*meta)


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
        epsilon = 0.00001
        frames = np.array([((f - f_mean) / (f_std + epsilon)) for f in frames])
        #targets = 1 * (np.gradient(targets, axis=0).mean(axis=0) > 0)
        targets = 1 * (targets[-1] > targets[0])
        return frames, targets


    def __len__(self):
        return self.df.shape[0] - self.window - self.stride


    @torch.no_grad()
    def _collate(self, samples):
        frames, targets = zip(*samples)
        frames = torch.FloatTensor(frames).transpose(0, 1)
        targets = torch.LongTensor(targets).transpose(0, 1)
        return frames, targets


    @classmethod
    def from_log(cls, ticker_log, window=3, stride=3, stream_window=600):
        products = streamable(ticker_log, window=stream_window)
        df = tickers_df(ticker_log, products, window=stream_window)
        return cls(df, window, stride)


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


def dataloaders(*ticker_logs,
                batch_size=16, num_workers=4,
                window=6, stride=6, products=None, stream_window=600):
    if products is None:
        products = streamable(*ticker_logs, window=stream_window)
    df = tickers_df(*ticker_logs, products=products, window=stream_window)

    many = df.shape[0]
    assert many > 3 * (window + stride),\
           f'insufficient data... (only {many} samples)'
    teny = int(0.05 * many)
    tiny = int(0.05 * many)
    test_df = df.iloc[-tiny:]
    val_df = df.iloc[-(teny + tiny):-tiny]
    index = np.random.RandomState(0).permutation(np.arange(many - (teny + tiny)))
    train_df = df.iloc[index]

    train_set = TickerDataset(train_df, window, stride)
    val_set = TickerDataset(val_df, window, stride)
    test_set = TickerDataset(test_df, window, stride)

    dl_kws = dict(num_workers=num_workers, batch_size=batch_size, products=products)
    train_dl = TickerDataLoader(train_set, resample=True, **dl_kws)
    val_dl = TickerDataLoader(val_set, **dl_kws)
    test_dl = TickerDataLoader(test_set, **dl_kws)

    return products, ticker_feature_count, train_dl, val_dl, test_dl


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
