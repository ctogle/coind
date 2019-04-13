from datetime import datetime, timedelta
from multiprocessing import Queue
import collections
import pandas as pd
import numpy as np
import json
import cbpro
import time


batch_meta_keys = ('time', 'mass', 'purity', 'prices')
batch_sample = collections.namedtuple('Sample', batch_meta_keys + ('features', ))


def df(*paths, products=None, window=600):
    dfs = []
    for path in paths:
        stream = stream_tickers(path, products)
        batches = stream_batches(stream, products, window)
        df = pd.DataFrame.from_records(batches, columns=batch_sample._fields)
        df.log_edge = False
        df.iloc[0].log_edge = True
        df.iloc[-1].log_edge = True
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def stream_batches(stream, products=None, window=600):
    global _cached
    zero = lambda : np.zeros(len(ticker_feature_keys))
    _cached = collections.defaultdict(zero)
    for time, batch in stream_buffer(stream, window):
        batch_products = set(s.product_id for s in batch)
        mass = len(batch)
        purity = (len(batch_products) / len(products))
        prices, features = collate_ticker_samples(batch, products)
        yield batch_sample(time, mass, purity, prices, features)


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
    return streamable


def collate_ticker_samples(batch, products):
    global _cached
    by_product_id = collections.defaultdict(list)
    for sample in batch:
        by_product_id[sample.product_id].append(sample.features)
    for product in products:
        if not by_product_id[product]:
            by_product_id[product].append(_cached[product])
    product_features = [np.array(by_product_id[p]) for p in sorted(by_product_id)]
    product_features = [np.mean(pfs, axis=0) for pfs in product_features]
    prices = {}
    for product, features in zip(products, product_features):
        _cached[product] = features
        prices[product] = features[ticker_price_index]
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
            assert buffer
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


ticker_meta_keys = ('product_id', 'sequence', 'time', 'trade_id')
ticker_sample = collections.namedtuple('TickerSample',
                                       ticker_meta_keys + ('features', ))
#ticker_feature_keys = ('best_ask', 'best_bid', 'high_24h', 'low_24h',
ticker_feature_keys = ('high_24h', 'low_24h',
                       'open_24h', 'price', 'volume_24h', 'volume_30d')
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


if __name__ == '__main__':
    products = [
        'BAT-USDC', 'BCH-BTC', 'BCH-EUR', 'BCH-GBP', 'BCH-USD', 'BTC-EUR', 'BTC-GBP',
        'BTC-USD', 'BTC-USDC', 'CVC-USDC', 'DAI-USDC', 'DNT-USDC', 'EOS-BTC', 'EOS-EUR',
        'EOS-USD', 'ETC-BTC', 'ETC-EUR', 'ETC-GBP', 'ETC-USD', 'ETH-BTC', 'ETH-EUR',
        'ETH-GBP', 'ETH-USD', 'ETH-USDC', 'GNT-USDC', 'LOOM-USDC', 'LTC-BTC', 'LTC-EUR',
        'LTC-GBP', 'LTC-USD', 'MANA-USDC', 'MKR-BTC', 'MKR-USDC', 'REP-BTC', 'REP-EUR',
        'REP-USD', 'XLM-BTC', 'XLM-EUR', 'XLM-USD', 'XRP-BTC', 'XRP-EUR', 'XRP-USD',
        'ZEC-USDC', 'ZRX-BTC', 'ZRX-EUR', 'ZRX-USD',
    ]
    with open('./stream.log', 'w') as f:
        for j, ticker in enumerate(stream_tickers_live(products=products)):
            print(f'tickers: {j}', end='\r')
            f.write(json.dumps(ticker, sort_keys=True))
            f.write('\n')