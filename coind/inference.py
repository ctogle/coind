import torch
import numpy as np
import pandas as pd
import coind.tickers as tickers


class InferenceStream:

    def __init__(self, products, source=None, window=3, stream_window=600):
        self.products = products
        if source is None:
            self.stream = tickers.stream_tickers_live(products)
        else:
            self.stream = tickers.stream_tickers(source, products)
        self.window = window
        self.stream_window = stream_window


    def __iter__(self):
        stream = tickers.stream_batches(self.stream, self.products, self.stream_window)
        queue = []
        for sample in stream:
            queue.append(sample)
            if len(queue) == self.window:
                snapshot = pd.DataFrame.from_records(queue,
                    columns=tickers.batch_sample._fields)
                yield (self._collate(snapshot), snapshot.iloc[-1].prices)
                queue.pop(0)


    @staticmethod
    def _collate(snapshot):
        frames = []
        for i, snap in enumerate(snapshot.itertuples(index=False)):
            fs = snap.features
            frames.append((fs - fs.mean()) / fs.std())
        frames = np.array([frames])
        frames = torch.FloatTensor(frames).transpose(0, 1)
        return frames


def inference(model, window=3, stream_window=600, source=None):
    stream = InferenceStream(model.products,
                             source=source,
                             window=window,
                             stream_window=stream_window)
    for batch, prices in stream:
        prediction = model(batch)
        bull = {}
        for product, hypothesis in zip(model.products, prediction):
            bull[product] = bool(torch.argmax(hypothesis[0]).item())
        yield (bull, prices)
