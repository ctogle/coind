from datetime import datetime, timedelta
from tqdm import tqdm
import collections
import math
import time
import json
import cbpro
import argparse


date_format = '%Y-%m-%d %H:%M:%S-05:00'


def pull_product(product, start, end, granularity):
    public_client = cbpro.PublicClient()
    days = math.ceil((end - start).total_seconds() / (24 * 60 * 60))
    last = time.time()
    current = start
    for day in tqdm(range(days), total=days, desc=f'pulling product: {product}'):
        step = current + timedelta(days=1)
        rates = public_client.get_product_historic_rates(product,
            current.strftime(date_format), step.strftime(date_format),
            granularity=granularity)
        yield rates
        time.sleep(max(0, 0.5 - (time.time() - last)))
        last = time.time()
        current = step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility for aggregating historical data')
    parser.add_argument('--output', default='history.log',
                        help='Path to store historical data')
    parser.add_argument('--products', default='products.txt',
                        help='Path to list of targeted products')
    parser.add_argument('--start', default='2018-01-01 00:00:00-05:00',
                        help='Default: 2018-01-01 00:00:00-05:00')
    parser.add_argument('--end', default='2019-01-01 00:00:00-05:00',
                        help='Default: 2019-01-01 00:00:00-05:00')
    parser.add_argument('--granularity', type=int, default=300,
                        help='Default: 300')
    args = parser.parse_args()

    with open(args.products, 'r') as f:
        products = [l.strip() for l in f.readlines() if not l.startswith('#')]
        products = [l for l in products if l]

    start = datetime.strptime(args.start, date_format)
    end = datetime.strptime(args.end, date_format)

    dataset = collections.defaultdict(list)
    for product in products:
        pulls = pull_product(product, start, end, args.granularity)
        for pull in pulls:
            dataset[product].extend(pull)
            msg = f'product: {product}, datapoints: {len(dataset[product])}'

    with open(args.output, 'w') as f:
        f.write(json.dumps(dataset, indent=4, sort_keys=True))
