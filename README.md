## Aggregating Ticker Data

Collect JSON encoded ticker messages in a log for a list of specified products.
These logs are used for training the predictive model.

`python -m coind.tickers --output tickerstream.log --products products.txt`
`python -m coind.tickers --help` for more info

notebooks/scrape\_to\_table.ipynb provides plotting of ticker time series
data and measurements of time delays between messages per ticker, which limits
the resolution of the predictive model with poorly chosen products to target.


## Training Predictive Model

notebooks/train\_model.ipynb demonstrates training and saving a predictive model.
Training loss, validation loss, and validation accuracies are also shown.


## Simulating Market Agent

notebooks/agent.ipynb demonstrates using the predictive model on historical or
live data to make rule-based decisions to simulate market activity.


