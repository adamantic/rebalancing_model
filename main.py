import yfinance as yf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt

today = dt.date.today()
top = '2021-12-01'

# Initialize variables
stock1 = 'QQQ'
stock2 = 'BIL'
weight1 = 0.99
start_date = pd.Timestamp('2000-03-01')
end_date = pd.Timestamp(today)

weight2 = 1 - weight1
rebalance_period = 5  # Number of weekdays between rebalancing

tickers = [stock1, stock2]
portfolio_weights = {stock1: weight1, stock2: weight2}


# Download historical data for tickers
ohlc_data = yf.download(tickers, start_date, end_date)['Close']
ohlc_data = ohlc_data.dropna()  # Remove any rows with missing data

initial_value = 100
portfolio_value = [initial_value]

# Initialize the starting weights based on the initial portfolio value
current_weights = {ticker: (initial_value * portfolio_weights[ticker]) / ohlc_data[ticker][0] for ticker in tickers}

# Create a list of rebalancing dates
rebalance_dates = pd.date_range(start_date, end_date, freq=f'{rebalance_period}B')

for i in range(1, len(ohlc_data)):
    row = ohlc_data.iloc[i]

    # Calculate the new portfolio value
    total_value = sum([row[ticker] * current_weights[ticker] for ticker in tickers])
    portfolio_value.append(total_value)

    # Rebalance the portfolio based on the rebalance_dates
    if row.name in rebalance_dates:
        for ticker in tickers:
            current_weights[ticker] = (total_value * portfolio_weights[ticker]) / row[ticker]

# Create a DataFrame with the portfolio value
portfolio_value_df = pd.DataFrame(portfolio_value, index=ohlc_data.index, columns=['Portfolio Value'])

# Calculate stock1 returns and normalize it to start at 100
stock1_returns = (1 + ohlc_data[stock1].pct_change()).cumprod()
stock1_normalized = initial_value * stock1_returns

# Calculate CAGR
years = (portfolio_value_df.index[-1] - portfolio_value_df.index[0]).days / 365
CAGR = (portfolio_value_df.iloc[-1] / portfolio_value_df.iloc[0]) ** (1 / years) - 1

# Calculate max drawdown
portfolio_value_df['max'] = portfolio_value_df['Portfolio Value'].cummax()
portfolio_value_df['drawdown'] = (portfolio_value_df['Portfolio Value'] - portfolio_value_df['max']) / portfolio_value_df['max']
max_drawdown = portfolio_value_df['drawdown'].min()

# Calculate volatility (annualized)
portfolio_value_df['daily_return'] = portfolio_value_df['Portfolio Value'].pct_change()
volatility = portfolio_value_df['daily_return'].std() * np.sqrt(252)

# Calculate Sharpe ratio (assuming risk-free rate of 0)
sharpe_ratio = (CAGR - 0) / volatility

print("CAGR:", CAGR[0])
print("Max Drawdown:", max_drawdown)
print("Volatility:", volatility)
print("Sharpe Ratio:", sharpe_ratio)


# Plot the portfolio value and the amounts of the two tickers
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(portfolio_value_df.index, portfolio_value_df['Portfolio Value'], label='Portfolio Value')
#ax.plot(stock1_normalized.index, stock1_normalized, label=stock1)  # This line plots stock1_normalized
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
plt.show()
