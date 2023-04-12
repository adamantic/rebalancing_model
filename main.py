import yfinance as yf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt

today = dt.date.today()

# Initialize variables
stock1 = 'TQQQ'
stock2 = 'BIL'
weight1 = 0.8
start_date = pd.Timestamp('2000-03-01')
end_date = pd.Timestamp(today)

weight2 = 1 - weight1
rebalance_period = 1  # Number of weekdays between rebalancing

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

# Calculate Stock1 stats
stock1_normalized = stock1_normalized.iloc[1:]  # Drop the first row to avoid division by zero



# CAGR
stock1_years = (stock1_normalized.index[-1] - stock1_normalized.index[0]).days / 365
stock1_CAGR = (stock1_normalized.iloc[-1] / stock1_normalized.iloc[0]) ** (1 / stock1_years) - 1

# Max Drawdown
stock1_normalized_df = pd.DataFrame(stock1_normalized)
stock1_normalized_df['max'] = stock1_normalized_df[stock1].cummax()
stock1_normalized_df['drawdown'] = (stock1_normalized_df[stock1] - stock1_normalized_df['max']) / stock1_normalized_df['max']
stock1_max_drawdown = stock1_normalized_df['drawdown'].min()

# Volatility (annualized)
stock1_normalized_df['daily_return'] = stock1_normalized_df[stock1].pct_change()
stock1_volatility = stock1_normalized_df['daily_return'].std() * np.sqrt(252)

# Sharpe Ratio (assuming risk-free rate of 0)
stock1_sharpe_ratio = (stock1_CAGR - 0) / stock1_volatility

# Calculate total return for stock1
initial_stock1_value = stock1_normalized.iloc[0]
ending_stock1_value = stock1_normalized.iloc[-1]
stock1_total_return = (ending_stock1_value / initial_stock1_value) - 1

print(stock1, "Stats")
print(f"Total Return: {stock1_total_return:.0f}x")
print(f"CAGR: {stock1_CAGR*100:.0f}%")
print(f"Max Drawdown: {stock1_max_drawdown*100:.0f}%")
print(f"Volatility: {stock1_volatility*100:.0f}%")
print(f"Sharpe Ratio: {stock1_sharpe_ratio:.0f}%")



# Calculate stats for the portfolio

# Calculate total return for the portfolio
initial_portfolio_value = portfolio_value_df.iloc[0, 0]
ending_portfolio_value = portfolio_value_df.iloc[-1, 0]
total_return = (ending_portfolio_value / initial_portfolio_value) - 1


# CAGR
years = (portfolio_value_df.index[-1] - portfolio_value_df.index[0]).days / 365
CAGR = (portfolio_value_df.iloc[-1] / portfolio_value_df.iloc[0]) ** (1 / years) - 1

# Max Drawdown
portfolio_value_df['max'] = portfolio_value_df['Portfolio Value'].cummax()
portfolio_value_df['drawdown'] = (portfolio_value_df['Portfolio Value'] - portfolio_value_df['max']) / portfolio_value_df['max']
max_drawdown = portfolio_value_df['drawdown'].min()

# Volatility (annualized)
portfolio_value_df['daily_return'] = portfolio_value_df['Portfolio Value'].pct_change()
volatility = portfolio_value_df['daily_return'].std() * np.sqrt(252)

# Sharpe Ratio (assuming risk-free rate of 0)
sharpe_ratio = (CAGR - 0) / volatility

print("\nPortfolio Stats")
print(f"Total Return: {total_return:0.0f}x")
print(f"CAGR: {CAGR[0]*100:0.0f}%")
print(f"Max Drawdown: {max_drawdown*100:0.0f}%")
print(f"Volatility: {volatility*100:0.0f}%")
print(f"Sharpe Ratio: {sharpe_ratio.values[0]*100:0.0f}%")

# Plot the portfolio value and the amounts of the two tickers
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(portfolio_value_df.index, portfolio_value_df['Portfolio Value'], label='Portfolio Value')
#ax.plot(stock1_normalized.index, stock1_normalized, label=stock1)  # This line plots stock1_normalized
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
plt.show()

