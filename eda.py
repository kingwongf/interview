import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("aapl_amz_koyfin.csv")

print(df.columns.tolist())
df['Date']= pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df['r_aapl'] = df['XLK Close']/ df['AAPL Close']
df['r_amzn'] = df['XLK Close']/ df['AMZN Close'].astype(float)

fig, (ax0,ax1) = plt.subplots(1,2)
df['r_aapl'].plot(ax=ax0)
df['r_amzn'].plot(ax=ax1)
# plt.show()
plt.close()

## cointegrated? copula? cointegrated with a drift?
## set drift as it's moving average, will that make spread/ residual stationary?

df['r_aapl'] = df['r_aapl'] - df['r_aapl'].rolling(60).mean()
df['r_amzn'] = df['r_amzn'] - df['r_amzn'].rolling(60).mean()
fig, (ax0,ax1) = plt.subplots(1,2)
df['r_aapl'].plot(ax=ax0)
df['r_amzn'].plot(ax=ax1)
plt.show()
plt.close()

## test for stationarity, ADF

