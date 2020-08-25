#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mstats
def dd(ts):
    return np.min(ts / np.maximum.accumulate(ts)) - 1


# In[3]:


def performance_analysis(equity_curve, b_equity_curve, port_name=None):
    port_ret = equity_curve.pct_change().dropna()
    port_ret.plot(legend=True, title=f"Daily Returns of {port_name}")
    plt.show()
    plt.close()
    
    ## End Equity
    print("Ending Equity/ Profitability")
    print(equity_curve.tail(1))
    
    ## Portfolio Return
    
    print(port_ret.describe())
    port_ret.describe()
    port_ret.hist(bins=100)
    plt.title(f"Daily Returns of {port_name}")
    plt.show()
    plt.close()
    
    print('strategy alpha beta')
    ret_df = equity_curve.rename('p').to_frame().pct_change()
    ret_df['b'] = b_equity_curve.pct_change()
    ret_df = ret_df.dropna()
    
    res = sm.OLS(ret_df['p'], sm.add_constant(ret_df['b'])).fit()
    print(res.summary())
    ## Max Drawdown
    equity_curve.rolling(126).apply(dd).plot(legend=True, title=f"Rolling Max Drawdown of {port_name}")
    plt.show()
    plt.close()
    print(f"max dd: {min(equity_curve.rolling(126).apply(dd).dropna())}")
    
    ## Annualised Information Ratio

    yr_ret_port_df = equity_curve.resample('1y').first().pct_change(1)
    yr_ret_b_df = b_equity_curve.resample('1y').first().pct_change(1)
    annual_std = equity_curve.pct_change(252).resample('1y').std()
    
    
    # print(annual_std)
    
    # print(yr_ret_port_df)
    annual_IR = (yr_ret_port_df- yr_ret_b_df) / annual_std

    print("IR")
    print(annual_IR)
    print(np.mean(annual_IR))
    
    #     pd.Series([equity_curve.tail(1).values[0],
    #                res.params[0],
    #               res.params[1],
    #               port_ret.std(),
    #               min(equity_curve.rolling(126).apply(dd).dropna()),
    #               np.mean(annual_IR)], index= ['Profit', 'Alpha', 'Beta', 'MaxDD', 'IR'])
    # print(f"Profit: {equity_curve.tail(1).values[0]} | Alpha: {res.params[0]}| Beta: {res.params[1]}| Vol: {port_ret.std()}| MaxDD: {min(equity_curve.rolling(126).apply(dd).dropna())} | IR: {np.mean(annual_IR)}")
    
    print(pd.Series([equity_curve.tail(1).values[0],
                        res.params[0],
                        res.params[1],
                        port_ret.std(),
                        min(equity_curve.rolling(126).apply(dd).dropna()),
                        np.mean(annual_IR)], index= ['Profit', 'Alpha', 'Beta', 'Daily Vol', 'MaxDD', 'IR']))
    
    return pd.Series([equity_curve.tail(1).values[0],
                        res.params[0],
                        res.params[1],
                        port_ret.std(),
                        min(equity_curve.rolling(126).apply(dd).dropna()),
                        np.mean(annual_IR)], index= ['Profit', 'Alpha', 'Beta', 'Daily Vol', 'MaxDD', 'IR'])
    
    


# In[4]:


import pandas as pd
import numpy as np

df = pd.read_pickle('new_close_stoxx600.pkl') # temp_close_stoxx600

df.ffill(inplace=True)
df.dropna(how='all', axis=1, inplace=True)
# df.Date = pd.to_datetime(df.Date)
# df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
today = pd.datetime.today().date()
df = df[today - pd.Timedelta('6Y'):]


## Data Quality issue
## Comment out to see issue in later print lines
df.loc['2020-07-22', 'CRDA.L'] = 5616.00

## Rerun all cells to modify weekly transaction cost (return based)
transaction_cost =  0.000 #0.0010


# In[5]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(ret_df.loc['2020-03-16':].head(100).describe().T.sort_values('max', ascending=False))
    
    # print(ew_eq_curve_20_50_150.loc['2020-01-14':'2020-07-29'])
    # print(ret_df.loc['2020-03-18'].sort_values())
    # print(ret_df.loc[np.isclose(ret_df['KESKOB.HE'], 3.023567)]['KESKOB.HE'])
    # print(df.loc['2020-03-15':'2020-03-24']['BMRA'])
    # print(signal_20_50.loc['2020-03-15':'2020-03-24']['BMRA'])
    # print(holdings_20_50.loc['2020-03-15':'2020-03-24']['BMRA'])
    # print((ret_df*holdings_20_50.mul(w_20_50, axis='index')).loc['2020-03-13':'2020-03-24'].describe().T.sort_values('max', ascending=False)['max'].head(10))
    # print((ret_df*holdings_20_50.mul(w_20_50, axis='index')).loc['2020-03-15':'2020-03-24']['BMRA'])
    # print((ret_df*holdings_20_50.mul(w_20_50, axis='index')).loc['2020-03-15':'2020-03-24']['BMRA'])

    
    # print((ret_df*holdings_20_50.mul(w_20_50, axis='index')).loc['2020-07-14':'2020-07-29'].describe().T.sort_values('max', ascending=False)['max'].head(10))
    # print((ret_df*holdings_20_50.mul(w_20_50, axis='index')).loc['2020-07-14':'2020-07-29']['CRDA.L'])
    print(df.loc['2020-07-14':'2020-07-29']['CRDA.L'])


# In[6]:


df_20MA = df.rolling(20).mean().fillna(0)
df_50MA = df.rolling(50).mean().fillna(0)
df_150MA = df.rolling(150).mean().fillna(0)


# In[7]:


stoxx600 = pd.read_pickle('stoxx600.pkl')['Close'][today - pd.Timedelta('5Y'):]
ret_stoxx600 = stoxx600.pct_change().fillna(0)
eq_curve_stoxx600 = (ret_stoxx600+ 1).cumprod()


# In[8]:


## Requested Portfolio
## Long trend following, short index
## equal weighted in stocks, and long short


binary_signal = (df_20MA > df_50MA)&(df_50MA>df_150MA)
ret_df = df[today - pd.Timedelta('5Y'):].pct_change(1).fillna(0)

arr_transaction_cost = [0,0,0,0,transaction_cost]*(len(ret_df.index)//5) + [0]*(len(ret_df.index)%5)


holdings = binary_signal.shift(1)[today - pd.Timedelta('5Y'):]
w = holdings.sum(axis=1)
w=(1/w).replace([np.inf, -np.inf], 0)


port_ret = ((ret_df*holdings.mul(w, axis='index')).sum(axis=1) - pd.Series(arr_transaction_cost, index=w.index)).rename('Long').to_frame()
port_ret['Short'] = ret_stoxx600

port_ret = port_ret.fillna(0)

ew_eq_curve = (port_ret+1).cumprod()


# In[9]:


((ret_df*holdings.mul(w, axis='index')).sum(axis=1) - pd.Series(arr_transaction_cost, index=w.index)).rename('Long').to_frame()


# In[10]:


## Long/ Short Side Plot
ew_eq_curve.plot(legend=True, title="Long Trend and Short Index Cumulative Returns")
plt.show()
plt.close()


# In[11]:


## combined PnL

combined_PnL = (port_ret['Long'] - port_ret['Short'] + 1).cumprod()
combined_PnL.plot(legend=True, title="Original Trend Following Combined PnL")

plt.show()
plt.close()


# In[12]:


performance_analysis(combined_PnL, (port_ret['Short'] + 1).cumprod(), port_name="Original Portfolio")


# In[13]:


## what if we seperate the two trend trading rules, what the portfolio will look like?
signal_20_50 = df_20MA - df_50MA
signal_50_150 = df_50MA - df_150MA


holdings_20_50 = np.sign(signal_20_50).replace(-1, 0)[today - pd.Timedelta('5Y'):]
holdings_50_150 = np.sign(signal_50_150).replace(-1, 0)[today - pd.Timedelta('5Y'):]

w_20_50 = (1/holdings_20_50.sum(axis=1)).replace([np.inf, -np.inf], 0)
w_50_150 = (1/holdings_50_150.sum(axis=1)).replace([np.inf, -np.inf], 0)


arr_transaction_cost = [0,0,0,0,transaction_cost]*(len(ret_df.index)//5) + [0]*(len(ret_df.index)%5)

#print((ret_df*holdings_20_50.mul(w_20_50, axis='index')).sum(axis=1).shape)

#print(holdings_20_50.shape)
# print(transaction_cost)
port_ret_20_50 = ((ret_df*holdings_20_50.mul(w, axis='index')).sum(axis=1) - ret_stoxx600 - pd.Series(arr_transaction_cost, index=w.index)).fillna(0)
port_ret_50_150 = ((ret_df*holdings_50_150.mul(w, axis='index')).sum(axis=1) - ret_stoxx600 - pd.Series(arr_transaction_cost, index=w.index)).fillna(0)
            
            
ew_eq_curve_20_50 = (port_ret_20_50+1).cumprod()
ew_eq_curve_50_150 = (port_ret_50_150+1).cumprod()

 #((ret_df*holdings_20_50.mul(w, axis='index')).sum(axis=1) - ret_stoxx600 - pd.Series(arr_transaction_cost, index=w.index)).rename('Long').to_frame()
# ((ret_df*holdings_50_150.mul(w, axis='index')).sum(axis=1) - ret_stoxx600 - pd.Series(arr_transaction_cost, index=w.index)).rename('Long').to_frame()


# In[14]:


len([0,0,0,0,transaction_cost]*(len(ret_df.index)//5) + [0]*(len(ret_df.index)%5))


# In[15]:


ew_eq_curve_20_50_150 = ew_eq_curve_20_50.rename('20_50').to_frame()
ew_eq_curve_20_50_150['50_150'] = ew_eq_curve_50_150
ew_eq_curve_20_50_150["original"] = combined_PnL


# In[16]:


ew_eq_curve_20_50_150.plot(legend=True, title="Portfolios PnLs of Seperated Signals")
plt.show()
plt.close()


# 
# We also discovered a most likely data quality issue, from yahoo finance
# seems like two decimal points are missing for CRDA.L
# We'll edit it for now and rerun previous analysis
# 
# 
# 
# 
# 
# 

# In[17]:


performance_analysis(ew_eq_curve_20_50_150['20_50'], (port_ret['Short'] + 1).cumprod(), port_name="20_50 Signal")


# In[18]:


performance_analysis(ew_eq_curve_20_50_150['50_150'], (port_ret['Short'] + 1).cumprod(), port_name="50_150 Signal")


# In[19]:


sep_signal_ret = port_ret_20_50.rename('20_50').to_frame()
sep_signal_ret['50_150'] = port_ret_50_150


# In[20]:


## The returns of the trading are plagued by outliers
sns.jointplot(sep_signal_ret['20_50'],sep_signal_ret['50_150'])


# In[21]:


## Can remove the outliers by winsorizing 
sep_signal_ret = pd.DataFrame(mstats.winsorize(sep_signal_ret, [0.05, 0.05]), index=sep_signal_ret.index, columns=sep_signal_ret.columns)


# In[22]:


sns.jointplot(sep_signal_ret['20_50'],sep_signal_ret['50_150'])


# In[23]:


sep_signal_ret.corr()


# In[24]:


def compute_MV_weights(cov_m):
    inv_covar = np.linalg.inv(cov_m)
    u = np.ones(len(cov_m))
    return np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u))


# In[25]:


## adjust by price volatility, so big moves of slow movers weight more than big moves of big movers
std_df = ret_df.rolling(60).std()
price_vol = df*std_df


# In[26]:


## try to build different portfolios to see if there's improvement with different weighting scheme


signal_20_50 = df_20MA - df_50MA
signal_50_150 = df_50MA -df_150MA

norm_signal_20_50 = ((signal_20_50)/price_vol).replace([np.inf, -np.inf, np.nan],0)
norm_signal_50_150 = ((signal_50_150)/price_vol).replace([np.inf, -np.inf, np.nan],0)

## we can weight the two signals by inversing the covariance matrix of the seperated portfolio returns, will ignore insample calculation for now


w_trading_rules = compute_MV_weights(sep_signal_ret.cov())
w_trading_rules


# In[27]:


combined_signal = norm_signal_20_50*w_trading_rules[0] + norm_signal_50_150*w_trading_rules[1]


# In[28]:


combined_signal.describe()


# In[29]:


def simple_ew_backtester(bt_signal_df, bt_ret_df, b_ret, rules=0, transaction_cost=transaction_cost):
    
    bt_signal_df = bt_signal_df.where(bt_signal_df < rules,np.inf)
    bt_signal_df = bt_signal_df.where(bt_signal_df > rules,np.nan)
    bt_signal_df = bt_signal_df.replace(np.inf,1).fillna(0)

    bt_holdings = bt_signal_df.shift(1)[today - pd.Timedelta('5Y'):]
    bt_w = (1/bt_holdings.sum(axis=1)).replace([np.inf, -np.inf], 0)


    arr_transaction_cost = [0,0,0,0,transaction_cost]*(len(ret_df.index)//5) + [0]*(len(ret_df.index)%5)
    
    
    
    bt_port_ret = (bt_ret_df*holdings.mul(bt_w, axis='index')).sum(axis=1) - b_ret - pd.Series(arr_transaction_cost, index=w.index)
    bt_ew_eq_curve = (bt_port_ret.fillna(0)+1).cumprod()
    

    return bt_port_ret, bt_ew_eq_curve
    
    
    
    
    


# In[30]:


combined_ew_ret_0, combined_ew_eq_curve_0 = simple_ew_backtester(combined_signal, ret_df, ret_stoxx600, rules=0, transaction_cost=transaction_cost)
combined_ew_ret_1, combined_ew_eq_curve_1 = simple_ew_backtester(combined_signal, ret_df, ret_stoxx600, rules=1, transaction_cost=transaction_cost)
combined_ew_ret_2, combined_ew_eq_curve_2 = simple_ew_backtester(combined_signal, ret_df, ret_stoxx600, rules=2, transaction_cost=transaction_cost)


# In[31]:



ew_eq_curve_20_50_150['ew_mv_signal_0'] = combined_ew_eq_curve_0
ew_eq_curve_20_50_150['ew_mv_signal_1'] = combined_ew_eq_curve_1
ew_eq_curve_20_50_150['ew_mv_signal_2'] = combined_ew_eq_curve_2





# In[33]:


ew_eq_curve_20_50_150.plot(legend=True, title="Portfolios PnLs of Combined Continuous Signals")
plt.show()
plt.close()


# In[34]:


performance_analysis(ew_eq_curve_20_50_150['ew_mv_signal_0'], (port_ret['Short'] + 1).cumprod(), port_name="Equal Eeighted Min. Variance Singal threshold 0")


# In[35]:


performance_analysis(ew_eq_curve_20_50_150['ew_mv_signal_1'], (port_ret['Short'] + 1).cumprod(), port_name="Equal Eeighted Min. Variance Signal threshold 1")


# In[36]:


performance_analysis(ew_eq_curve_20_50_150['ew_mv_signal_2'], (port_ret['Short'] + 1).cumprod(), port_name="Equal Eeighted Min. Variance Signal threshold 2")


# In[37]:


def simple_mv_backtester(bt_signal_df, ret_df, min_periods=20, rules=1.645):
    
    bt_signal_df = bt_signal_df.where(bt_signal_df < rules,np.inf)
    bt_signal_df = bt_signal_df.where(bt_signal_df > rules,np.nan)
    bt_signal_df = bt_signal_df.replace(np.inf,1).dropna(how='all', axis=1).fillna(0)
    bt_holdings = bt_signal_df.shift(1)[today - pd.Timedelta('5Y'):]
    
    bt_ret_df = ret_df*bt_holdings
    bt_ret_df = ret_df.replace(-0,0)

    
    ## to avoid sigular matrix error due to zero prices returns, adding tiny tiny nois to the covariance matrix
    noised_bt_ret_df = bt_ret_df+0.00000001*np.random.rand(*bt_ret_df.shape)
    noised_bt_ret_df = noised_bt_ret_df.dropna(how='all', axis=1).fillna(0)
    cov_df = noised_bt_ret_df.expanding( min_periods=min_periods).cov().dropna(axis=0) # window=min_periods,
    

    w_MV = cov_df.groupby(level=0, axis=0).apply(compute_MV_weights).apply(pd.Series)
    
    w_MV.columns = cov_df.columns
    
    bt_port_ret = (bt_ret_df.mul(w_MV, axis='index').fillna(0)).sum(axis=1) 
    bt_mv_eq_curve = (bt_port_ret.fillna(0)+1).cumprod()
    

    return bt_port_ret, bt_mv_eq_curve
    
    
    


# In[38]:


mv_port_ret_2, mv_eq_curv_2 = simple_mv_backtester(combined_signal, ret_df, rules=2)


# In[39]:


mv_eq_curv_2.plot(legend=True,title="Portfolio PnL of Minimum Variance")
plt.show()
plt.close()


# In[40]:


ew_eq_curve_20_50_150['MV_port_2'] = mv_eq_curv_2


# In[41]:


ew_eq_curve_20_50_150.plot(legend=True,title="All Portfolios PnLs")
plt.show()
plt.close()


# In[42]:


performance_analysis(ew_eq_curve_20_50_150['MV_port_2'], (port_ret['Short'] + 1).cumprod(), port_name="Min. Variance threshold 0")






