import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests
import json
import yfinance as yf
import re
import os
import seaborn as sns
from scipy.stats import mstats
def compute_MV_weights(cov_m):
    inv_covar = np.linalg.inv(cov_m)
    u = np.ones(len(cov_m))
    return np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u))


def simple_ew_backtester(bt_signal_df, bt_ret_df, b_ret, five_yrs_ago_date, rules=0, transaction_cost=0):
    bt_signal_df = bt_signal_df.where(bt_signal_df < rules, np.inf)
    bt_signal_df = bt_signal_df.where(bt_signal_df > rules, np.nan)
    bt_signal_df = bt_signal_df.replace(np.inf, 1).fillna(0)

    bt_holdings = bt_signal_df.shift(1)[five_yrs_ago_date:]
    bt_w = (1 / bt_holdings.sum(axis=1)).replace([np.inf, -np.inf], 0)

    arr_transaction_cost = [0, 0, 0, 0, transaction_cost] * (len(bt_ret_df.index) // 5) + [0] * (len(bt_ret_df.index) % 5)

    bt_port_ret = (bt_ret_df * bt_holdings.mul(bt_w, axis='index')).sum(axis=1) - b_ret - pd.Series(arr_transaction_cost,
                                                                                                 index=bt_w.index)
    bt_ew_eq_curve = (bt_port_ret.fillna(0) + 1).cumprod()

    return bt_port_ret, bt_ew_eq_curve


def dd(ts):
    return np.min(ts / np.maximum.accumulate(ts)) - 1

def alph_beta(equity_curve, b_equity_curve):
    ret_df = equity_curve.rename('p').to_frame().pct_change()
    ret_df['b'] = b_equity_curve.pct_change()
    ret_df = ret_df.dropna()
    res = sm.OLS(ret_df['p'], sm.add_constant(ret_df['b'])).fit()
    return res.params[0], res.params[1]

def performance_analysis(equity_curve, b_equity_curve, port_name=None, plot=False):
    port_ret = equity_curve.pct_change().dropna()
    port_ret.plot(legend=True, title=f"Daily Returns of {port_name}")
    if plot:
        plt.show()
    plt.close()

    ## End Equity
    # print("Ending Equity/ Profitability")
    # print(equity_curve.tail(1))

    ## Portfolio Return

    # print(port_ret.describe())
    port_ret.describe()
    port_ret.hist(bins=100)
    plt.title(f"Daily Returns of {port_name}")

    if plot:
        plt.show()
    plt.close()

    # print('strategy alpha beta')
    ret_df = equity_curve.rename('p').to_frame().pct_change()
    ret_df['b'] = b_equity_curve.pct_change()
    ret_df = ret_df.dropna()

    res = sm.OLS(ret_df['p'], sm.add_constant(ret_df['b'])).fit()
    # print(res.summary())
    ## Max Drawdown
    equity_curve.rolling(126).apply(dd).plot(legend=True, title=f"Rolling Max Drawdown of {port_name}")
    if plot:
        plt.show()
    plt.close()
    print(f"max dd: {min(equity_curve.rolling(126).apply(dd).dropna())}")

    ## Annualised Information Ratio

    yr_ret_port_df = equity_curve.resample('1y').last().pct_change(1)
    yr_ret_b_df = b_equity_curve.resample('1y').last().pct_change(1)
    annual_std = equity_curve.pct_change(252).resample('1y').std()

    # print(annual_std)

    # print(yr_ret_port_df)
    annual_IR = (yr_ret_port_df - yr_ret_b_df) / annual_std

    # print("IR")
    # print(annual_IR)
    # print(np.mean(annual_IR))

    #     pd.Series([equity_curve.tail(1).values[0],
    #                res.params[0],
    #               res.params[1],
    #               port_ret.std(),
    #               min(equity_curve.rolling(126).apply(dd).dropna()),
    #               np.mean(annual_IR)], index= ['Profit', 'Alpha', 'Beta', 'MaxDD', 'IR'])
    # print(f"Profit: {equity_curve.tail(1).values[0]} | Alpha: {res.params[0]}| Beta: {res.params[1]}| Vol: {port_ret.std()}| MaxDD: {min(equity_curve.rolling(126).apply(dd).dropna())} | IR: {np.mean(annual_IR)}")

    # print(pd.Series([equity_curve.tail(1).values[0],
    #                  res.params[0],
    #                  res.params[1],
    #                  port_ret.std(),
    #                  min(equity_curve.rolling(126).apply(dd).dropna()),
    #                  np.mean(annual_IR)], index=['Profit', 'Alpha', 'Beta', 'Daily Vol', 'MaxDD', 'IR']))

    return pd.Series([equity_curve.tail(1).values[0],
                      res.params[0],
                      res.params[1],
                      port_ret.std(),
                      min(equity_curve.rolling(126).apply(dd).dropna()),
                      np.mean(annual_IR)], index=['Profit', 'Alpha', 'Beta', 'Daily Vol', 'MaxDD', 'IR'])



def df_percent_null(check_df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

        null_count = check_df.isnull().mean().sort_values()
        print(null_count)
        print(len(null_count[null_count < 0.50]))

def dwnld_df(today):
    csv_url = "https://github.com/kingwongf/interview/blob/master/SXXGR.csv?raw=true"
    company_names = pd.read_csv(csv_url, error_bad_lines=False)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    tickers = []
    for name in company_names['Company'].tolist():
        yhoo_url = f"http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={name}&region=1&lang=en&callback=YAHOO.Finance.SymbolSuggest.ssCallback"
        r = requests.get(yhoo_url, headers=headers)
        try:
            ticker = [x for x in json.loads(re.search(r'\((.*?)\)', r.text).group(1))['ResultSet']['Result'] if
                      x['exch'] != 'PNK' and x['exchDisp'] != 'OTC Markets' and x['exchDisp'] != 'NYSE']
            print(f"{name}: {ticker[0]['symbol']}")
            tickers.append(ticker[0]['symbol'])
        except Exception as e:
            print(f"fail {name}")
    ## filter out possible US companies, not within stoxx 600 based on it must have an . extension
    tickers = [ticker for ticker in tickers if '.' in ticker]
    print(len(tickers))
    tickers_str = " ".join(tickers)
    new_data = yf.download(tickers=tickers_str, period='max')
    new_data.dropna(how='all', axis=1, inplace=True)
    new_data['Close'].to_pickle(f"closes_stoxx600_{str(today)}.pkl")

    ## Stoxx 600, index price
    stoxx = yf.Ticker('^STOXX')
    stoxx = stoxx.history(period="max")

    stoxx.to_pickle(f"stoxx600_{str(today)}.pkl")





def get_df_stoxx(start_bt_date_1yr_plus='2007-01-01', end_bt_date='2010-01-01'):
    ## redownload if today not exsits
    today = pd.datetime.today().date()
    try:
        df = pd.read_pickle(f"closes_stoxx600_{str(today)}.pkl")  # temp_close_stoxx600
    except Exception as e:
        dwnld_df(today)
        df = pd.read_pickle(f"closes_stoxx600_{str(today)}.pkl")  # temp_close_stoxx600


    df.ffill(inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    df.sort_index(inplace=True)


    # Data Quality issues
    df.loc['2020-07-22', 'CRDA.L'] = 5616.00
    df.loc[:'2019-01-01', 'KGF.L'] = df.loc[:'2019-01-01', 'KGF.L']*100 if (df.loc['2019-01-01', 'KGF.L'] < 100) else df.loc[:'2019-01-01', 'KGF.L']
    df.loc[:'2019-01-01', 'TW.L'] = df.loc[:'2019-01-01', 'TW.L']*100 if (df.loc['2019-01-01', 'TW.L'] < 100) else df.loc[:'2019-01-01', 'TW.L']

    df.loc['2019-07-19', 'KGF.L'] = 218.6
    df.loc['2019-07-19', 'TW.L'] = 165.00

    df.loc['2020-08-14', 'DLN.L'] = 2816.0

    df.loc['2020-09-02', 'EVR.L'] = 334.5000
    df.loc['2020-09-08', 'EVR.L'] = 321.4000

    df.loc['2020-08-14', 'MNDI.L'] = 1529.000


    df = df[start_bt_date_1yr_plus:end_bt_date]
    df.dropna(how='all', axis=1, inplace=True)

    ## Stoxx 600, index price
    stoxx = yf.Ticker('^STOXX')
    stoxx = stoxx.history(period="max")['Close']


    return df, stoxx

def trend_follow_sigs(df, freqs):
    ''' remeber parse uncut df to get at least +slow freq dates'''
    fast, mid, slow = freqs
    return df.rolling(fast).mean().fillna(0), df.rolling(mid).mean().fillna(0), df.rolling(slow).mean().fillna(0)

def cont_trend_follow_sigs(df, freqs):
    ''' remeber parse uncut df to get at least +slow freq dates'''
    fast, mid, slow = freqs
    return df.rolling(fast).mean().fillna(0) -  df.rolling(mid).mean().fillna(0),  df.rolling(mid).mean().fillna(0) - df.rolling(slow).mean().fillna(0)

def get_df_sp500(start_bt_date_1yr_plus, end_bt_date):
    today = pd.datetime.today().date()
    try:
        df = pd.read_pickle(f"closes_sp500_{str(today)}.pkl")  # temp_close_stoxx600
    except Exception as e:
        dfs = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = dfs[0]['Symbol'].tolist()
        tickers = list(map(lambda x:x.replace('.','-'), tickers))
        tickers_str = " ".join(tickers)
        new_data = yf.download(tickers=tickers_str, period='max')
        new_data.dropna(how='all', axis=1, inplace=True)
        new_data['Close'].to_pickle(f"closes_sp500_{str(today)}.pkl")

        df = new_data['Close']

    df.ffill(inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    df.sort_index(inplace=True)

    df = df[start_bt_date_1yr_plus:end_bt_date]
    df.dropna(how='all', axis=1, inplace=True)


    sp500 = yf.Ticker('SPY')
    sp500 = sp500.history(period="max")['Close']

    return df, sp500
