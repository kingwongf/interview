import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests
import json
import yfinance as yf
import re
import requests
import glob
from risk_parity import risk_parity_weighting
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

def cap_w_sp500():
    url = 'https://www.slickcharts.com/sp500'
    header = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    r = requests.get(url, headers=header)
    df = pd.read_html(r.text)[0]

    df['Symbol'] = df['Symbol'].str.replace('.','-')
    return df[['Symbol','Weight']].set_index('Symbol')['Weight']*0.01

def _binary_signal(temp_fast, temp_mid, temp_slow, direction='long', signal_type='all'):
    if signal_type=='all':
        return (temp_fast > temp_mid) & (temp_mid > temp_slow) if direction=='long' else (temp_fast < temp_mid) & (temp_mid < temp_slow)
    elif signal_type == 'fast_mid':
        return (temp_fast > temp_mid) if direction == 'long' else (temp_fast < temp_mid)

    elif signal_type == 'mid_slow':
        return (temp_mid> temp_slow) if direction == 'long' else (temp_mid < temp_slow)


def trend_trading(df, st_date, end_date, sigs=(20,50,150), transaction_cost=0, direction='long', signal_type='all', w=None, binary_signal=None):
    ''' return long side returns
     ## Requested Portfolio
        ## Long trend following, short index
        ## equal weighted in stocks, and long short
    '''
    temp_fast, temp_mid, temp_slow = trend_follow_sigs(df, sigs)

    binary_signal = _binary_signal(temp_fast, temp_mid, temp_slow, direction=direction, signal_type=signal_type) if binary_signal is None else binary_signal

    ret_df = df.loc[st_date:end_date].pct_change(1).fillna(0)


    arr_transaction_cost = [0, 0, 0, 0, transaction_cost] * (len(ret_df.index) // 5) + [0] * (len(ret_df.index) % 5)

    holdings = binary_signal.astype(int).shift(1)[st_date:end_date]
    if w is None:
        w = holdings.sum(axis=1)
        w = (1 / w).replace([np.inf, -np.inf], 0)
    w_holdings = holdings.mul(w, axis='index') if w.index.name == 'Date' else holdings.mul(w)


    long_side_ret = ((ret_df * w_holdings).sum(axis=1) - pd.Series(arr_transaction_cost,index=ret_df.index))
    return long_side_ret

def momentum_trading(df, st_date, end_date):

    mom_score = df.pct_change(252)
    z_mom_score = (mom_score - mom_score.rolling(252, min_periods=1).mean()) / mom_score.rolling(252, min_periods=1).std()
    z_mom_score = z_mom_score.loc[st_date:end_date]
    ret_df = df.loc[st_date:end_date].pct_change(1).shift(1).fillna(0)

    port_ret = pd.Series(index= ret_df.index)
    rebalance =0
    for date, row in ret_df.iterrows():

        if rebalance%25==0 or rebalance==0:
            w = pd.Series(index=z_mom_score.columns, name=date)
            td_z_score = z_mom_score.loc[date].sort_values(ascending=False).index
            long = td_z_score[:int(len(td_z_score)*0.20)]
            short = td_z_score[int(len(td_z_score)*0.80):]
            w.loc[long] = 1/len(long)
            w.loc[short] = -1/len(short)

        port_ret.loc[date] = (ret_df.loc[date]*w).sum()
        rebalance+=1
    return port_ret


def optimal_trend_trading(df, st_date, end_date, sigs=(20,50,150), transaction_cost=0, direction='long', signal_type='all', w=None, binary_signal=None):
    ''' return long side returns
     ## Requested Portfolio
        ## Long trend following, short index
        ## equal weighted in stocks, and long short
    '''
    temp_fast, temp_mid, temp_slow = trend_follow_sigs(df, sigs)

    entry_signal = temp_fast > temp_mid

    exit_signal = None

    ret_df = df.loc[st_date:end_date].pct_change(1).fillna(0)


    arr_transaction_cost = [0, 0, 0, 0, transaction_cost] * (len(ret_df.index) // 5) + [0] * (len(ret_df.index) % 5)

    holdings = binary_signal.astype(int).shift(1)[st_date:end_date]
    if w is None:
        w = holdings.sum(axis=1)
        w = (1 / w).replace([np.inf, -np.inf], 0)



    w_holdings = holdings.mul(w, axis='index') if w.index.name == 'Date' else holdings.mul(w)


    # print(holdings.mul(w))
    # print(ret_df * w_holdings)
    # print(ret_df)

    long_side_ret = ((ret_df * w_holdings).sum(axis=1) - pd.Series(arr_transaction_cost,index=ret_df.index))
    return long_side_ret

def sig_s(df, entry, exit, p_v, sl):
    if len(entry)==0:
        return p_v

    ## take the first index as first entry
    p_v.append((entry[0], 'entry'))

    ## store all exit indices if they are larger than current entry
    exit = [x for x in exit if x > p_v[-1][0]]


    if not exit:
        entry =[]

    if exit:
        if sl is not None:
            sl_vec = df.loc[p_v[-1][0] +1: exit[0], 'index_return'].cumsum() < sl
            p_v.append(sl.idxmax(), 'sl_exit' if sl_vec.any() else (sl_vec.index[-1], 'exit'))
        else:
            ## mark the closest exit
            p_v.append((exit[0], 'exit'))

    ## look entry indices that are larger than last exit index
    entry = [x for x in entry if x > p_v[-1][0]]

    return sig_s(df,entry, exit, p_v, sl)

def pos_idx_vec(df, sl):
    entry_idx = np.where(df['entry']>0)[0].tolist()
    exit_idx = np.where(df['exit']>0)[0].tolist()

    if (not exit_idx) or (not entry_idx):
        return []
    return sig_s(df, entry_idx, exit_idx, [], sl)

def get_position_df(df, entry_df, exit_df):
    ''' returns a holding/ position dataframe of each day for each ticker '''
    df2 = df.copy().reset_index(drop=False)
    entry_df, exit_df = entry_df.astype(int), exit_df.astype(int)

    pos_df = df2['Date'].to_frame()

    for ticker in df2.columns:
        if ticker != 'Date':
            entry_exit_df = pd.concat([entry_df[ticker].rename('entry'), exit_df[ticker].rename('exit')], axis=1)

            pos_ticker = pd.DataFrame(pos_idx_vec(entry_exit_df, None), columns=['int_index', 'signal'])
            pos_df[ticker] = pos_ticker.set_index('int_index')['signal'].map({'entry': 1, 'exit': 0})

    return pos_df.ffill().set_index('Date').shift(1)


def equal_weighting(pos_df):
    return (1 / pos_df.sum(axis=1)).replace([np.inf, -np.inf], 0)

def equal_risk_weighting(df):
    ret_df = df.pct_change()
    ## TODO: finish the vcv
    risk_parity_weighting(vcv, 'equal')

def trend_trading_2(df, entry, exit, st_date, ed_date, arr_transaction_cost=0):

    pos2_df = get_position_df(df, entry, exit)

    w = equal_weighting(pos2_df)

    w_holdings = pos2_df.mul(w, axis='index')

    ret_df = df.loc[st_date:ed_date].pct_change().fillna(0)

    long_side_ret = ((ret_df * w_holdings).sum(axis=1) - pd.Series(arr_transaction_cost, index=ret_df.index))
    return long_side_ret


def check_short_trend_trading(sigs=(20,50,150), transaction_cost=0):

    start_bt_date_1yr_plus = '2014-09-13'
    st_date = '2015-09-13'
    end_date = '2020-09-13'
    df, stoxx600 = get_df_stoxx600(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_date)



    temp_fast, temp_mid, temp_slow = trend_follow_sigs(df, sigs)
    binary_signal = (temp_fast < temp_mid) & (temp_mid < temp_slow)

    # binary_signal = (temp_fast < temp_mid)


    ret_df = df[st_date:end_date].pct_change(1).fillna(0)
    arr_transaction_cost = [0, 0, 0, 0, transaction_cost] * (len(ret_df.index) // 5) + [0] * (len(ret_df.index) % 5)

    holdings = binary_signal.astype(int).shift(1)[st_date:end_date]



    ## plotting

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(binary_signal.tail(10))
    #     pass
    #
    # ticker = 'YAR.OL'
    # fig, axes = plt.subplots(4,1, sharex=True)
    # df[ticker].plot(ax=axes[0])
    # (1 + holdings['YAR.OL']*(-ret_df[ticker])).cumprod().plot(ax=axes[1])
    #
    #
    # binary_signal[ticker].astype(int).plot(ax=axes[2])
    # temp_fast[ticker].loc[st_date:end_date].plot(ax=axes[3], label=sigs[0], c='g')
    # temp_mid[ticker].loc[st_date:end_date].plot(ax=axes[3], label=sigs[1], c='b')
    # temp_slow[ticker].loc[st_date:end_date].plot(ax=axes[3], label=sigs[2], c='r')
    #
    # axes[3].legend()
    # plt.subplots_adjust(left=0.03, bottom=0.08, right=1, top=0.97, wspace=0.20, hspace=0.1)
    # plt.show()
    # plt.close()
    # (1 - short_side_ret).cumprod().plot()
    # plt.ylabel('Cumulative Return')
    # plt.show()
    # plt.close()



    w = holdings.sum(axis=1)
    w = (1 / w).replace([np.inf, -np.inf], 0)

    short_side_ret = ((ret_df * holdings.mul(w, axis='index')).sum(axis=1) - pd.Series(arr_transaction_cost,
                                                                                         index=w.index))

    short_side_ret = short_side_ret.fillna(0)

    return (1 - short_side_ret).cumprod()[-1]




def alph_beta(equity_curve, b_equity_curve):
    ret_df = equity_curve.rename('p').to_frame().pct_change()
    ret_df['b'] = b_equity_curve.pct_change()
    ret_df = ret_df.dropna()
    res = sm.OLS(ret_df['p'], sm.add_constant(ret_df['b'])).fit()
    return res.params[0], res.params[1]

def port_stats(equity_curve, b_equity_curve):
    port_ret = equity_curve.pct_change().dropna()

    ret_df = equity_curve.rename('p').to_frame().pct_change()
    ret_df['b'] = b_equity_curve.pct_change()
    ret_df = ret_df.dropna()

    res = sm.OLS(ret_df['p'], sm.add_constant(ret_df['b'])).fit()


    ## avg. alpha
    avg_alpha = np.mean(ret_df['p'] - ret_df['b'])



    ## Annualised Information Ratio

    yr_ret_port_df = equity_curve.resample('1y').last().pct_change(1)
    yr_ret_b_df = b_equity_curve.resample('1y').last().pct_change(1)
    annual_std = (equity_curve.pct_change(252) - b_equity_curve.pct_change(252)).resample('1y').std()

    # print(annual_std)

    # print(yr_ret_port_df)
    annual_IR = (yr_ret_port_df - yr_ret_b_df) / annual_std

    ## combined end PnL

    end_PnL = (ret_df['p'] - ret_df['b'] +1).cumprod()[-1]

    return pd.Series([end_PnL,
                      avg_alpha,
                      res.params[0],
                      res.params[1],
                      port_ret.std(),
                      min(equity_curve.rolling(126).apply(dd).dropna()),
                      np.mean(annual_IR)], index=['Profit', 'avg. alpha', 'reg. alpha', 'reg. beta', 'Daily Vol', 'MaxDD', 'IR'])

def performance_analysis(equity_curve, b_equity_curve, port_name=None, plot=False,plt_save_path=None):
    port_ret = equity_curve.pct_change().dropna()


    ## End Equity
    # print("Ending Equity/ Profitability")
    # print(equity_curve.tail(1))

    ## Portfolio Return

    # print(port_ret.describe())
    # port_ret.describe()

    plt.rcParams["figure.dpi"] = 800
    port_ret.hist(bins=80)
    plt.ylabel('Freq.')
    plt.xlabel('Daily Returns')
    plt.savefig(f"{plt_save_path}hist.png", bbox_inches='tight', figsize=(5,3.5))
    # plt.title(f"Daily Returns of {port_name}")

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
    plt.rcParams["figure.dpi"] = 800
    if plt_save_path is not None:
        equity_curve.rolling(126).apply(dd).plot(legend=True)
        b_equity_curve.rolling(126).apply(dd).plot(legend=True)
        plt.ylabel('6 months Rolling Max DD')

        plt.savefig(f"{plt_save_path}maxdd.png", bbox_inches='tight', figsize=(5,3.5))
        if plot:
            plt.show()
        plt.close()
    else:
        equity_curve.rolling(126).apply(dd).rename('long_short_trend').plot(legend=True)
        b_equity_curve.rolling(126).apply(dd).rename('benchmark').plot(legend=True)
        plt.ylabel('6 months Rolling Max DD')

        plt.savefig(f"{plt_save_path}maxdd.png", bbox_inches='tight', figsize=(5, 3.5))
        if plot:
            plt.show()
        plt.close()


    ## Annualised Information Ratio

    yr_ret_port_df = equity_curve.resample('1y').last().pct_change(1)
    yr_ret_b_df = b_equity_curve.resample('1y').last().pct_change(1)
    annual_std = (equity_curve.pct_change(252) - b_equity_curve.pct_change(252)).resample('1y').std()

    # print(annual_std)

    # print(yr_ret_port_df)
    annual_IR = (yr_ret_port_df - yr_ret_b_df) / annual_std


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





def get_df_stoxx600(start_bt_date_1yr_plus='2007-01-01', end_bt_date='2010-01-01'):
    ## redownload if today not exsits
    today = date.today()
    if pd.to_datetime(end_bt_date) == today:
        try:
            df = pd.read_pickle(f"closes_stoxx600_{str(today)}.pkl")  # temp_close_stoxx600
        except Exception as e:
            dwnld_df(today)
            df = pd.read_pickle(f"closes_stoxx600_{str(today)}.pkl")  # temp_close_stoxx600
    else:
        datasheet_path_pattern = "closes_stoxx600_*"
        datasheet_paths = sorted([path for path in glob.iglob(datasheet_path_pattern)])
        df = pd.read_pickle(datasheet_paths[-1])  # temp_close_stoxx600




    df.ffill(inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    df.sort_index(inplace=True)


    # Data Quality issues

    df.loc['2020-07-22', 'CRDA.L'] = 5616.00
    df.loc[:'2018-12-28', 'KGF.L'] = df.loc[:'2018-12-28', 'KGF.L']*100 if (df.loc['2018-12-28', 'KGF.L'] < 100) else df.loc[:'2018-12-28', 'KGF.L']
    df.loc[:'2018-12-28', 'TW.L'] = df.loc[:'2018-12-28', 'TW.L']*100 if (df.loc['2018-12-28', 'TW.L'] < 100) else df.loc[:'2018-12-28', 'TW.L']

    df.loc['2018-12-31', 'KGF.L'] = 208.5000
    df.loc['2018-12-31', 'TW.L'] = 136.9000

    df.loc['2019-01-01', 'KGF.L'] = 207.5
    df.loc['2019-01-01', 'TW.L'] = 136.2500

    df.loc['2019-07-19', 'KGF.L'] = 218.6
    df.loc['2019-07-19', 'TW.L'] = 165.00

    df.loc['2020-08-14', 'DLN.L'] = 2816.0

    df.loc['2020-09-02', 'EVR.L'] = 334.5000
    df.loc['2020-09-08', 'EVR.L'] = 321.4000

    df.loc['2020-08-14', 'MNDI.L'] = 1529.000

    # df.loc[:'2012-05-21', 'KGHA.F'] = np.nan

    df.loc['2020-09-09', 'BVIC.L'] = 857.5
    df.loc['2020-09-09', 'EVR.L'] = 328.4000
    df.loc['2020-09-09', 'BNZL.L'] = 2398.00
    df.loc['2020-09-09', 'HAS.L'] = 120.0000
    df.loc['2020-09-09', 'OMU.JO'] = 1155.00

    df.loc['2007-02-19', 'GIVN.SW'] = 1112.47
    df.loc['2007-04-25', 'GIVN.SW'] = 1185.32

    df.loc['2020-09-14', 'EDEN.PA'] = 43.26

    df.loc['2004-12-31', 'DSV.CO'] = 37.10

    df.loc['2005-03-15':'2005-03-30', 'EQNR.OL'] = np.nan

    df.loc['2005-05-05', 'EQNR.OL'] =np.nan
    df.loc['2005-05-16':'2005-05-17', 'EQNR.OL'] =np.nan

    df.loc['2004-05-31','SGRO.L'] = np.nan
    df.loc['2004-05-31','DSV.CO'] = 28.00
    df.drop(['KGHA.F','EDEN.PA','SGRO.L'], axis=1, inplace=True)

    df = _replace_abnormal_prices(df)

    df = df.loc[start_bt_date_1yr_plus:end_bt_date]
    df.dropna(how='all', axis=1, inplace=True)




    ## Stoxx 600, index price
    stoxx = yf.Ticker('^STOXX')
    stoxx = stoxx.history(period="max")['Close'][start_bt_date_1yr_plus:end_bt_date].ffill()


    return df, stoxx

def trend_follow_sigs(df, freqs):
    ''' remeber parse uncut df to get at least +slow freq dates'''
    fast, mid, slow = freqs
    return df.rolling(fast).mean().fillna(0), df.rolling(mid).mean().fillna(0), df.rolling(slow).mean().fillna(0)


def cont_trend_follow_sigs(df, freqs):
    ''' remeber parse uncut df to get at least +slow freq dates'''
    fast, mid, slow = freqs
    return df.rolling(fast).mean().fillna(0) -  df.rolling(mid).mean().fillna(0),  df.rolling(mid).mean().fillna(0) - df.rolling(slow).mean().fillna(0)


def _get_sp500_tickers():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    tickers = list(map(lambda x: x.replace('.', '-'), tickers))
    return tickers

def get_df_sp500(start_bt_date_1yr_plus, end_bt_date):
    return _get_df('sp500', start_bt_date_1yr_plus, end_bt_date)


def df_sanity_check(df, date):
    '''
    To deal with Data Quality issues on Yahoo finance data, this function
    returns the largest to lowest returns on a given date

    df: price df
    date: identify by eyeballing the largest jump in PnL chart
    '''

    return df.pct_change().loc[date].sort_values(ascending=False)

def _get_nasdaq100_tickers():
    ''' you'll need selenium and the chrome driver for this  '''

    import pandas as pd
    import time
    from selenium import webdriver
    from bs4 import BeautifulSoup
    import pathlib
    pathlib.Path().absolute()


    driver = webdriver.Chrome(str(pathlib.Path().absolute())+ "/chromedriver")
    base_url = "https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index"
    driver.get(base_url)
    time.sleep(12)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    div = soup.select_one("table")
    tables = pd.read_html(str(div))

    return tables[0]['Symbol'].tolist()


def get_df_nasdaq100(start_bt_date_1yr_plus, end_bt_date):
    return _get_df('nasdaq100', start_bt_date_1yr_plus, end_bt_date)


def _replace_abnormal_prices(df):
    ret = df.pct_change()
    fwd_ret = ret.shift(-1)
    ''' 
                modify prices that are like the following

                Date
                2020-09-01    783.500
                2020-09-02    789.000
                2020-09-03    780.000
                2020-09-04    769.500
                2020-09-07    786.000
                2020-09-08    792.500
                2020-09-09      7.975
                2020-09-10    791.000
                2020-09-11    800.000
                2020-09-14    795.500
                2020-09-15    802.500
                2020-09-16    789.500
                Name: IGG.L, dtype: float64

                to

                Date
                2020-09-01    783.5
                2020-09-02    789.0
                2020-09-03    780.0
                2020-09-04    769.5
                2020-09-07    786.0
                2020-09-08    792.5
                2020-09-09    792.5
                2020-09-10    791.0
                2020-09-11    800.0
                2020-09-14    795.5
                2020-09-15    802.5
                2020-09-16    789.5
                Name: IGG.L, dtype: float64

                downside is it also changes the following

                Date
                2020-09-01      35.650002
                2020-09-02      35.650002
                2020-09-03      35.650002
                2020-09-04      35.650002
                2020-09-07      35.650002
                2020-09-08      35.650002
                2020-09-09      35.650002
                2020-09-10      35.650002
                2020-09-11      35.650002
                2020-09-14      35.650002
                2020-09-15      35.650002
                2020-09-16    3575.000000

                to



                '''
    for col in ret.columns:

        problems_dates = fwd_ret[col][fwd_ret[col] > 1.0].index.tolist()
        if problems_dates:
            df[col].loc[problems_dates] = np.nan
    df = df.ffill()
    '''
    Next we need to change the below
    WWH.L
    price
    Date
    2014-09-16            NaN
    2014-09-17            NaN
    2014-09-18            NaN
    2014-09-19            NaN
    2014-09-22            NaN
             ...     
    2020-09-10      35.650002
    2020-09-11      35.650002
    2020-09-14      35.650002
    2020-09-15      35.650002
    2020-09-16    3575.000000
    
    '''
    ret = df.pct_change()
    for col in ret.columns:

        problems_dates = ret[col][ret[col] > 1.0].index.tolist()
        if problems_dates:
            df[col].loc[problems_dates] = np.nan
    df = df.ffill()
    return df

def get_df_ftse250(start_bt_date_1yr_plus, end_bt_date):

    df, b_df = _get_df('ftse250', start_bt_date_1yr_plus, end_bt_date)

    ## drop bad ticker columns

    df = df.drop(['ICGT.L', 'HILS.L', 'GSS.L','SONG.L', 'RCP.L'], axis=1)
    # df.loc['2014-09-30','3IN.L'] = 1.871850*100

    df.loc[:'2017-05-02', 'PSH.L'] = np.nan
    df.loc['2020-09-02', 'UKW.L'] = 141.4000
    df.loc['2020-09-08', 'UKW.L'] = 134.6


    df.loc['2020-09-02', 'EQN.L'] = 110.0000
    df.loc['2020-09-02', 'WIZZ.L'] = 3590.000


    df.loc['2020-09-08', 'EQN.L'] = 113.4000

    df.loc['2020-09-09', 'BVIC.L'] = 857.5
    df.loc['2020-09-09', 'EVR.L'] = 328.4000
    df.loc['2020-09-09', 'BNZL.L'] = 2398.00

    df.loc['2020-07-22', 'CRDA.L'] = 5616.00


    df.loc['2019-07-19', 'KGF.L'] = 218.6
    df.loc['2019-07-19', 'TW.L'] = 165.00

    df.loc['2020-08-14', 'DLN.L'] = 2816.0

    df.loc['2020-09-02', 'EVR.L'] = 334.5000
    df.loc['2020-09-08', 'EVR.L'] = 321.4000

    df.loc['2020-08-14', 'MNDI.L'] = 1529.000

    df.loc[:'2013-06-19','BCPT.L'] = np.nan

    df = _replace_abnormal_prices(df)


    return df[start_bt_date_1yr_plus:end_bt_date], b_df[start_bt_date_1yr_plus:end_bt_date]



def _get_ftse250_tickers():
    tickers= pd.read_html("https://en.wikipedia.org/wiki/FTSE_250_Index#cite_note-3")[1]['Ticker[4]'].tolist()
    return map(lambda x:x.upper().replace('.','') +'.L', tickers)

def _get_df(index_name, start_bt_date_1yr_plus, end_bt_date):

    today = date.today()
    if pd.to_datetime(end_bt_date) == today:
        try:
            df = pd.read_pickle(f"closes_{index_name}_{str(today)}.pkl")  # temp_close_stoxx600
        except Exception as e:
            tickers = globals()[f"_get_{index_name}_tickers"]()
            tickers_str = " ".join(tickers)
            new_data = yf.download(tickers=tickers_str, period='max')
            new_data.dropna(how='all', axis=1, inplace=True)
            new_data['Close'].to_pickle(f"closes_{index_name}_{str(today)}.pkl")
            df = new_data['Close']
    else:
        datasheet_path_pattern = f"closes_{index_name}_*"
        datasheet_paths = sorted([path for path in glob.iglob(datasheet_path_pattern)])

        df = pd.read_pickle(datasheet_paths[-1])


    df.ffill(inplace=True)
    df.sort_index(inplace=True)

    df = df[start_bt_date_1yr_plus:end_bt_date] if index_name!='ftse250' else df
    df.dropna(how='all', axis=1, inplace=True)

    index_dict  = {'ftse250':'^FTMC', 'sp500': 'SPY', 'nasdaq100':'^NDX'}

    b_df = yf.Ticker(index_dict[index_name])
    b_df = b_df.history(period="max")['Close']

    return df, b_df

def print_sth():
    print(globals())
