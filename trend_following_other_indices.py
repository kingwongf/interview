import pandas as pd
from bt_tools import get_df_stoxx, performance_analysis, trend_follow_sigs, cont_trend_follow_sigs,compute_MV_weights, simple_ew_backtester, get_df_sp500
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import re
import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mstats


def orig_long_trend_bt(df, b_df, sigs= (20,50,150), transaction_cost=0, plot=False):


    temp_fast, temp_mid, temp_slow = trend_follow_sigs(df, sigs)


    today = pd.datetime.today().date()
    five_yrs_ago_date = "-".join([str(today.year - 5), str(today.month), str(today.day)])
    b_df = b_df[five_yrs_ago_date:]
    ret_b = b_df.pct_change().fillna(0)

    ## Requested Portfolio
    ## Long trend following, short index
    ## equal weighted in stocks, and long short

    binary_signal = (temp_fast > temp_mid) & (temp_mid > temp_slow)
    ret_df = df[five_yrs_ago_date:].pct_change(1).fillna(0)

    arr_transaction_cost = [0, 0, 0, 0, transaction_cost] * (len(ret_df.index) // 5) + [0] * (len(ret_df.index) % 5)

    holdings = binary_signal.shift(1)[five_yrs_ago_date:]
    w = holdings.sum(axis=1)
    w = (1 / w).replace([np.inf, -np.inf], 0)

    port_ret = ((ret_df * holdings.mul(w, axis='index')).sum(axis=1) - pd.Series(arr_transaction_cost,
                                                                                 index=w.index)).rename(
        'Long').to_frame()
    port_ret['Short'] = ret_b

    port_ret = port_ret.fillna(0)

    ew_eq_curve = (port_ret + 1).cumprod()

    ## Long/ Short Side Plot
    ew_eq_curve.plot(figsize=(100, 50), legend=True)
    plt.title('Long Short Orig Portfolio')
    if plot:
        plt.show()
    plt.close()

    ## combined PnL

    combined_PnL = (port_ret['Long'] - port_ret['Short'] + 1).cumprod()
    combined_PnL.plot(figsize=(100, 50), legend=True)

    plt.title('Combined PnL Orig Port')
    if plot:
        plt.show()
    plt.close()

    return combined_PnL, (1 + ret_b).cumprod()

def continuous_sig_bt(df, b_df, combined_PnL, sigs= (20,50,150), transaction_cost=0, plot=False):
    today = pd.datetime.today().date()
    five_yrs_ago_date = "-".join([str(today.year - 5), str(today.month), str(today.day)])
    b_df = b_df[five_yrs_ago_date:]
    ret_b = b_df.pct_change().fillna(0)

    ret_df = df[five_yrs_ago_date:].pct_change()

    signal_fast, signal_slow = cont_trend_follow_sigs(df, sigs)

    holdings_fast = np.sign(signal_fast).replace(-1, 0)[five_yrs_ago_date:]
    holdings_slow = np.sign(signal_slow).replace(-1, 0)[five_yrs_ago_date:]

    w_fast = (1 / holdings_fast.sum(axis=1)).replace([np.inf, -np.inf], 0)
    w_slow = (1 / holdings_slow.sum(axis=1)).replace([np.inf, -np.inf], 0)

    arr_transaction_cost = [0, 0, 0, 0, transaction_cost] * (len(ret_b.index) // 5) + [0] * (len(ret_b.index) % 5)

    port_ret_fast = ((ret_df * holdings_fast.mul(w_fast, axis='index')).sum(axis=1) - ret_b - pd.Series(
        arr_transaction_cost, index=ret_b.index)).fillna(0)

    port_ret_slow = ((ret_df * holdings_slow.mul(w_slow, axis='index')).sum(axis=1) - ret_b - pd.Series(
        arr_transaction_cost, index=ret_b.index)).fillna(0)

    print(port_ret_slow)

    ew_eq_curve_fast = (port_ret_fast + 1).cumprod()
    ew_eq_curve_slow = (port_ret_slow + 1).cumprod()

    fast, slow = "_".join(list(map(str,sigs[:-1]))), "_".join(list(map(str,sigs[1:])))

    ew_eq_curve = ew_eq_curve_fast.rename(fast).to_frame()
    ew_eq_curve[slow] = ew_eq_curve_slow
    ew_eq_curve["original"] = combined_PnL

    ew_eq_curve.plot(figsize=(100, 50), legend=True, title="Portfolios PnLs of Seperated Signals")
    if plot:
        plt.show()
    plt.close()

    sep_signal_ret = port_ret_fast.rename('fast').to_frame()
    sep_signal_ret['slow'] = port_ret_slow

    sep_signal_ret = pd.DataFrame(mstats.winsorize(sep_signal_ret, [0.05, 0.05]), index=sep_signal_ret.index,
                                  columns=sep_signal_ret.columns)

    # sns.jointplot(sep_signal_ret['fast'],sep_signal_ret['slow'])

    ## adjust by price volatility, so big moves of slow movers weight more than big moves of big movers
    std_df = ret_df.rolling(60).std()
    price_vol = df * std_df

    norm_signal_fast = ((signal_fast) / price_vol).replace([np.inf, -np.inf, np.nan], 0)
    norm_signal_slow = ((signal_slow) / price_vol).replace([np.inf, -np.inf, np.nan], 0)


    w_trading_rules = compute_MV_weights(sep_signal_ret.cov())

    combined_signal = norm_signal_fast * w_trading_rules[0] + norm_signal_slow * w_trading_rules[1]

    combined_ew_ret_0, combined_ew_eq_curve_0 = simple_ew_backtester(combined_signal, ret_df, ret_b, five_yrs_ago_date
                                                                     , rules=0, transaction_cost=transaction_cost)

    combined_ew_ret_1, combined_ew_eq_curve_1 = simple_ew_backtester(combined_signal, ret_df, ret_b, five_yrs_ago_date
                                                                     , rules=1, transaction_cost=transaction_cost)

    combined_ew_ret_2, combined_ew_eq_curve_2 = simple_ew_backtester(combined_signal, ret_df, ret_b, five_yrs_ago_date
                                                                     , rules=2, transaction_cost=transaction_cost)

    combined_ew_eq_curve_0 = combined_ew_eq_curve_0.rename('0').to_frame()
    combined_ew_eq_curve_0['1'] = combined_ew_eq_curve_1
    combined_ew_eq_curve_0['2'] = combined_ew_eq_curve_2

    combined_ew_eq_curve_0['orig'] = combined_PnL
    combined_ew_eq_curve_0['benchmark'] = (1+ret_b).cumprod()

    combined_ew_eq_curve_0.plot(figsize=(50, 100), legend=True, title="Portfolios PnLs of Combined Continuous Signals")
    plt.show()
    plt.close()

today = pd.datetime.today().date()
start_bt_date_1yr_plus="-".join([str(today.year-6), str(today.month), str(today.day)])
end_bt_date=today
start_bt_date = "-".join([str(today.year-5), str(today.month), str(today.day)])


## Stoxx 600
# df_stoxx, b_stoxx = get_df_stoxx(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)
# orig_combined_PnL, _ = orig_long_trend_bt(df_stoxx, b_stoxx)
# continuous_sig_bt(df_stoxx, b_stoxx, orig_combined_PnL)

## SP 500
df_sp, b_sp = get_df_sp500(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)
orig_combined_PnL, _ = orig_long_trend_bt(df_sp, b_sp, plot=True)
continuous_sig_bt(df_sp, b_sp, orig_combined_PnL, plot=True)