import bt_tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from matplotlib.lines import Line2D
import seaborn as sns; sns.set_theme()

def _bt_date():
    return pd.datetime.today().date(), "-".join([str(pd.datetime.today().date().year - 5), str(pd.datetime.today().date().month), str(pd.datetime.today().date().day)])

def orig_long_trend_bt(df, b_df, st_date, ed_date, sigs= (20,50,150), transaction_cost=0, plot=False, index_name=None):

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=800)

    ax0s = []


    b_df = b_df[st_date:ed_date]
    ret_b = b_df.pct_change().fillna(0)

    port_ret = ret_b.rename('short').to_frame()

    port_ret['long'] = bt_tools.trend_trading(df,st_date, ed_date, sigs=sigs, transaction_cost=transaction_cost)

    port_ret = port_ret.fillna(0)


    ew_eq_curve = (port_ret + 1).cumprod()

    # print(port_ret['long'])

    ## Long/ Short Side Plot

    ax0s.append(ew_eq_curve['long'].plot(lw=0.75, ax=ax0, c='b'))
    ax0s.append(ew_eq_curve['short'].plot(lw=0.75, ax=ax0, c='r'))
    # plt.title(f"Long Short {index_name} Portfolio")


    ## combined PnL
    combined_PnL = (port_ret['long'] - port_ret['short'] + 1).cumprod()
    ax0s.append(combined_PnL.plot(lw=0.75, ax=ax1, c='g'))


    if plot:
        custom_lines = [Line2D([0], [0], color='b', lw=0.75),
                        Line2D([0], [0], color='r', lw=0.75),
                        Line2D([0], [0], color='g', lw=0.75)]

        fig.legend(custom_lines, ['long', 'short','combined'], ncol=len(custom_lines),
                   loc="upper center")
        plt.setp(ax0s, ylabel='Cumulative Return')

        plt.savefig(f"other_indices/{index_name}/pnls.png", bbox_inches='tight') #
    plt.close()

    return ew_eq_curve['long'], ew_eq_curve['short']

def continuous_sig_bt(df, b_df, combined_PnL, st_date, ed_date, sigs= (20,50,150), transaction_cost=0, plot=False):



    b_df = b_df[st_date:ed_date]
    ret_b = b_df.pct_change().fillna(0)

    ret_df = df[st_date:ed_date].pct_change()

    signal_fast, signal_slow = bt_tools.cont_trend_follow_sigs(df, sigs)

    holdings_fast = np.sign(signal_fast).replace(-1, 0)[st_date:]
    holdings_slow = np.sign(signal_slow).replace(-1, 0)[st_date:]

    w_fast = (1 / holdings_fast.sum(axis=1)).replace([np.inf, -np.inf], 0)
    w_slow = (1 / holdings_slow.sum(axis=1)).replace([np.inf, -np.inf], 0)

    arr_transaction_cost = [0, 0, 0, 0, transaction_cost] * (len(ret_b.index) // 5) + [0] * (len(ret_b.index) % 5)

    port_ret_fast = ((ret_df * holdings_fast.mul(w_fast, axis='index')).sum(axis=1) - ret_b - pd.Series(
        arr_transaction_cost, index=ret_b.index)).fillna(0)

    port_ret_slow = ((ret_df * holdings_slow.mul(w_slow, axis='index')).sum(axis=1) - ret_b - pd.Series(
        arr_transaction_cost, index=ret_b.index)).fillna(0)

    ew_eq_curve_fast = (port_ret_fast + 1).cumprod()
    ew_eq_curve_slow = (port_ret_slow + 1).cumprod()

    fast, slow = "_".join(list(map(str,sigs[:-1]))), "_".join(list(map(str,sigs[1:])))

    ew_eq_curve = ew_eq_curve_fast.rename(fast).to_frame()
    ew_eq_curve[slow] = ew_eq_curve_slow
    ew_eq_curve["original"] = combined_PnL

    ew_eq_curve.plot(figsize=(100, 30), legend=True, title="Portfolios PnLs of Seperated Signals")
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


    w_trading_rules = bt_tools.compute_MV_weights(sep_signal_ret.cov())

    combined_signal = norm_signal_fast * w_trading_rules[0] + norm_signal_slow * w_trading_rules[1]

    combined_ew_ret_0, combined_ew_eq_curve_0 = bt_tools.simple_ew_backtester(combined_signal, ret_df, ret_b, st_date
                                                                     , rules=0, transaction_cost=transaction_cost)

    combined_ew_ret_1, combined_ew_eq_curve_1 = bt_tools.simple_ew_backtester(combined_signal, ret_df, ret_b, st_date
                                                                     , rules=1, transaction_cost=transaction_cost)

    combined_ew_ret_2, combined_ew_eq_curve_2 = bt_tools.simple_ew_backtester(combined_signal, ret_df, ret_b, st_date
                                                                     , rules=2, transaction_cost=transaction_cost)

    combined_ew_eq_curve_0 = combined_ew_eq_curve_0.rename('0').to_frame()
    combined_ew_eq_curve_0['1'] = combined_ew_eq_curve_1
    combined_ew_eq_curve_0['2'] = combined_ew_eq_curve_2

    combined_ew_eq_curve_0['orig'] = combined_PnL
    combined_ew_eq_curve_0['benchmark'] = (1+ret_b).cumprod()

    combined_ew_eq_curve_0.ffill(inplace=True)


    ew_0 = bt_tools.port_stats(combined_ew_eq_curve_0['0'], combined_ew_eq_curve_0['benchmark'])
    ew_1 = bt_tools.port_stats(combined_ew_eq_curve_0['1'], combined_ew_eq_curve_0['benchmark'])
    ew_2 = bt_tools.port_stats(combined_ew_eq_curve_0['2'], combined_ew_eq_curve_0['benchmark'])
    orig = bt_tools.port_stats(combined_PnL, combined_ew_eq_curve_0['benchmark'])

    print(pd.DataFrame({'orig':orig, 'ew_0': ew_0, 'ew_1': ew_1, 'ew_2':ew_2}).to_latex())


def trend_follow(index, transaction_cost=0, sigs=(20,50,150), analysis=True, plt_save_path=None):

    '''index: sp500, nasdaq100 or stoxx600   '''


    end_bt_date="2020-09-13"
    start_bt_date_1yr_plus="-".join([str(pd.to_datetime(end_bt_date).year-6), str(pd.to_datetime(end_bt_date).month), str(pd.to_datetime(end_bt_date).day)])
    start_bt_date = "-".join([str(pd.to_datetime(end_bt_date).year-5), str(pd.to_datetime(end_bt_date).month), str(pd.to_datetime(end_bt_date).day)])



    df, b_df = getattr(bt_tools, f"get_df_{index}")(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)

    long_equity_curve, short_equity_curve= orig_long_trend_bt(df, b_df, st_date=start_bt_date, ed_date=end_bt_date,
                                                               sigs=sigs, plot=True, transaction_cost=transaction_cost, index_name=index)

    if index=='ftse250':
        continuous_sig_bt(df, b_df, long_equity_curve,
                          st_date=start_bt_date, ed_date=end_bt_date, sigs=sigs, plot=False, transaction_cost=transaction_cost)

    if analysis:
        if plt_save_path!=None:
            bt_tools.performance_analysis(long_equity_curve, short_equity_curve, port_name=f"{index}_{sigs}", plot=False, plt_save_path=plt_save_path)
        return bt_tools.port_stats(long_equity_curve, short_equity_curve)
    # continuous_sig_bt(df, b_df, orig_combined_PnL, sigs=sigs, plot=True, transaction_cost=transaction_cost)

    # return orig_combined_PnL


