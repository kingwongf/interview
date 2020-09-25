from bt_tools import get_df_stoxx600, performance_analysis, trend_follow_sigs, port_stats, trend_trading, get_df_sp500
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme()





def long_short_trend(transaction_cost = 0.0000, plot=False):
    start_bt_date_1yr_plus='2014-09-13'
    start_bt_date='2015-09-13'
    end_bt_date = '2020-09-13'
    df, stoxx600 = get_df_stoxx600(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)
    # df, stoxx600 = get_df_sp500(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)



    ## short index

    port_ret = stoxx600.loc[start_bt_date:end_bt_date].pct_change().rename('short_index').to_frame()

    ## sides return
    port_ret['long'] = trend_trading(df, start_bt_date, end_bt_date, sigs=(20,50,150), transaction_cost=0)
    port_ret['short_trend'] = trend_trading(df, start_bt_date, end_bt_date, sigs=(20,50,150), transaction_cost=0, direction='short')
    port_ret = port_ret.fillna(0)

    ## short trend side

    plt.rcParams["figure.dpi"] = 800

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(port_ret['short_trend'].sort_values(ascending=False))
    (-port_ret['short_trend'].fillna(0) + 1).cumprod().plot(figsize=(5, 3.5), legend=True, lw=0.75, fontsize=10).legend(loc=2)
    plt.ylabel('Cumulative Return')
    plt.savefig("short_trend_follow/short_side_pnl.png")
    plt.close()


    ## ew_eq_curves
    eq_eq_curves = (port_ret + 1).cumprod()


    plt.rcParams["figure.dpi"] = 800

    eq_eq_curves.plot(figsize=(5,3.5), legend=True, lw=0.75, fontsize=10).legend(loc=2)
    plt.ylabel('Cumulative Return')
    plt.savefig("short_trend_follow/pnls")
    plt.close()

    ## combined Pnl

    plt.rcParams["figure.dpi"] = 800
    combined_PnL = (port_ret['long'] - port_ret['short_trend']).rename('long_short_trend').to_frame()
    combined_PnL['benchmark'] = (port_ret['long'] - port_ret['short_index'])
    combined_PnL = (combined_PnL.fillna(0) + 1).cumprod()
    combined_PnL.plot(figsize=(5, 3.5), legend=True, lw=0.75, fontsize=10).legend(loc=2)
    plt.ylabel('Cumulative Return')
    plt.savefig("short_trend_follow/combined_pnl")
    plt.close()

    # eq_eq_curves['long'] = (port_ret['long'] +1).cumprod()
    # eq_eq_curves['short'] = (port_ret['short'] +1).cumprod()
    ## performance_analysis

    performance_analysis(eq_eq_curves['long'], eq_eq_curves['short_trend'])

    ## portfolio analysis

    return pd.DataFrame({'benchmark': port_stats(eq_eq_curves['long'], eq_eq_curves['short_index']),
    'long_short_trend': port_stats(eq_eq_curves['long'], eq_eq_curves['short_trend'])})
