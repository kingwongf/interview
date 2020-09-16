from bt_tools import get_df_stoxx600, performance_analysis, trend_follow_sigs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme()





def long_short_trend(transaction_cost = 0.0000):
    today = pd.datetime.today().date()
    start_bt_date_1yr_plus="-".join([str(today.year-6), str(today.month), str(today.day)])
    end_bt_date=today

    start_bt_date = "-".join([str(today.year-5), str(today.month), str(today.day)])


    df, stoxx600 = get_df_stoxx600(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)




    ## Benchmark

    ret_stoxx600 = stoxx600.pct_change().fillna(0)


    sigs = [20, 50, 150]


    fast, mid, slow = trend_follow_sigs(df, sigs)



    long_signal = ((fast > mid)&(mid>slow)).astype(int)
    short_signal = ((fast < mid)&(mid < slow)).astype(int)


    long_holdings = long_signal.shift(1)[start_bt_date:]
    short_holdings = short_signal.shift(1)[start_bt_date:]



    ret_df = df[start_bt_date:].pct_change(1).fillna(0)

    arr_transaction_cost = [0,0,0,0,transaction_cost]*(len(ret_df.index)//5) + [0]*(len(ret_df.index)%5)



    long_w = long_holdings.sum(axis=1)
    long_w = (1/long_w).replace([np.inf, -np.inf], 0)

    short_w = short_holdings.sum(axis=1)
    short_w = (1/short_w).replace([np.inf, -np.inf], 0)


    port_ret = (long_holdings.mul(long_w, axis='index')*ret_df).sum(axis=1).rename('long').to_frame()


    port_ret['long'] = port_ret['long'] - arr_transaction_cost
    port_ret['short'] = (-1*short_holdings.mul(short_w, axis='index')*ret_df).sum(axis=1) - arr_transaction_cost

    port_ret['benchmark'] = ret_stoxx600


    port_ret = port_ret.fillna(0)

    ew_eq_curve = (port_ret+1).cumprod()

    ## Long/ Short Side Plot
    ew_eq_curve.plot(figsize=(100,30), legend=True) #.loc['2020-08-01':]
    plt.title('Long Short Sides')
    plt.show()
    plt.close()

    ## combined PnL
    combined_PnL = (port_ret['long'] + port_ret['short'] + 1).cumprod()
    combined_PnL.plot(figsize=(100,30), legend=True)
    plt.title('Combined PnL')
    plt.show()
    plt.close()

    ## Performance Analysis

    return performance_analysis(combined_PnL, ew_eq_curve['benchmark'], port_name='Long_Short_Trend', plot=True)
