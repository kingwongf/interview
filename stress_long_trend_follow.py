from bt_tools import get_df, performance_analysis, trend_follow_sigs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()



start_bt_date_1yr_plus='2006-01-01'
start_bt_date = '2007-01-01'
end_bt_date='2010-01-01'


today = pd.datetime.today().date()

df = get_df(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)
transaction_cost = 0.0010


## Benchmark

stoxx600 = pd.read_pickle(f"stoxx600_{str(today)}.pkl")['Close'][start_bt_date:end_bt_date]
ret_stoxx600 = stoxx600.pct_change().fillna(0)
eq_curve_stoxx600 = (ret_stoxx600+ 1).cumprod()


sigs = [20, 50, 150]

neg_deviations = range(-20, 1, 1)


pos_deviations = range(0, 51, 1)

for deviations in (neg_deviations, pos_deviations):

    cmaps = sns.diverging_palette(250, 15, s=75, l=40,center="light",n=len(deviations))
    fig, (ax0,ax1) = plt.subplots(2,1, figsize=(50, 100), sharex=True)

    ax0s = []
    ax1s = []
    labels = []
    long_end_PnLs = []
    for i, plus_t in enumerate(deviations):
        new_sigs = list(map(lambda x: x+plus_t, sigs))
        temp_fast, temp_mid, temp_slow = trend_follow_sigs(df, new_sigs)

        long_signal = ((temp_fast > temp_mid) & (temp_mid > temp_slow)).astype(int)
        long_holdings = long_signal.shift(1)[start_bt_date:]

        ret_df = df[start_bt_date:].pct_change(1).fillna(0)

        arr_transaction_cost = [0, 0, 0, 0, transaction_cost] * (len(ret_df.index) // 5) + [0] * (len(ret_df.index) % 5)

        ## equal weighting
        long_w = (1 / long_holdings.sum(axis=1)).replace([np.inf, -np.inf], 0)

        port_ret = (long_holdings.mul(long_w, axis='index') * ret_df).sum(axis=1).rename('long').to_frame()
        port_ret['benchmark'] = ret_stoxx600
        port_ret = port_ret.fillna(0)

        ## Long/ Short Side Plot
        ew_eq_curve = (port_ret + 1).cumprod()
        ax0s.append(ew_eq_curve['long'].plot(ax=ax0, lw=0.7, c = cmaps[i]))
        labels.append(",".join(map(str,new_sigs)))

        long_end_PnLs.append((",".join(map(str,new_sigs)), ew_eq_curve['long'].tail(1).values[0]))

        # plt.show()
        # plt.close()

        ## combined PnL


        combined_PnL = (port_ret['long'] - port_ret['benchmark'] - arr_transaction_cost + 1).cumprod()

        combined_PnL.plot(ax=ax1, lw=0.7, c = cmaps[i])





        # combined_PnL.plot(figsize=(100, 20), legend=True)
        # plt.show()
        # plt.close()

    # ax0.legend(loc='upper center', fontsize='small', ncol=len(deviations)//5)
    # ax1.legend(loc='upper center', fontsize='small', ncol=len(deviations)//5)

    eq_curve_stoxx600.plot(ax=ax0, lw=1.5, c='g', label='benchmark')

    fig.legend(ax0s,     # The line objects
               labels=labels,   # The labels for each line
               loc="upper center",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               fontsize='small',
               ncol=len(deviations)//5)

    plt.subplots_adjust(left=0.03, bottom=0.04, right=0.99, top=0.90, wspace=0.08, hspace=0.01)

    # plt.show()

    plt.close()


    long_end_PnLs = pd.DataFrame(long_end_PnLs, columns=['sig','PnL'])
    print(long_end_PnLs.sort_values('PnL', ascending=False))
    ## for ascending freq, end PnL increases in general until (30, 60, 160) peak, then it starts to tank
    ## for descending freq, all freqs have negative returns, end PnL decreases in general until (11,41, 141), then starts to tank again



    plt.bar(long_end_PnLs['sig'], long_end_PnLs['PnL'])
    plt.xticks(rotation=90)
    plt.title('End PnL')
    plt.show()
    plt.close()