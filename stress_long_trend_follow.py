from bt_tools import get_df_stoxx600, performance_analysis, trend_follow_sigs, df_sanity_check, port_stats, trend_trading, sharpe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
from matplotlib.lines import Line2D
from tqdm import tqdm
import itertools

def stress_trend_follow(direction='Pos Deviations', start_bt_date= '2007-01-01', end_bt_date= '2010-01-01'):
    dev_name = direction  ##
    dd_devs = {'Pos Deviations': range(0, 51, 1), 'Neg Deviations': range(0, -21, -1)}

    start_bt_date_1yr_plus = "-".join([str(int(start_bt_date.split('-')[0])- 1)] + start_bt_date.split('-')[1:])


    df, stoxx600 = get_df_stoxx600(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)
    transaction_cost = 0.0010

    ret_stoxx600 = stoxx600[start_bt_date:].pct_change().fillna(0)
    eq_curve_stoxx600 = (ret_stoxx600 + 1).cumprod()




    sigs = [20, 50, 150]



    deviations = dd_devs[dev_name]

    cmaps = sns.diverging_palette(250, 15, s=75, l=40,center="dark",n=len(deviations))
    fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,3.5), dpi=800)


    ax0s = []

    labels = []
    long_end_PnLs = []
    long_mean_Sharpe = []
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
        port_ret['index'] = ret_stoxx600
        port_ret = port_ret.fillna(0)

        ## Long/ Short Side Plot
        ew_eq_curve = (port_ret + 1).cumprod()

        plt_freq = 10 if direction=='Neg Deviations' else 30

        # print(f"new sigs {new_sigs}, {type(new_sigs[0])}")
        if i%plt_freq==0 or new_sigs==[30,60,160]:
            ax0s.append(ew_eq_curve['long'].plot(ax=ax0, lw=0.75, c = cmaps[i]))
            if new_sigs==[20,50, 150]:
                labels.append(f"{','.join(map(str,new_sigs))} (benchmark)")
            else:
                labels.append(",".join(map(str,new_sigs)))

            ## combined PnL
            combined_PnL = (port_ret['long'] - port_ret['index'] - arr_transaction_cost + 1).cumprod()
            combined_PnL.plot(ax=ax1, lw=0.70, c=cmaps[i])
        ## bar EndPnL
        long_end_PnLs.append((",".join(map(str,new_sigs)), ew_eq_curve['long'].tail(1).values[0]))
        long_mean_Sharpe.append((",".join(map(str,new_sigs)), sharpe(ew_eq_curve['long'])))






    # ax0.legend(loc='upper center', fontsize='small', ncol=len(deviations)//5)
    # ax1.legend(loc='upper center', fontsize='small', ncol=len(deviations)//5)

    b = eq_curve_stoxx600.plot(ax=ax0, lw=0.70, c='g', label='index')
    ax0s.append(b)
    labels.append("index")

    fig.legend(ax0s,     # The line objects
               labels=labels,   # The labels for each line
               loc="upper center",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               # fontsize='small',
               ncol=len(labels))

    plt.setp(ax0s, ylabel='Cumulative Return')
    # plt.show()
    plt.savefig(f"stress_trend_follow/pnl_{direction.lower().replace(' ','_')}.png", bbox_inches='tight')
    plt.close()


    ## Bar mean Sharpe ratio
    long_mean_Sharpe = pd.DataFrame(long_mean_Sharpe, columns=['sig', 'sharpe'])

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=800)
    ax.bar(long_mean_Sharpe['sig'], long_mean_Sharpe['sharpe'])
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel('avg. annual Sharpe')
    # plt.show()
    plt.savefig(f"stress_trend_follow/bar_mean_sharpe_{direction.lower().replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()


    ## Bar End PnL
    long_end_PnLs = pd.DataFrame(long_end_PnLs, columns=['sig','PnL'])

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=800)
    ax.bar(long_end_PnLs['sig'], long_end_PnLs['PnL'])
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel('End Cumulative Return')
    # plt.show()
    plt.savefig(f"stress_trend_follow/bar_end_{direction.lower().replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()




def structural_pred(start_bt_date= '2007-01-01', end_bt_date= '2010-01-01'):

    '''
    deviate from the optimal trading rule by using

    signal_{x} > signal_{x+1} > signal_{x+2}.


    :param sigs:
    :param start_bt_date:
    :param end_bt_date:
    :return:
    '''

    start_bt_date_1yr_plus = "-".join([str(int(start_bt_date.split('-')[0]) - 1)] + start_bt_date.split('-')[1:])


    df, stoxx600 = get_df_stoxx600(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=end_bt_date)
    transaction_cost = 0.0000

    ret_stoxx600 = stoxx600[start_bt_date:].pct_change().fillna(0)
    eq_curve_stoxx600 = (ret_stoxx600 + 1).cumprod()

    sigs=(30,60,160)

    closing_period = 100
    # deviations = list(zip(list(range(sigs[0]+2, closing_period+2)), list(range(sigs[1]+1, closing_period+1)), [sigs[2]] * closing_period))
    # deviations.insert(0,sigs)


    deviations = [(20,50,150), (30,59,158),(30,60,160),(32,61,160)] #, (30,58,157) ,(40,65,160)





    cmaps = sns.diverging_palette(220, 20, center="dark", n=len(deviations))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=800)
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(100, 50))

    ax0s = []

    labels = []
    long_end_PnLs = []

    plt_freq = 1
    for i, new_sigs in enumerate(deviations):

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

        # plt_freq = 10 if direction == 'Neg Deviations' else 30


        # print(f"new sigs {new_sigs}, {type(new_sigs[0])}")
        if i % plt_freq == 0 or new_sigs == [30, 60, 160]:
            ax0s.append(ew_eq_curve['long'].plot(ax=ax0, lw=0.75, c=cmaps[i]))
            if new_sigs == (20, 50, 150):
                labels.append(f"{','.join(map(str, new_sigs))} (benchmark)")
            else:
                labels.append(",".join(map(str, new_sigs)))

            ## combined PnL
            combined_PnL = (port_ret['long'] - port_ret['benchmark'] - arr_transaction_cost + 1).cumprod()
            combined_PnL.plot(ax=ax1, lw=0.70, c=cmaps[i])

        long_end_PnLs.append((",".join(map(str, new_sigs)), ew_eq_curve['long'].tail(1).values[0]))

    # ax0.legend(loc='upper center', fontsize='small', ncol=len(deviations)//5)
    # ax1.legend(loc='upper center', fontsize='small', ncol=len(deviations)//5)


    b = eq_curve_stoxx600.plot(ax=ax0, lw=0.70, c='g', label='index')
    ax0s.append(b)
    labels.append("index")



    fig.legend(ax0s,  # The line objects
               labels=labels,  # The labels for each line
               loc="upper center",  # Position of legend
               borderaxespad=0.1,  # Small spacing around legend box
               # fontsize='small',
               ncol=len(labels))

    plt.setp(ax0s, ylabel='Cumulative Return')
    # plt.show()
    plt.savefig(f"stress_trend_follow/pnl_struct_pred.png", bbox_inches='tight')
    plt.close()


def coherent_evolve(transaction_cost=0):
    bt_st_date = "2020-09-13"

    bt_date_range = ["-".join([str(pd.to_datetime(bt_st_date).year - i), str(pd.to_datetime(bt_st_date).month),
              str(pd.to_datetime(bt_st_date).day)]) for i in range(0,21,5)]

    fig, axs = plt.subplots(4, 2, figsize=(10, 14), dpi=800)
    cmaps = sns.diverging_palette(220, 20, center="dark", n=2)

    ax0s = []

    labels = []

    perfs = pd.DataFrame(columns=pd.MultiIndex.from_product([['2020','2015','2010','2005'], [(20,50,150),(30,60,160)]], names=['end_year','trading_rules']))




    for i, end_date in enumerate(bt_date_range[:-1]):

        st_date = bt_date_range[i + 1]
        start_bt_date_1yr_plus = "-".join([str(pd.to_datetime(st_date).year - 1), str(pd.to_datetime(st_date).month),
                                           str(pd.to_datetime(st_date).day)])

        df, b_df = get_df_stoxx600(start_bt_date_1yr_plus, end_date)


        b_df = b_df[st_date:end_date]
        ret_b = b_df.pct_change().fillna(0)


        port_ret = ret_b.rename('short').to_frame()

        ## index

        ew_eq_curve = (1+ret_b).cumprod().rename('short').to_frame()

        ax0s.append(ew_eq_curve['short'].plot(ax=axs[i, 0], lw=0.75, c='g', label='index'))


        for s,sigs in enumerate([(20,50,150),(30,60,160)]):



            port_ret['long'] = trend_trading(df, st_date, end_date, sigs=sigs, transaction_cost=0)

            port_ret = port_ret.fillna(0)




            ew_eq_curve = (port_ret + 1).cumprod()

            ax0s.append(ew_eq_curve['long'].plot(ax=axs[i,0], lw=0.75, c=cmaps[s]))


            ## combined PnL

            combined_PnL = (port_ret['long'] - port_ret['short'] + 1).cumprod()
            combined_PnL.plot(ax=axs[i,1], c=cmaps[s])


            perfs[(end_date[:4], sigs)] = port_stats(ew_eq_curve['long'], ew_eq_curve['short'])

            # print(port_ret['long'])


            # print(port_stats(ew_eq_curve['long'], ew_eq_curve['short']))





    custom_lines = [Line2D([0], [0], color=cmaps[0], lw=0.75),
                    Line2D([0], [0], color=cmaps[1], lw=0.75),
                    Line2D([0], [0], color='g', lw=0.75)]

    fig.legend(custom_lines, ['long (20,50,150)','long (30,60,160)', 'short index'], ncol=len(custom_lines), loc="upper center")
    plt.setp(ax0s, ylabel='Cumulative Return')
    # plt.show()
    plt.savefig(f"stress_trend_follow/coherent_evolve.png", bbox_inches='tight') #
    plt.close()
    # perfs = perfs.T




    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print(perfs.T.to_latex())

        r1 = perfs.iloc[:, perfs.columns.get_level_values(1) == (30,60,160)].droplevel('trading_rules', axis=1)
        r2 = perfs.iloc[:, perfs.columns.get_level_values(1) == (20,50,150)].droplevel('trading_rules', axis=1)

        print((r1-r2).sort_index(axis=1).to_latex())



