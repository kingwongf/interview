import bt_tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from matplotlib.lines import Line2D
import pickle
import seaborn as sns; sns.set_theme()
from itertools import permutations, combinations
from tqdm import tqdm

ed_date = '2010-01-01' # '2020-09-13'
st_date = '2007-01-01' # '2015-09-13'
start_bt_date_1yr_plus = '2005-01-01' # '2013-09-13'


index = 'sp500'
direction = 'short' ## long
rules_range = range(5,205, 5)


def opt_sigs(sigs, df, st_date= st_date, ed_date= ed_date, direction='long'):

    sharpe_r, end_pnl = bt_tools.perf_trend_trading(sigs, df, st_date= st_date, ed_date= ed_date, direction=direction)

    return (sigs, sharpe_r, end_pnl)





df, b_df = getattr(bt_tools, f"get_df_{index}")(start_bt_date_1yr_plus=start_bt_date_1yr_plus, end_bt_date=ed_date)



# futures = []
#
# for i,j, k in tqdm([sorted((i,j,k)) for i,j,k in list(combinations(range(5, 205,5),3))]):
#     futures.append(opt_sigs(sigs=(i,j,k), df=df, direction=direction))


## custom set of periods to try first

periods_combo = list(combinations((10,20,25,30,40,50,60,70,150,160,170, *(190,195,200, 205)), 3))
results = [opt_sigs(sigs=(i,j,k), df=df, direction=direction)
           for i,j, k in tqdm([sorted((i,j,k)) for i,j,k in periods_combo])] #205 list(combinations(range(5, 205 ,5),3))])

results = pd.DataFrame(results, columns=['sigs','sharpe','end_pnl'])
results.to_pickle(f"{direction}_trend_{index}_rules.pkl")


# def perf_trend_trading(sigs, direction='long'):
#     fast_MA, mid_MA, slow_MA = bt_tools.trend_follow_sigs(df, sigs)
#
#     entry_sig = ((fast_MA > mid_MA) & (mid_MA > slow_MA)).astype(int) if direction == 'long' else ((fast_MA < mid_MA) &(mid_MA < slow_MA)).astype(int)
#     exit_sig = ((fast_MA < mid_MA) & (mid_MA > slow_MA)).astype(int) if direction == 'long' else ((fast_MA > mid_MA) & (mid_MA <  slow_MA)).astype(int)
#     pos_df = bt_tools.get_position_df(df, entry_sig, exit_sig)
#
#     eq_risk_w = bt_tools.equal_risk_weighting(df, pos_df, st_date, ed_date)
#
#     port_ret = bt_tools.trend_trading_2(df, eq_risk_w, pos_df, st_date, ed_date).fillna(0)
#
#     port_ret = port_ret if direction == 'long' else -1 * port_ret
#
#     eq_curve = (1 + port_ret).cumprod()
#
#     sharpe = bt_tools.sharpe(eq_curve)
#
#     return sharpe, eq_curve[-1]

#
#
# success_signal ={}
# for i, j, k in tqdm([(i,j,k) for i,j,k in list(itertools.product(range(1,100), range(1,100), range(1,100))) if i<j<k]):
#     long_ret = bt_tools.trend_trading(df, st_date, ed_date, sigs=(i,j,k), signal_type='all')
#     cumret = (1+long_ret).cumprod()[-1]
#     if cumret >1.882489:
#         success_signal[(i,j,k)] = cumret
#
#
# print(success_signal)
#
