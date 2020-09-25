import bt_tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from matplotlib.lines import Line2D
import seaborn as sns; sns.set_theme()


ed_date = '2020-09-13'
st_date = '2015-09-13'
start_bt_date_1yr_plus = '2012-09-13'

df, b_df = bt_tools.get_df_sp500(start_bt_date_1yr_plus, ed_date)


w = bt_tools.cap_w_sp500()


## plotting sp500 signals
fig, ax = plt.subplots(3,1, sharex=True, figsize=(50, 25))
b_df.loc[st_date:ed_date].plot(ax=ax[0])

bf, bm, bs = b_df.rolling(20).mean().loc[st_date:ed_date], b_df.rolling(30).mean().loc[st_date:ed_date], b_df.rolling(150).mean().loc[st_date:ed_date]
bf.plot(ax=ax[1],  label='20')
bm.plot(ax=ax[1], label='30')
# bs.plot(ax=ax[1], label='150')




sig = (bf> bm) # &(bm > bs)

ret_b = b_df.loc[st_date:ed_date].pct_change()

(ret_b +1).cumprod().plot(ax=ax[2], label='sp500')

(ret_b*(sig.shift(1)) +1).cumprod().plot(ax=ax[2], label='trend')


plt.legend(loc='upper left', prop={'size': 6})

plt.subplots_adjust(left=0.03, bottom=0.08, right=1, top=0.97, wspace=0.20, hspace=0.1)

plt.show()
plt.close()


sigs=(20,50,150)
transaction_cost=0

# fig, (ax0,ax1) = plt.subplots(2, 1, figsize=(5, 3.5), dpi=800)
fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5), dpi=800)


b_df = b_df[st_date:ed_date]
ret_b = b_df.pct_change().fillna(0)

port_ret = ret_b.rename('sp500').to_frame()

port_ret['trend_cap_w'] = bt_tools.trend_trading(df,st_date, ed_date, sigs=sigs, transaction_cost=transaction_cost,w=w, signal_type='all')
port_ret['trend_ew'] = bt_tools.trend_trading(df,st_date, ed_date, sigs=sigs, transaction_cost=transaction_cost,w=None, signal_type='all')

# port_ret['mom'] = bt_tools.momentum_trading(df,st_date, ed_date)


port_ret = port_ret.fillna(0)

port_ret['sp500_cap_w'] = df.pct_change().mul(w).sum(axis=1)


port_ret['sp500_ew'] = df.pct_change().mean(axis=1)


ew_eq_curve = (port_ret + 1).cumprod()

print(ew_eq_curve)


# print(port_ret['long'])

## Long/ Short Side Plot
# ew_eq_curve[['trend_cap_w', 'sp500_cap_w','sp500']].plot(lw=1.0, ax=ax0, legend=True)
ew_eq_curve[['trend_ew', 'sp500_ew','sp500']].plot(lw=0.75, ax=ax1, legend=True)


plt.ylabel('Cumulative Return')

# ax0.legend(loc='upper left', prop={'size': 6})
ax1.legend(loc='upper left', prop={'size': 6})
plt.savefig(f"other_indices/sp500/testing/pnls.png", bbox_inches='tight') #
plt.close()

