import bt_tools
import pandas as pd
import matplotlib.pyplot as plt

from risk_parity import risk_parity_weighting


ed_date = '2020-09-13'
st_date = '2015-09-13'
st_two_date = '2014-09-13'
start_bt_date_1yr_plus = '2012-09-13'

# df, b_df = bt_tools.get_df_sp500(start_bt_date_1yr_plus, ed_date)
df, b_df = bt_tools.get_df_stoxx600(start_bt_date_1yr_plus, ed_date)
# df, b_df = bt_tools.get_df_ftse250(start_bt_date_1yr_plus, ed_date)


##  index returns
port_df = b_df.loc[st_date:ed_date].pct_change().fillna(0).rename('short').to_frame()



sigs = (30, 60, 160)
fast_MA, mid_MA, slow_MA = bt_tools.trend_follow_sigs(df,sigs)

entry_long = ((fast_MA > mid_MA) &(mid_MA > slow_MA)).astype(int)
exit_long = ((fast_MA < mid_MA) & (mid_MA> slow_MA)).astype(int)
pos_df_long = bt_tools.get_position_df(df, entry_long, exit_long)


# eq_risk_w_long = bt_tools.equal_risk_weighting(df, pos_df_long, st_date, ed_date)
# eq_risk_w_long.to_pickle('eq_risk_w_long.pkl')

eq_risk_w_long = pd.read_pickle("eq_risk_w_long.pkl")

port_df['long_trend'] = bt_tools.trend_trading_2(df, eq_risk_w_long, pos_df_long, st_date, ed_date).fillna(0)


entry_short = ((fast_MA < mid_MA) &(mid_MA < slow_MA)).astype(int)
exit_short = ((fast_MA > mid_MA) & (mid_MA <  slow_MA)).astype(int)
pos_df_short = bt_tools.get_position_df(df, entry_short, exit_short)


# eq_risk_w_short = bt_tools.equal_risk_weighting(df, pos_df_short, st_date, ed_date)


eq_risk_w_short = pd.read_pickle('eq_risk_w_short.pkl')

port_df['short_trend'] = bt_tools.trend_trading_2(df, eq_risk_w_short, pos_df_short, st_date, ed_date).fillna(0)


# eq_risk_w_short.to_pickle('eq_risk_w_short.pkl')




## plotting old trend follow
(port_df +1 ).cumprod().plot(lw=0.75)
plt.show()
plt.close()


port_df['long_short_trend'] = port_df['long_trend'] - port_df['short_trend']
port_df['benchmark'] = port_df['long_trend'] - port_df['short']


(port_df[['long_short_trend','benchmark']] +1 ).cumprod().plot(lw=0.75)
plt.show()
plt.close()


