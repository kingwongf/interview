import bt_tools
import pandas as pd
import matplotlib.pyplot as plt


ed_date = '2020-09-13'
st_date = '2015-09-13'
start_bt_date_1yr_plus = '2012-09-13'

# df, b_df = bt_tools.get_df_sp500(start_bt_date_1yr_plus, ed_date)
df, b_df = bt_tools.get_df_stoxx600(start_bt_date_1yr_plus, ed_date)
# df, b_df = bt_tools.get_df_ftse250(start_bt_date_1yr_plus, ed_date)


##  index returns
port_df = b_df.loc[st_date:ed_date].pct_change().fillna(0).rename('short').to_frame()



sigs = (30, 60, 160)
fast_MA, mid_MA, slow_MA = bt_tools.trend_follow_sigs(df,sigs)

entry = ((fast_MA > mid_MA) &(mid_MA > slow_MA)).astype(int)
exit = ((fast_MA < mid_MA) & (mid_MA> slow_MA)).astype(int)


port_df['long_trend'] = bt_tools.trend_trading_2(df, entry, exit, st_date, ed_date).fillna(0)



(port_df +1 ).cumprod().plot()
plt.show()