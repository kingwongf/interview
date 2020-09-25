import bt_tools
import pandas as pd

ed_date = '2020-09-13'
st_date = '2015-09-13'
start_bt_date_1yr_plus = '2012-09-13'

df = pd.DataFrame({'A': [100,101,105,106,108], 'B':[101,102,106,107,105]}, index=range(0,5))
df.index.name = 'Date'
binary_signal = pd.DataFrame({'A': [0,1,0,1,1], 'B':[0,0,0,1,1]}, index=df.index)
ret = bt_tools.trend_trading(df, None, None, binary_signal =binary_signal)


print(ret)