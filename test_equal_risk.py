import bt_tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from matplotlib.lines import Line2D
import seaborn as sns; sns.set_theme()
import swifter
from risk_parity import risk_parity_weighting


ed_date = '2020-09-13'
st_date = '2015-09-13'
start_bt_date_1yr_plus = '2012-09-13'

df, b_df = bt_tools.get_df_sp500(start_bt_date_1yr_plus, ed_date)
ret_df = df.pct_change()
cov_df = ret_df.rolling(252).cov()

w_EQ = cov_df.groupby(level=0, axis=0).apply(risk_parity_weighting).apply(pd.Series)

w_EQ.columns = ret_df.columns

print(w_EQ)