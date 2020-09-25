import bt_tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from matplotlib.lines import Line2D
import seaborn as sns; sns.set_theme()
import itertools
from tqdm import tqdm

ed_date = '2020-09-13'
st_date = '2015-09-13'
start_bt_date_1yr_plus = '2012-09-13'

df, b_df = bt_tools.get_df_sp500(start_bt_date_1yr_plus, ed_date)



success_signal ={}
for i, j, k in tqdm([(i,j,k) for i,j,k in list(itertools.product(range(1,100), range(1,100), range(1,100))) if i<j<k]):
    long_ret = bt_tools.trend_trading(df, st_date, ed_date, sigs=(i,j,k), signal_type='all')
    cumret = (1+long_ret).cumprod()[-1]
    if cumret >1.882489:
        success_signal[(i,j,k)] = cumret


print(success_signal)