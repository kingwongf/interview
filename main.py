from bt_tools import get_df_ftse250, df_sanity_check, check_short_trend_trading
from datetime import date
from short_down_trend_follow import long_short_trend
from trend_following_other_indices import trend_follow
import itertools
from tqdm import tqdm
from stress_long_trend_follow import stress_trend_follow, structural_pred, coherent_evolve
import pandas as pd
import matplotlib.pyplot as plt
today = date.today()
start_bt_date_1yr_plus="-".join([str(today.year-6), str(today.month), str(today.day)])
end_bt_date=today

success_signal ={}
for i, j, k in tqdm(itertools.product(range(1,100), range(1,100), range(1,100))):
    if i!=j!=k:

        end_pnl = check_short_trend_trading((i,j,k))
        if end_pnl >1:
            success_signal[(i,j,k)] = end_pnl

print(success_signal)
# perf_short_trend = long_short_trend()
#
# print(perf_short_trend.to_latex())
# coherent_evolve()




# nasdaq100 = trend_follow('nasdaq100', plt_save_path="other_indices/nasdaq100/")
#
# stoxx600 = trend_follow('stoxx600', plt_save_path="other_indices/stoxx600/")
# ftse250 = trend_follow('ftse250', plt_save_path="other_indices/ftse250/")
# sp500 = trend_follow('sp500', plt_save_path="other_indices/sp500/")
#
# print(pd.DataFrame({'stoxx600': stoxx600, 'sp500': sp500, 'nasdaq100': nasdaq100, 'ftse250':ftse250}).to_latex())


exit()

# stress_trend_follow('Pos Deviations')
# stress_trend_follow('Neg Deviations')
# long_short_trend()
orig_df = trend_follow("stoxx600")



# print(orig_df)


# structural_pred()
# stoxx600 = trend_follow("stoxx600")
#
# sp500 = trend_follow("sp500")
#
# nasdaq100 = trend_follow("nasdaq100")
# ftse250 = trend_follow("ftse250")`


# print(pd.DataFrame({'stoxx600': stoxx600, 'sp500': sp500, 'nasdaq100': nasdaq100, 'ftse250':ftse250}).to_latex())

# stress_trend_follow('Pos Deviations')


# get_df_ftse250(start_bt_date_1yr_plus, end_bt_date)

short_df = long_short_trend()

# new_df = trend_follow("stoxx600", sigs=(30,60,160))
#


print(pd.DataFrame({'short_index': orig_df, 'short_trend': short_df}).to_latex())
#
#
# print(orig_df)
# print(new_df)
# print(pd.DataFrame({'20_50_150': orig_df, '30_60_160': new_df}).to_latex())



exit()
orig_combined_PnL = trend_follow('ftse250', transaction_cost=0, analysis=False)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(orig_combined_PnL.pct_change().sort_values(ascending=False))
    print(orig_combined_PnL)
exit()




ftse, b_ftse = get_df_ftse250(None, end_bt_date)
# ftse['PSH.L'].plot()
# plt.show()
print(ftse['PSH.L'].loc['2017-04-28':])
print(df_sanity_check(ftse, '2017-03-10'))


ret = ftse.pct_change()


ftse['AML.L'].plot()
plt.show()

print(ftse['ICGT.L'])
print(ftse['ICGT.L'].loc['2020-03-20':'2020-04-10'])

print(ftse['AML.L'].loc['2019-07-15':])


print(ftse['SDP.L'].loc['2020-07-25':])
print(ftse['IGG.L'].loc['2020-09-01':])



print(ftse['3IN.L'].loc['2014-09-28':'2014-10-05'])
print(ret['3IN.L'].loc['2014-09-28':'2014-10-05'])

print(ftse['WTAN.L'].loc['2020-09-10 ':])
print(ret['WTAN.L'].loc['2020-09-10 ':])


print(ftse['UKW.L'].loc['2020-09-01 ':])
print(ret['UKW.L'].loc['2020-09-01 ':])