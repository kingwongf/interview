from bt_tools import get_df_ftse250, df_sanity_check
from trend_following_other_indices import trend_follow
import pandas as pd
import matplotlib.pyplot as plt
today = pd.datetime.today().date()
start_bt_date_1yr_plus="-".join([str(today.year-6), str(today.month), str(today.day)])
end_bt_date=today

# get_df_ftse250(start_bt_date_1yr_plus, end_bt_date)


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