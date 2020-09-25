from trend_following_other_indices import trend_follow
from bt_tools import get_df_ftse250
import matplotlib.pyplot as plt
trend_follow('nasdaq100', transaction_cost=0)
# ftse, b_ftse = get_df_ftse250(None, None)
# ftse['BCPT.L'].plot()
# plt.show()