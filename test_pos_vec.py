import bt_tools
import pandas as pd


df = pd.DataFrame({'entry':[1,0,0,1,0,0,0,1,1,0,0],
                   'exit': [0,0,1,0,1,1,1,0,0,1,1]},
                     index=[0,1,2,3,4,5,6,7,8,9,10])

price_df = pd.DataFrame({'A':[100,101,103,104,105,106,107,102, 100, 90, 80]},
                     index=[0,1,2,3,4,5,6,7,8,9,10])

price_df['ret'] = price_df.pct_change()
## expecting
# pos = (0, entry) (2, exit), (3, entry), (4,exit),  (7, entry), (9, exit)
print(df)

pos_df = pd.DataFrame(bt_tools.pos_idx_vec(df,None), columns=['date','signal'])

price_df['signal'] = pos_df.set_index('date')['signal']

price_df['pos'] = price_df['signal'].map({'entry':1, 'exit':0}).ffill().shift(1)


print(price_df)