import pandas as pd
from bt_tools import get_df_ftse250
pd.options.plotting.backend = "plotly"

df, b_df = get_df_ftse250(None, None)
fig = df.plot(title="Pandas Backend Example", template="simple_white")
fig.update_yaxes(tickprefix="$")
fig.show()