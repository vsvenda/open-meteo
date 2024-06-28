import geoglows
import pandas as pd

# example river segments from the Potpec
river_ids = [220249952]
# as dataframes
df = geoglows.data.forecast_stats(river_id=river_ids, date='20240627')
df = df[['flow_avg', ]].dropna()
df = df.loc[df.index.get_level_values('time') < df.index.get_level_values('time')[0] + pd.Timedelta(days=1)]
print(df)
