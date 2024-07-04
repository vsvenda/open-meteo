import geoglows
from utils import gglow_csv
from datetime import datetime

# Define river ids (LINKNO) and names
river_ids = [220252711, 220249952, 220212799, 220227955, 220232074,
             220267840, 220302223, 220284319, 220348963]
meteo_stations = ["Uvac", "Kokin Brod", "Bistrica", "Piva", "HC Prijepolj",
               "Potpeć", "Višegrad", "Bajina Bašta", "Zvornik"]
# Create river dictionary
river_dict = dict(zip(river_ids, meteo_stations))

# forecast
df_forecast = geoglows.data.forecast(river_id=river_ids)
df_forecast = gglow_csv(df_forecast, river_dict, "forecast")
csv_forecast = datetime.now().strftime("forecast_%Y-%m-%d.csv")
df_forecast.to_csv(csv_forecast, mode='a', index=False, encoding='utf-8-sig')

# forecast_ensembles
df_forecast_ensembles = geoglows.data.forecast_ensembles(river_id=river_ids)
df_forecast_ensembles = gglow_csv(df_forecast_ensembles, river_dict, "forecast")
csv_forecast_ensembles = datetime.now().strftime("forecast_ensembles_%Y-%m-%d.csv")
df_forecast_ensembles.to_csv(csv_forecast_ensembles, mode='a', index=False, encoding='utf-8-sig')

# retrospective
df_retrospective = geoglows.data.retrospective(river_id=river_ids)
df_retrospective = gglow_csv(df_retrospective, river_dict, "historical")
csv_retrospective = datetime.now().strftime("retrospective_%Y-%m-%d.csv")
df_retrospective.to_csv(csv_retrospective, mode='a', index=False, encoding='utf-8-sig')

# daily_averages
df_daily_averages = geoglows.data.daily_averages(river_id=river_ids)
df_daily_averages = gglow_csv(df_daily_averages, river_dict, "historical")
csv_daily_averages = datetime.now().strftime("daily_averages_%Y-%m-%d.csv")
df_daily_averages.to_csv(csv_daily_averages, mode='a', index=False, encoding='utf-8-sig')

