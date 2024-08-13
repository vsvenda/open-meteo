import geoglows
from utils import gglow_csv
from datetime import datetime
import sys

# ----------------------------------------------------------------------------------------------------------------------
# geoglows.data module
# The data module provides functions for requesting forecasted and historical data river discharge simulations.
# The data can be retrieved from the REST data service hosted by ECMWF, or it can be retrieved from the repository
# sponsored by the AWS Open Data Program.
# The speed and reliability of the AWS source is typically better than the REST service.

# Each function requires a river ID. The name for the ID varies based on the streams network dataset.
# It is called LINKNO in GEOGLOWS which uses the TDX-Hydro streams dataset.
# To find a LINKNO (river ID number), please refer to https://data.geoglows.org and browse the tutorials.
# ----------------------------------------------------------------------------------------------------------------------

# Define river ids (LINKNO) and names
river_ids = [220252711, 220249952, 220212799, 220227955, 220232074,
             220267840, 220302223, 220284319, 220348963]
meteo_stations = ["Uvac", "Kokin Brod", "Bistrica", "Piva", "HC Prijepolj",
                  "Potpeć", "Višegrad", "Bajina Bašta", "Zvornik"]
# Create river dictionary
river_dict = dict(zip(river_ids, meteo_stations))

# Create log file based on today's date and redirect print statements to it
today_date = datetime.now().strftime('gglows_forecast_%Y-%m-%d')
log_filename = f'{today_date}.txt'
f = open(log_filename, 'w', encoding='utf-8')
sys.stdout = f

# forecast
print("Launching geoglows.data.forecast.")
df_forecast = geoglows.data.forecast(river_id=river_ids)
df_forecast = gglow_csv(df_forecast, river_dict, "forecast")
print("\nWriting geoglows.data.forecast csv file.")
csv_forecast = datetime.now().strftime("forecast_%Y-%m-%d.csv")
df_forecast.to_csv(csv_forecast, mode='w', index=False, encoding='utf-8-sig')
print("\nFinished geoglows.data.forecast.")

# forecast_ensembles
print("\n\nLaunching geoglows.data.forecast_ensembles.")
df_forecast_ensembles = geoglows.data.forecast_ensembles(river_id=river_ids)
df_forecast_ensembles = gglow_csv(df_forecast_ensembles, river_dict, "forecast")
print("\nWriting geoglows.data.forecast_ensembles csv file.")
csv_forecast_ensembles = datetime.now().strftime("forecast_ensembles_%Y-%m-%d.csv")
df_forecast_ensembles.to_csv(csv_forecast_ensembles, mode='w', index=False, encoding='utf-8-sig')
print("\nFinished geoglows.data.forecast_ensembles.")

