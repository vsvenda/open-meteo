import geoglows
from utils import gglow_csv
from datetime import datetime, timedelta
import sys
import pandas as pd

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

def gglows_historical(river_ids, meteo_stations):
    # Create river dictionary
    river_dict = dict(zip(river_ids, meteo_stations))

    # Create log file based on today's date and redirect print statements to it
    today_date = datetime.now().strftime('gglows_historical_%Y-%m-%d')
    log_filename = f'{today_date}.txt'
    f = open(log_filename, 'w', encoding='utf-8')
    sys.stdout = f

    # retrospective
    # print("\n\nLaunching geoglows.data.retrospective.")
    # df_retrospective = geoglows.data.retrospective(river_id=river_ids)
    # df_retrospective = gglow_csv(df_retrospective, river_dict, "historical")
    # print("\nWriting geoglows.data.retrospectives csv file.")
    # csv_retrospective = "retrospective.csv"
    # df_retrospective.to_csv(csv_retrospective, mode='w', index=False, encoding='utf-8-sig')
    # print("\nFinished geoglows.data.retrospectives.")

    # daily_averages
    print("\n\nLaunching geoglows.data.daily_averages.")
    df_daily_averages = geoglows.data.daily_averages(river_id=river_ids)
    df_daily_averages = gglow_csv(df_daily_averages, river_dict, "historical")
    print("\nWriting geoglows.data.daily_averages csv file.")
    csv_daily_averages = datetime.now().strftime("daily_averages_%Y-%m-%d.csv")
    df_daily_averages.to_csv(csv_daily_averages, mode='w', index=False, encoding='utf-8-sig', date_format='%m-%d')
    print("\nFinished geoglows.data.daily_averages.")

    # standardized csv file for ML models
    current_year = datetime.now().year
    df_daily_averages['date-time'] = pd.to_datetime(df_daily_averages['date-time'] + f'/{current_year}', format='%m/%d/%Y',
                                                    errors='coerce')
    df_daily_averages = df_daily_averages.dropna(subset=['date-time'])
    today = datetime.now()
    seven_days_ago = today - timedelta(days=7)
    standardized_data = df_daily_averages[(df_daily_averages['date-time'] >= pd.Timestamp(seven_days_ago.date())) &
                                          (df_daily_averages['date-time'] < pd.Timestamp(today.date()))]
    standardized_data.loc[:, 'date-time'] = standardized_data['date-time'].dt.date
    standardized_data = standardized_data.rename(columns={'date-time': 'date'})
    output_file = f'gglows_discharge_{today.strftime("%Y-%m-%d")}.csv'
    standardized_data.to_csv(output_file, index=False)
