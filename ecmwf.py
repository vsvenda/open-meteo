import os
import numpy as np
import openmeteo_requests
import requests_cache
import pandas as pd
import sys
from retry_requests import retry
from datetime import datetime
from utils import closest_quarters, inverse_distance_weighting, standardized_csv_files

def ecmwf(longitude, latitude, meteo_station, past_days, forecast_days):
    # ----------------------------------------------------------------------------------------------------------------------
    # ECMWF Weather Forecast API
    # Global High Frequency Forecasts at 0.25° resolution
    # The API utilizes open-data ECMWF weather forecasts from the IFS weather model,
    # which has a resolution of 25 km and 3-hourly values.
    # ----------------------------------------------------------------------------------------------------------------------


    # Create a filename with today's date to write results
    csv_filename = datetime.now().strftime("ecmwf_%Y-%m-%d.csv")

    # Create log file based on today's date and redirect print statements to it
    today_date = datetime.now().strftime('ecmwf_%Y-%m-%d')
    log_filename = f'{today_date}.txt'
    f = open(log_filename, 'w', encoding='utf-8')
    sys.stdout = f

    # Setup Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    for i in range(len(latitude)):
        # Get four closest coordinates (0.25 resolution)
        closest_latitude = closest_quarters(latitude[i])
        closest_longitude = closest_quarters(longitude[i])

        # API call
        url = "https://customer-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": [closest_latitude[0], closest_latitude[1], closest_latitude[0], closest_latitude[1]],
            "longitude": [closest_longitude[0], closest_longitude[1], closest_longitude[1], closest_longitude[0]],
            "hourly": ["temperature_2m", "precipitation"],
            "past_days": past_days,
            "forecast_days": forecast_days,
            "models": "ecmwf_ifs025",
            "apikey": "<INSERT API KEY HERE>"
        }
        responses = openmeteo.weather_api(url, params=params)

        # Print out meteo station info
        print(f"\nMeteo station {meteo_station[i]}")
        print(f"True coordinates {latitude[i]}°N {longitude[i]}°E")
        print(f"Coordinates of closest 4 points: [{responses[0].Latitude()}°N {responses[0].Longitude()}°N],"
              f"[{responses[1].Latitude()}°N {responses[1].Longitude()}°N],"
              f"[{responses[2].Latitude()}°N {responses[2].Longitude()}°N],"
              f"[{responses[3].Latitude()}°N {responses[3].Longitude()}°N]")

        # Interpolate weather from near points
        points = [(responses[0].Latitude(), responses[0].Longitude()), (responses[1].Latitude(), responses[1].Longitude()),
                  (responses[2].Latitude(), responses[2].Longitude()), (responses[3].Latitude(), responses[3].Longitude())]
        # Temperature
        values_temp = [responses[0].Hourly().Variables(0).ValuesAsNumpy(), responses[1].Hourly().Variables(0).ValuesAsNumpy(),
                       responses[2].Hourly().Variables(0).ValuesAsNumpy(), responses[3].Hourly().Variables(0).ValuesAsNumpy()]
        temp = inverse_distance_weighting(latitude[i], longitude[i], points, values_temp)
        temp = np.around(temp, decimals=2)  # round to 2 decimals
        # Precipitation
        values_prec = [responses[0].Hourly().Variables(1).ValuesAsNumpy(), responses[1].Hourly().Variables(1).ValuesAsNumpy(),
                       responses[2].Hourly().Variables(1).ValuesAsNumpy(), responses[3].Hourly().Variables(1).ValuesAsNumpy()]
        precipitation = inverse_distance_weighting(latitude[i], longitude[i], points, values_prec)
        precipitation = np.around(precipitation, decimals=2)  # round to 2 decimals

        # Create data frame from open-meteo results
        hourly_data = {"meteo-station": [meteo_station[i]]*len(temp),
                       "date-time": pd.date_range(
                        start=pd.to_datetime(responses[0].Hourly().Time(), unit="s", utc=True),
                        end=pd.to_datetime(responses[0].Hourly().TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=responses[0].Hourly().Interval()),
                        inclusive="left"),
                       "temperature": temp, "precipitation": precipitation}
        hourly_dataframe = pd.DataFrame(data=hourly_data)

        # Append to CSV, only include header in the first iteration
        hourly_dataframe.to_csv(csv_filename, mode='a', index=False, encoding='utf-8-sig', header=not i)

    print(precipitation, " nnn ", temp, " nnn ", values_prec)
    # Create standardized csv files for further use
    standardized_csv_files(csv_filename, 'ecmwf')
    os.remove(csv_filename)

    # Restore the default stdout and close the file
    sys.stdout = sys.__stdout__
    f.close()

