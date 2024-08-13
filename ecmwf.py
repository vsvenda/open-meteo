import numpy as np
import openmeteo_requests
import requests_cache
import pandas as pd
import sys
from retry_requests import retry
from datetime import datetime
from utils import closest_quarters, inverse_distance_weighting


# ----------------------------------------------------------------------------------------------------------------------
# ECMWF Weather Forecast API
# Global High Frequency Forecasts at 0.25° resolution
# The API utilizes open-data ECMWF weather forecasts from the IFS weather model,
# which has a resolution of 25 km and 3-hourly values.
# ----------------------------------------------------------------------------------------------------------------------

# Setup parameters for open-meteo forecast
latitude = [43.35, 42.83, 43.74, 43.27, 43.80, 43.52, 43.16, 43.16, 42.85, 43.04, 42.60, 42.84, 42.96,
            42.96, 42.73, 44.54, 44.76, 43.26, 44.09, 43.51, 44.44, 43.62, 43.93, 43.95]  # geographic coordinates
longitude = [19.36, 19.52, 19.71, 19.99, 19.30, 18.79, 18.85, 19.12, 19.88, 19.74, 19.94, 20.17, 19.58,
             19.10, 19.79, 19.23, 19.20, 18.61, 18.95, 18.45, 19.15, 19.37, 18.79, 19.57]
meteo_station = ["Pljevlja", "Kolašin", "Zlatibor", "Sjenica", "Višegrad", "Foča", "Plužine", "Žabljak",  # station names
                 "Berane", "Bijelo Polje", "Plav", "Rožaje", "Mojkovac", "Šavnik", "Andrijevica", "Loznica",
                 "Bijeljina", "Čemerno", "Han Pijesak", "Kalinovik", "Zvornik", "Rudo", "Sokolac", "Goražde"]
past_days = 2  # weather info for how many past days (possible values: 0, 1, 2, 3, 5, 7, 14, 31, 61, 92)
forecast_days = 7  # weather info for how many future days (possible values: 1, 3, 5, 7, 10, 15)

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
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": [closest_latitude[0], closest_latitude[1], closest_latitude[0], closest_latitude[1]],
        "longitude": [closest_longitude[0], closest_longitude[1], closest_longitude[1], closest_longitude[0]],
        "hourly": ["temperature_2m", "precipitation"],
        "past_days": past_days,
        "forecast_days": forecast_days
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
    hourly_dataframe.to_csv(csv_filename, mode='w', index=False, encoding='utf-8-sig', header=not i)

# Restore the default stdout and close the file
sys.stdout = sys.__stdout__
f.close()

