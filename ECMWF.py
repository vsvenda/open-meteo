import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry


# Setup parameters for open-meteo forecast
latitude = [43.35, 42.83, 43.74, 43.27, 43.80, 43.52, 43.16, 43.16, 42.85, 43.04, 42.60, 42.84, 42.96,
            42.96, 42.73, 44.54, 44.76, 43.26, 44.09, 43.51, 44.44, 43.62, 43.93, 43.95]  # coordinates
longitude = [19.36, 19.52, 19.71, 19.99, 19.30, 18.79, 18.85, 19.12, 19.88, 19.74, 19.94, 20.17, 19.58,
             19.10, 19.79, 19.23, 19.20, 18.61, 18.95, 18.45, 19.15, 19.37, 18.79, 19.57]
meteo_station = ["Пљевља", "Колашин", "Златибор", "Сјеница", "Вишеград", "Фоча", "Плужине", "Жабљак",
                 "Беране", "Бијело Поље", "Плав", "Рожаје", "Мојковац", "Шавник", "Андријевица", "Лозница",
                 "Бијељина", "Чемерно", "Хан Пијесак", "Калиновик", "Зворник", "Рудо", "Соколац", "Горажде"]
past_days = 3  # weather info for how many past days
forecast_days = 7  # weather info for how many future days (forecast)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

for i in range(len(latitude)):
    # API call
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": latitude[i],
        "longitude": longitude[i],
        "hourly": ["temperature_2m", "precipitation"],
        "past_days": past_days,
        "forecast_days": forecast_days
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "temperature_2m": hourly_temperature_2m, "precipitation": hourly_precipitation}

    print(meteo_station[i])
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    print(hourly_dataframe)
