
# Download weather info from open-meteo (https://open-meteo.com/).
1. ecmwf.py   (ECMWF Weather Forecast API) 
2. gfs.py     (GFS & HRRR Forecast API)
3. weather.py (Weather Forecast API)

These Python scripts utilize the open-data APIs to gather and interpolate high-resolution weather data for specified
geographic coordinates. All scripts go through the following steps:
- Fetch Weather Data: Uses open-meteo APIs to retrieve temperature and precipitation data for past and future days.
- Data Interpolation: Implements inverse distance weighting for interpolating weather data from the nearest grid points to the actual coordinates.
- Logging and Output: Outputs logs to a text file and stores weather data into a CSV file dated with the current date.

You can customize the following parameters in the script according to your needs:
- latitude and longitude: Arrays of geographic coordinates for which you want to fetch and interpolate weather data.
- meteo_station: Array of station names which correspond to geographic coordinates.
- past_days and forecast_days: Number of past and future days to fetch data for.


# Download streamflow info from geoglows (https://data.geoglows.org/):
1. gglows_forecast.py
2. gglows_historical.py

This Python script utilizes the geoglows python package to access and process river discharge data, both historical
and forecasted. This is done through the following steps:
- Data Retrieval: Fetches both forecasted and historical river discharge data using river IDs (LINKNO).
- Data Parsing: Custom parsing function gglow_csv that formats the API data into a more readable and structured format.
- Logging and Output: Outputs logs to a text file and stores processed data into CSV files dated with the current date.

You can customize the following parameters in the script according to your needs:
- river_ids: List of river IDs for which the data is to be fetched.
- meteo_stations: List of station names corresponding to river ids.

