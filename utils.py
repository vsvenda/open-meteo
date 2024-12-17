import numpy as np
import pandas as pd
from datetime import datetime

def closest_quarters(point):
    """
    Calculate closest quarters (resolution 0.25)

    Args:
        point: examined point

    Returns:
         lower: nearest lower point
         upper: nearest upper point
    """
    # Calculate the nearest lower multiple of 0.25
    lower = float(np.floor(point * 4) / 4)
    # Calculate the nearest higher multiple of 0.25
    upper = float(np.ceil(point * 4) / 4)
    return [lower, upper]


def inverse_distance_weighting(x, y, points, values, power=2):
    """
    Perform Inverse Distance Weighting (IDW) interpolation.

    Args:
        x, y: The coordinates of the point to interpolate
        points: A list of tuples containing the coordinates of the known data points
        values: A list of values at the known data points
        power: The power parameter which controls how the weight decreases with distance

    Returns:
        Interpolated value at the point (x, y)
    """
    # Ensure values are NumPy arrays
    values = [np.array(v) for v in values]

    # Initialize numerator and denominator for IDW as arrays of zeros
    weighted_sum = np.zeros_like(values[0], dtype=np.float64)
    weight_sum = np.zeros_like(values[0], dtype=np.float64)

    # Compute weights and weighted values
    for (x0, y0), v in zip(points, values):
        distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

        if distance > 0:  # To avoid division by zero
            weight = 1 / distance ** power

            # Mask for valid (non-NaN) values
            valid_mask = ~np.isnan(v)

            # Add weighted values where valid (ignoring NaNs)
            weighted_sum[valid_mask] += weight * v[valid_mask]
            weight_sum[valid_mask] += weight
        else:
            # If the target point coincides with one of the data points
            return np.where(~np.isnan(v), v, np.nan)

    # Final result: weighted_sum / weight_sum where weight_sum > 0
    result = np.divide(weighted_sum, weight_sum, where=weight_sum > 0)

    # Assign NaN where no valid weights exist
    result[weight_sum == 0] = np.nan

    return result


def bilinear_interpolation(x, y, x1, x2, y1, y2, T11, T12, T21, T22):
    """
    Perform bilinear or linear interpolation for arrays.

    Args:
        x, y  : The longitude and latitude of the point to interpolate
        x1, x2: Longitudes of the data points
        y1, y2: Latitudes of the data points
        T11, T12, T21, T22: Arrays of temperatures at the points (x1, y1), (x1, y2), (x2, y1), and (x2, y2)

    Returns:
        Array of temperatures at the point (x, y)
    """
    # Ensure all temperature inputs are numpy arrays for element-wise operations
    T11, T12, T21, T22 = map(np.array, [T11, T12, T21, T22])

    if x1 == x2 and y1 == y2:
        # No interpolation needed if all points are the same
        T = T11
    elif x1 == x2:
        # Linear interpolation in y-direction only
        T = (T11 * (y2 - y) + T12 * (y - y1)) / (y2 - y1)
    elif y1 == y2:
        # Linear interpolation in x-direction only
        T = (T11 * (x2 - x) + T21 * (x - x1)) / (x2 - x1)
    else:
        # Bilinear interpolation
        T = ((T11 * (x2 - x) * (y2 - y) +
              T21 * (x - x1) * (y2 - y) +
              T12 * (x2 - x) * (y - y1) +
              T22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1)))
    return np.round(T, 2)


def gglow_csv(df, dictionary, csv_type):
    """
    Parse csv files created by geoglows API calls

    Args:
        df: original dataframe to be parsed
        dictionary: dict relating river_ids and meteo station names
        csv_type: data type (historical/forecast)

    Returns:
        Parsed dataframe
    """
    if csv_type == 'forecast':  # forecast data
        df = df.dropna(how='all')  # remove nan rows
        df.reset_index(inplace=True)
        df['river_id'] = df['river_id'].map(dictionary)  # replace river_id values with meteo-station
        df.rename(columns={'river_id': 'meteo-station'}, inplace=True)
        df.rename(columns={'time': 'date-time'}, inplace=True)
        df = df.sort_values(by=['meteo-station', 'date-time'], ascending=[True, True])
    elif csv_type == "historical":  # historical data
        df = df.dropna(how='all')  # remove nan rows
        new_columns = [dictionary[int(col)] if int(col) in dictionary else col for col in df.columns]
        df.columns = new_columns
        df.reset_index(inplace=True)
        df.rename(columns={'time': 'date-time'}, inplace=True)
    else:
        print("csv_type must be either forecast of historical")
    return df

def standardized_csv_files(file_path, prefix):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Convert date-time to pandas datetime object
    data['date-time'] = pd.to_datetime(data['date-time'], errors='coerce')
    data = data.dropna(subset=['date-time'])

    # Filter out rows where temperature or precipitation is not numeric
    data = data[data['temperature'].apply(lambda x: x.replace('.', '', 1).isdigit() if isinstance(x, str) else True)]
    data = data[data['precipitation'].apply(lambda x: x.replace('.', '', 1).isdigit() if isinstance(x, str) else True)]

    # Convert columns to numeric after filtering out non-numeric entries
    data['temperature'] = data['temperature'].astype(float)
    data['precipitation'] = data['precipitation'].astype(float)

    # Define the current date and filter for historical and forecast data
    today = pd.Timestamp.now().normalize()  # Current date at midnight
    start_historical = today - pd.Timedelta(days=2)
    end_forecast = today + pd.Timedelta(days=4)

    # Create a column for date starting from 07:00 AM
    data['date'] = (data['date-time'] - pd.Timedelta(hours=7)).dt.date

    # Separate the data into historical and forecast based on date
    historical_data = data[(data['date'] >= start_historical.date()) & (data['date'] < today.date())]
    forecast_data = data[(data['date'] >= today.date()) & (data['date'] <= end_forecast.date())]

    # Function to aggregate data for each meteo station
    def aggregate_data(df, agg_func):
        # Group by meteo-station and date, then apply the aggregation function
        return df.groupby(['meteo-station', 'date']).agg(agg_func).unstack(level=0)

    # Aggregate temperature (mean) and precipitation (sum) for historical and forecast data
    temperature_historical = aggregate_data(historical_data, {'temperature': 'mean'})['temperature']
    precipitation_historical = aggregate_data(historical_data, {'precipitation': 'sum'})['precipitation']
    temperature_forecast = aggregate_data(forecast_data, {'temperature': 'mean'})['temperature']
    precipitation_forecast = aggregate_data(forecast_data, {'precipitation': 'sum'})['precipitation']

    # Save each dataframe to a separate CSV file with the current date in the filename
    current_date = datetime.now().strftime('%Y-%m-%d')
    temperature_historical.to_csv(f'{prefix}_temp_hist_{current_date}.csv', index_label=None, encoding='utf-8-sig')
    precipitation_historical.to_csv(f'{prefix}_precip_hist_{current_date}.csv', index_label=None, encoding='utf-8-sig')
    temperature_forecast.to_csv(f'{prefix}_temp_forecast_{current_date}.csv', index_label=None, encoding='utf-8-sig')
    precipitation_forecast.to_csv(f'{prefix}_precip_forecast_{current_date}.csv', index_label=None, encoding='utf-8-sig')

    print("CSV files have been created successfully.")
