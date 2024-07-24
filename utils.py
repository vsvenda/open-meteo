import numpy as np


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
    # Initialize numerator and denominator for IDW
    weighted_sum = 0
    weight_sum = 0
    # Compute weights and weighted values
    for (x0, y0), v in zip(points, values):
        distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        if distance > 0:  # To avoid division by zero
            weight = 1 / distance ** power
            weighted_sum += weight * v
            weight_sum += weight
        else:
            # If the target point coincides with one of the data points
            return v
    # Compute the final interpolated value
    if weight_sum > 0:
        return weighted_sum / weight_sum
    else:
        return None


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
