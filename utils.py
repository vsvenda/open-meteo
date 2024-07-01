import numpy as np

def closest_quarters(n):
    # Calculate the nearest lower multiple of 0.25
    lower = float(np.floor(n * 4) / 4)
    # Calculate the nearest higher multiple of 0.25
    upper = float(np.ceil(n * 4) / 4)
    return [lower, upper]


def inverse_distance_weighting(x, y, points, values, power=2):
    """
    Perform Inverse Distance Weighting (IDW) interpolation.
    Parameters:
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
        distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        if distance > 0:  # To avoid division by zero
            weight = 1 / distance**power
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

    Parameters:
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



