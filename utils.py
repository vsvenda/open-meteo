import numpy as np

def closest_quarters(n):
    # Calculate the nearest lower multiple of 0.25
    lower = float(np.floor(n * 4) / 4)
    # Calculate the nearest higher multiple of 0.25
    upper = float(np.ceil(n * 4) / 4)
    return [lower, upper]


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



