
import numpy as np
import astropy.units as u
from astropy.time import Time


def extract_pos(filepath):
    """Extracts position and time data from an exported GMAT file

    Args:
        filepath (str):
            Path to the file you want to extract position data from. File contains data of the form
            [x_position, y_position, z_position, time]

    Returns:
        pos (astropy Quantity array):
            Position array in km
        time (astropy Time array):
            Time vector in MJD

    """

    data = []
    with open(filepath) as file:
        next(file)
        for line in file:
            row = line.split()
            row = [float(x) for x in row]
            data.append(row)

    x = list(map(lambda x: x[0], data)) * u.km
    y = list(map(lambda x: x[1], data)) * u.km
    z = list(map(lambda x: x[2], data)) * u.km
    pos = np.array([x, y, z]).T
    time = Time(list(map(lambda x: x[3], data)), format='mjd', scale='utc')

    return pos, time
