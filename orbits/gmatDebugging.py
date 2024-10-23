import numpy as np
import astropy.units as u
import sys
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.coordinates.solar_system import get_body_barycentric_posvel
sys.path.insert(1, 'tools')
import gmatTools
import frameConversion
import plot_tools
import unitConversion
from scipy.spatial.transform import Rotation as R

t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d  # + 1*u.yr/4
t_start = Time(57727, format='mjd', scale='utc')
C_I2G = frameConversion.inert2geo(t_start, t_veq)

# ~~~~~GET DATA~~~~~

# Get GMAT data
file_name = "gmatFiles/Moon_EMInert.txt"  # MJ2000Eq centered at EM barycenter
moon_inert, times = gmatTools.extract_pos(file_name)  # Array in km, time in MJD
moon_inert = np.array((moon_inert * u.km).to('AU'))  # Array in AU

# file_name = "gmatFiles/Moon_EMRot.txt"  # Rotating with the moon centered at EM barycenter
# moon_rot, _ = gmatTools.extract_pos(file_name)  # Array in km
# moon_rot = np.array((moon_rot * u.km).to('AU'))  # Array in AU
#
# file_name = "gmatFiles/Moon_GMEc.txt"  # GMEc centered at EM barycenter
# moon_gmec, _ = gmatTools.extract_pos(file_name)  # Array in km
# moon_gmec = np.array((moon_gmec * u.km).to('AU'))  # Array in AU

# Get Astropy moon data (I frame, AU)
C_I2G = frameConversion.inert2geo(t_start, t_veq)
astro_moon = np.zeros([len(times), 3])
for ii in np.arange(len(times)):
    _, _, astro_moon[ii, :] = frameConversion.getSunEarthMoon(times[ii], C_I2G)


# ~~~~~OBTAIN DCM DIFFERENCES~~~~~

# Resample positions to equal time intervals
def resample_positions(positions, times, new_times):
    resampled_positions = np.zeros((len(new_times), 3))
    for i in range(3):  # Interpolate for x, y, z separately
        resampled_positions[:, i] = np.interp(new_times.mjd, times.mjd, positions[:, i])
    return resampled_positions


# Calculate DCM between two time steps
def calculate_dcm(pos1, pos2):
    # Normalizing position vectors
    pos1_unit = pos1 / np.linalg.norm(pos1)
    pos2_unit = pos2 / np.linalg.norm(pos2)

    # Define rotation axis and angle
    axis = np.cross(pos1_unit, pos2_unit)
    if np.linalg.norm(axis) < 1e-8:  # Check for nearly parallel vectors
        return np.eye(3)  # Return identity if vectors are parallel
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(pos1_unit, pos2_unit))

    # Create DCM using scipy Rotation
    dcm = R.from_rotvec(angle * axis).as_matrix()
    return dcm


# Compare DCMs over time
def compare_dcms(positions, times, interval):
    # Create a new time array with equal intervals
    start_time = times[0]
    end_time = times[-1]
    delta_time = interval * u.s  # Interval in seconds

    new_times = Time(np.arange(start_time.mjd, end_time.mjd, delta_time.to(u.day).value), format='mjd')

    resampled_positions = resample_positions(positions, times, new_times)

    # Calculate a DCM between each position vector
    dcms = []
    for i in range(1, len(resampled_positions)):
        dcm = calculate_dcm(resampled_positions[i - 1], resampled_positions[i])
        dcms.append(dcm)

    # Compare DCMs (using Frobenius norm)
    differences = []
    for i in range(1, len(dcms)):
        diff = np.linalg.norm(dcms[i] - dcms[i - 1], ord='fro')  # Frobenius norm
        differences.append(diff)

    return differences, new_times[2:]


interval = 3600  # Time interval in seconds for a new, even time array (e.g., 1 hour)
gmat_differences, plot_times = compare_dcms(moon_inert, times, interval)
astro_differences, plot_times_astro = compare_dcms(astro_moon, times, interval)

breakpoint()


# ~~~~~PLOT~~~~~

plt.figure(figsize=(10, 6))
plt.plot(plot_times.value, gmat_differences, marker='.', linestyle='-', color='b', label="GMAT")
plt.plot(plot_times.value, astro_differences, marker='.', linestyle='-', color='r', label="Astropy")

plt.xlabel('Time [MJD]')
plt.ylabel('Frobenius Norm of DCM Differences')
plt.title('Differences in DCMs Over Time')
plt.grid(True)
plt.legend()
plt.show()

breakpoint()
