import numpy as np
import os.path
import pickle
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
import plot_tools
import gmatTools
#import tools.unitConversion as unitConversion
#import tools.frameConversion as frameConversion
#import tools.orbitEOMProp as orbitEOMProp
#import tools.plot_tools as plot_tools
import pdb


# ~~~~~PROPAGATE THE DYNAMICS~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de440')

# Parameters
t_mjd = Time(57727, format='mjd', scale='utc')
days = 30
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star
moon_r_can = 1-mu_star
moon_r = (unitConversion.convertPos_to_dim(moon_r_can)).to('AU')
earth_r_can = mu_star
earth_r = (unitConversion.convertPos_to_dim(earth_r_can)).to('AU')

# Initial condition in non dimensional units in rotating frame R [pos, vel, T/2]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2
# IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]  # DRO

# Convert the velocity to inertial from R
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar_CRTBP = np.array([IC[0], IC[2], vI[1], days])

# Propagate the dynamics in the CRTBP model
statesCRTBP, timesCRTBP = orbitEOMProp.statePropCRTBP(freeVar_CRTBP, mu_star)
posCRTBP = statesCRTBP[:, 0:3]
velCRTBP = statesCRTBP[:, 3:6]

# Preallocate space
r_PEM_r = np.zeros([len(timesCRTBP), 3])
# r_EarthEM_r = np.zeros([len(timesCRTBP), 3])
# r_MoonEM_r = np.zeros([len(timesCRTBP), 3])

# Sim time in mjd
timesCRTBP_mjd = Time(timesCRTBP + t_mjd.value, format='mjd', scale='utc')
timesCRTBP_can = unitConversion.convertTime_to_canonical(timesCRTBP_mjd.value * u.d)
#
# # DCM for G frame and I frame
# C_B2G = frameConversion.body2geo(t_mjd, t_mjd, mu_star)
# C_G2B = C_B2G.T

for ii in np.arange(len(timesCRTBP)):
    # time = timesCRTBP_mjd[ii]
    #
    # # Positions of the Moon and EM barycenter relative SS barycenter in H frame
    # r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU').value
    # EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)
    # r_EMO = EMO[0].get_xyz().to('AU').value
    #
    # # Convert from H frame to GCRS frame
    # r_EMG = frameConversion.icrs2gcrs(r_EMO*u.AU, time)
    # r_MoonG = frameConversion.icrs2gcrs(r_MoonO*u.AU, time)
    #
    # # Change the origin to the EM barycenter, G frame
    # r_EarthEM = -r_EMG
    # r_MoonEM = r_MoonG - r_EMG
    #
    # # Convert from G frame to I frame
    # C_B2G = frameConversion.body2geo(time, t_mjd, mu_star)
    # C_G2B = C_B2G.T
    # r_EarthEM_r[ii, :] = C_G2B@r_EarthEM.to('AU')
    # r_MoonEM_r[ii, :] = C_G2B@r_MoonEM.to('AU')

    # Convert to AU
    r_PEM_r[ii, :] = (unitConversion.convertPos_to_dim(posCRTBP[ii, :])).to('AU')


# ~~~~~PLOT SOLUTION AND GMAT FILE IN THE INERTIAL FRAME~~~~

# Obtain CRTBP data from GMAT
file_name = "gmatFiles/CRTBP_rot.txt"
gmat_km, gmat_time = gmatTools.extract_pos(file_name)
gmat_posrot = np.array((gmat_km * u.km).to('AU'))

# Convert to I frame from R frame
gmat_posinert = np.zeros([len(gmat_time), 3])
for ii in np.arange(len(gmat_time)):
    gmat_posinert[ii, :] = frameConversion.rot2inertP(gmat_posrot[ii, :], gmat_time[ii], gmat_time[0])

# Plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_PEM_r[:, 0], r_PEM_r[:, 1], r_PEM_r[:, 2], color='blue', label='Propagated CRTBP')
ax.plot(moon_r*np.cos(timesCRTBP_can), moon_r*np.sin(timesCRTBP_can), 0, color='gray', label='Moon')
ax.plot(earth_r*np.cos(timesCRTBP_can), earth_r*np.sin(timesCRTBP_can), 0, color='green', label='Earth')
# ax.plot(gmat_posinert[:, 0], gmat_posinert[:, 1], gmat_posinert[:, 2], color='red', label='GMAT Orbit')

# ax.plot(r_EarthEM_r[:, 0], r_EarthEM_r[:, 1], r_EarthEM_r[:, 2], color='green', label='Earth')
# ax.plot(r_MoonEM_r[:, 0], r_MoonEM_r[:, 1], r_MoonEM_r[:, 2], color='gray', label='Moon')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
ax.set_box_aspect([1.0, 1.0, 1.0])
plot_tools.set_axes_equal(ax)
plt.title('CRTBP in the Inertial (I) Frame')
plt.legend()

# ax2d = plt.figure().add_subplot()
# ax2d.plot(r_PEM_r[:, 0], r_PEM_r[:, 1], color='blue', label='Propagated CRTBP')
# ax2d.plot(earth_r*np.cos(timesCRTBP_can), earth_r*np.sin(timesCRTBP_can), color='green', label='Earth')
# ax2d.plot(moon_r*np.cos(timesCRTBP_can), moon_r*np.sin(timesCRTBP_can), color='gray', label='Moon')


# ~~~~~ANIMATION~~~~~

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Collect animation data for CRTBP
N = len(r_PEM_r[:, 0])  # number of frames in animation
P = 50  # number of points plotted per frame

data_CRTBP = np.array([r_PEM_r[:, 0], r_PEM_r[:, 1], r_PEM_r[:, 2]])
data_Earth = np.array([earth_r*np.cos(timesCRTBP_can), earth_r*np.sin(timesCRTBP_can), np.zeros(len(timesCRTBP_can))])
data_Moon = np.array([moon_r*np.cos(timesCRTBP_can), moon_r*np.sin(timesCRTBP_can), np.zeros(len(timesCRTBP_can))])

# Initialize the first point for each body
line_CRTBP, = ax.plot(data_CRTBP[0, 0:1], data_CRTBP[1, 0:1], data_CRTBP[2, 0:1], color='blue', label='Orbit')
line_Earth, = ax.plot(data_Earth[0, 0:1], data_Earth[1, 0:1], data_Earth[2, 0:1], color='green', label='Earth')
line_Moon, = ax.plot(data_Moon[0, 0:1], data_Moon[1, 0:1], data_Moon[2, 0:1], color='gray', label='Moon')


def animate(i):
    line_CRTBP.set_data(data_CRTBP[0, :i*P], data_CRTBP[1, :i*P])  # Set the x and y positions
    line_CRTBP.set_3d_properties(data_CRTBP[2, :i*P])  # Set the z position
    line_Earth.set_data(data_Earth[0, :i*P], data_Earth[1, :i*P])
    line_Earth.set_3d_properties(data_Earth[2, :i * P])
    line_Moon.set_data(data_Moon[0, :i*P], data_Moon[1, :i*P])
    line_Moon.set_3d_properties(data_Moon[2, :i * P])


ani_CRTBP = animation.FuncAnimation(fig, animate, frames=N//P, interval=1, repeat=True)

# Set axes limits
ax.set_xlim3d(min(data_CRTBP[0]), max(data_CRTBP[0]))
ax.set_ylim3d(min(data_CRTBP[1]), max(data_CRTBP[1]))
if IC[2] != 0:
    ax.set_zlim3d(min(data_CRTBP[2]), max(data_CRTBP[2]))
ax.set_box_aspect([1.0, 1.0, 1.0])
plot_tools.set_axes_equal(ax)

# Set labels
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
plt.legend()
plt.title('CRTBP model in the I frame')


plt.show()

breakpoint()
