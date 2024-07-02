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
import tools.unitConversion as unitConversion
import tools.frameConversion as frameConversion
import tools.orbitEOMProp as orbitEOMProp
import tools.plot_tools as plot_tools
import pdb

# ~~~~~PROPAGATE THE DYNAMICS~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de432s')

# Parameters
t_mjd = Time(57727, format='mjd', scale='utc')
days = 200
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in non dimensional units in rotating frame R [pos, vel]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]

# Convert the velocity to I frame from R frame (position is the same in both)
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Convert from I frame to H frame
pos_H, vel_H, Tp_dim = orbitEOMProp.convertIC_R2H(IC[0:3], vI, t_mjd, IC[-1], mu_star)

# Define the initial state array
state0 = np.append(np.append(pos_H.value, vel_H.value), days)   # Tp_dim.value

# Propagate the dynamics
statesFF, timesFF = orbitEOMProp.statePropFF(state0, t_mjd)
posFF = statesFF[:, 0:3]
velFF = statesFF[:, 3:6]

breakpoint()

# Preallocate space
r_PEM_r = np.zeros([len(timesFF), 3])
r_SunEM_r = np.zeros([len(timesFF), 3])
r_EarthEM_r = np.zeros([len(timesFF), 3])
r_MoonEM_r = np.zeros([len(timesFF), 3])

# Sim time in mjd
times_dim = unitConversion.convertTime_to_dim(timesFF)
timesFF_mjd = times_dim + t_mjd

# DCM for G frame and I frame
C_B2G = frameConversion.body2geo(t_mjd, t_mjd, mu_star)
C_G2B = C_B2G.T

for ii in np.arange(len(timesFF)):
    time = timesFF_mjd[ii]
    
    # Positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)
    r_EMO = EMO[0].get_xyz().to('AU').value
    
    # convert from H frame to GCRS frame
    r_PG = frameConversion.icrs2gcrs(posFF[ii]*u.AU, time)
    r_EMG = frameConversion.icrs2gcrs(r_EMO*u.AU, time)
    r_SunG = frameConversion.icrs2gcrs(r_SunO*u.AU, time)
    r_MoonG = frameConversion.icrs2gcrs(r_MoonO*u.AU, time)
    
    # change the origin to the EM barycenter, G frame
    r_PEM = r_PG - r_EMG
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG
    
    # convert from G frame to I frame
    r_PEM_r[ii, :] = C_G2B@r_PEM.to('AU')
    r_SunEM_r[ii, :] = C_G2B@r_SunEM.to('AU')
    r_EarthEM_r[ii, :] = C_G2B@r_EarthEM.to('AU')
    r_MoonEM_r[ii, :] = C_G2B@r_MoonEM.to('AU')


# ~~~~~PLOT FF SOLUTION AND GMAT FILE IN THE INERTIAL FRAME~~~~
# NEEDS FIXING

# Obtain FF data from GMAT
file_name = "gmatFiles/FF_ECNP.txt"
gmat = []
with open(file_name) as file:
    next(file)
    for line in file:
        row = line.split()
        row = [float(x) for x in row]
        gmat.append(row)

gmat_x_km = list(map(lambda x: x[0], gmat)) * u.km
gmat_y_km = list(map(lambda x: x[1], gmat)) * u.km
gmat_z_km = list(map(lambda x: x[2], gmat)) * u.km
gmat_time = Time(list(map(lambda x: x[3], gmat)), format='mjd', scale='utc')

# Convert to AU and put in a single matrix
gmat_xrot = gmat_x_km.to(u.AU)
gmat_yrot = gmat_y_km.to(u.AU)
gmat_zrot = gmat_z_km.to(u.AU)

gmat_posrot = np.array([gmat_xrot.value, gmat_yrot.value, gmat_zrot.value]).T

# Preallocate space
gmat_posinert = np.zeros([len(gmat_time), 3])

# Convert to I frame from R frame
for ii in np.arange(len(gmat_time)):
    gmat_posinert[ii, :] = frameConversion.rot2inertP(gmat_posrot[ii, :], gmat_time[ii], gmat_time[0])

# Plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_PEM_r[:, 0], r_PEM_r[:, 1], r_PEM_r[:, 2], color='blue', label='Propagated FF')
ax.plot(r_EarthEM_r[:, 0], r_EarthEM_r[:, 1], r_EarthEM_r[:, 2], color='green', label='Earth')
ax.plot(r_MoonEM_r[:, 0], r_MoonEM_r[:, 1], r_MoonEM_r[:, 2], color='gray', label='Moon')
ax.plot(r_SunEM_r[:, 0], r_SunEM_r[:, 1], r_SunEM_r[:, 2], color='orange', label='Sun')
ax.plot(gmat_posinert[:, 0], gmat_posinert[:, 1], gmat_posinert[:, 2], color='red', label='GMAT Orbit')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
ax.set_box_aspect([1.0, 1.0, 1.0])
plot_tools.set_axes_equal(ax)
plt.title('FF Model in the Inertial (I) Frame')
plt.legend()


# ~~~~~ANIMATION~~~~~

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Collect animation data for full force
N_FF = len(r_PEM_r[:, 0])  # Number of frames in animation
P_FF = 8  # Number of points plotted per frame

data_FF = np.array([r_PEM_r[:, 0], r_PEM_r[:, 1], r_PEM_r[:, 2]])
data_EarthFF = np.array([r_EarthEM_r[:, 0], r_EarthEM_r[:, 1], r_EarthEM_r[:, 2]])
data_MoonFF = np.array([r_MoonEM_r[:, 0], r_MoonEM_r[:, 1], r_MoonEM_r[:, 2]])
data_SunFF = np.array([r_SunEM_r[:, 0], r_SunEM_r[:, 1], r_SunEM_r[:, 2]])

line_FF, = ax.plot(data_FF[0, 0:1], data_FF[1, 0:1], data_FF[2, 0:1], color='blue', label='Orbit')
line_EarthFF, = ax.plot(data_EarthFF[0, 0:1], data_EarthFF[1, 0:1], data_EarthFF[2, 0:1], color='green', label='Earth')
line_MoonFF, = ax.plot(data_MoonFF[0, 0:1], data_MoonFF[1, 0:1], data_MoonFF[2, 0:1], color='gray', label='Moon')
line_SunFF, = ax.plot(data_SunFF[0, 0:1], data_SunFF[1, 0:1], data_SunFF[2, 0:1], color='orange', label='Sun')


def animate(i):
    line_FF.set_data(data_FF[0, :i*P_FF], data_FF[1, :i*P_FF])
    line_FF.set_3d_properties(data_FF[2, :i*P_FF])
    line_EarthFF.set_data(data_EarthFF[0, :i*P_FF], data_EarthFF[1, :i*P_FF])
    line_EarthFF.set_3d_properties(data_EarthFF[2, 0:i*P_FF])
    line_MoonFF.set_data(data_MoonFF[0, :i*P_FF], data_MoonFF[1, :i*P_FF])
    line_MoonFF.set_3d_properties(data_MoonFF[2, 0:i*P_FF])
    line_SunFF.set_data(data_SunFF[0, :i*P_FF], data_SunFF[1, :i*P_FF])
    line_SunFF.set_3d_properties(data_SunFF[2, 0:i*P_FF])


ani_FF = animation.FuncAnimation(fig, animate, frames=N_FF//P_FF, interval=1, repeat=False)

# Set axes limits
ax.set_xlim3d(min(data_FF[0]), max(data_FF[0]))
ax.set_ylim3d(min(data_FF[1]), max(data_FF[1]))
ax.set_zlim3d(min(data_FF[2]), max(data_FF[2]))
ax.set_box_aspect([1.0, 1.0, 1.0])
plot_tools.set_axes_equal(ax)

# Set labels
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
plt.legend()
plt.title('Full force model in the I frame')


plt.show()
