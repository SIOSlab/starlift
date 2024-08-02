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
import pdb

# ~~~~~PROPAGATE THE DYNAMICS~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de432s')

# Parameters
t_mjd = Time(57727, format='mjd', scale='utc')
days = 100
days_can = unitConversion.convertTime_to_canonical(days * u.d)
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in non dimensional units in rotating frame R [pos, vel]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period

# Convert the velocity to I frame from R frame (position is the same in both)
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# DCM for G frame and I frame
C_I2G = frameConversion.inert2geo(t_mjd)
C_G2I = C_I2G.T

# Convert ICs to H frame (AU and AU/d) from I frame (canonical)
pos_H, vel_H = frameConversion.convertIC_I2H(IC[0:3], vI, t_mjd, C_I2G, Tp_can=None)

# Define the initial state array
state0 = np.append(np.append(pos_H.value, vel_H.value), days_can)

# Propagate the dynamics (states in AU, times in DU)
states, times = orbitEOMProp.statePropFF(state0, t_mjd)
pos = states[:, 0:3]
vel = states[:, 3:6]

# Sim time in mjd
times_dim = unitConversion.convertTime_to_dim(times)  # days
times_mjd = times_dim + t_mjd

# Preallocate space
r_PEM_r = np.zeros([len(times), 3])
r_SunEM_r = np.zeros([len(times), 3])
r_EarthEM_r = np.zeros([len(times), 3])
r_MoonEM_r = np.zeros([len(times), 3])

for ii in np.arange(len(times)):
    time = times_mjd[ii]
    
    # Positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU').value
    r_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)[0].get_xyz().to('AU').value
    
    # Convert from H frame (AU) to GMEc frame (km)
    r_PG = frameConversion.icrs2gmec(pos[ii]*u.AU, time)
    r_EMG = frameConversion.icrs2gmec(r_EMO*u.AU, time)
    r_SunG = frameConversion.icrs2gmec(r_SunO*u.AU, time)
    r_MoonG = frameConversion.icrs2gmec(r_MoonO*u.AU, time)

    # Change the origin to the EM barycenter, G frame (all km)
    r_PEM = r_PG - r_EMG
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG

    # Convert from G frame (in km) to I frame (in AU)
    r_PEM_r[ii, :] = C_G2I@r_PEM.to('AU')
    r_SunEM_r[ii, :] = C_G2I@r_SunEM.to('AU')
    r_EarthEM_r[ii, :] = C_G2I@r_EarthEM.to('AU')
    r_MoonEM_r[ii, :] = C_G2I@r_MoonEM.to('AU')


# ~~~~~PLOT FF SOLUTION AND GMAT FILE IN THE INERTIAL FRAME~~~~

# Obtain CRTBP data from GMAT
file_name = "gmatFiles/FF_rot.txt"
gmat_km, gmat_time = gmatTools.extract_pos(file_name)
gmat_posrot = np.array((gmat_km * u.km).to('AU'))

# Convert to I frame from R frame
gmat_posinert = np.zeros([len(gmat_time), 3])
for ii in np.arange(len(gmat_time)):
    C_I2R = frameConversion.inert2rot(gmat_time[ii], gmat_time[0])
    C_R2I = C_I2R.T
    gmat_posinert[ii, :] = C_R2I @ gmat_posrot[ii, :]

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


# # ~~~~~ANIMATION~~~~~ needs fixing, eventually
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# # Collect animation data for full force
# N_FF = len(r_PEM_r[:, 0])  # Number of frames in animation
# P_FF = 8  # Number of points plotted per frame
#
# data_FF = np.array([r_PEM_r[:, 0], r_PEM_r[:, 1], r_PEM_r[:, 2]])
# data_EarthFF = np.array([r_EarthEM_r[:, 0], r_EarthEM_r[:, 1], r_EarthEM_r[:, 2]])
# data_MoonFF = np.array([r_MoonEM_r[:, 0], r_MoonEM_r[:, 1], r_MoonEM_r[:, 2]])
# data_SunFF = np.array([r_SunEM_r[:, 0], r_SunEM_r[:, 1], r_SunEM_r[:, 2]])
#
# line_FF, = ax.plot(data_FF[0, 0:1], data_FF[1, 0:1], data_FF[2, 0:1], color='blue', label='Orbit')
# line_EarthFF, = ax.plot(data_EarthFF[0, 0:1], data_EarthFF[1, 0:1], data_EarthFF[2, 0:1], color='green', label='Earth')
# line_MoonFF, = ax.plot(data_MoonFF[0, 0:1], data_MoonFF[1, 0:1], data_MoonFF[2, 0:1], color='gray', label='Moon')
# line_SunFF, = ax.plot(data_SunFF[0, 0:1], data_SunFF[1, 0:1], data_SunFF[2, 0:1], color='orange', label='Sun')
#
#
# def animate(i):
#     # line_FF.set_data(data_FF[0, :i*P_FF], data_FF[1, :i*P_FF])
#     # line_FF.set_3d_properties(data_FF[2, :i*P_FF])
#     line_EarthFF.set_data(data_EarthFF[0, :i*P_FF], data_EarthFF[1, :i*P_FF])
#     line_EarthFF.set_3d_properties(data_EarthFF[2, 0:i*P_FF])
#     line_MoonFF.set_data(data_MoonFF[0, :i*P_FF], data_MoonFF[1, :i*P_FF])
#     line_MoonFF.set_3d_properties(data_MoonFF[2, 0:i*P_FF])
#     # line_SunFF.set_data(data_SunFF[0, :i*P_FF], data_SunFF[1, :i*P_FF])
#     # line_SunFF.set_3d_properties(data_SunFF[2, 0:i*P_FF])
#
#
# ani_FF = animation.FuncAnimation(fig, animate, frames=N_FF//P_FF, interval=1, repeat=True)
#
# # # Set axes limits
# # ax.set_xlim3d(min(data_FF[0]), max(data_FF[0]))
# # ax.set_ylim3d(min(data_FF[1]), max(data_FF[1]))
# # ax.set_zlim3d(min(data_FF[2]), max(data_FF[2]))
# # ax.set_box_aspect([1.0, 1.0, 1.0])
# # plot_tools.set_axes_equal(ax)
#
# # Set labels
# ax.set_xlabel('X [AU]')
# ax.set_ylabel('Y [AU]')
# ax.set_zlabel('Z [AU]')
# plt.legend()
# plt.title('Full force model in the I frame')


plt.show()
