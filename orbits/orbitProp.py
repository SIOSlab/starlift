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
#sys.path.insert(1, 'tools')
#import unitConversion
#import frameConversion
#import orbitEOMProp
#import plot_tools
import tools.unitConversion as unitConversion
import tools.frameConversion as frameConversion
import tools.orbitEOMProp as orbitEOMProp
import tools.plot_tools as plot_tools
import pdb

# ~~~~~PROPAGATE THE DYNAMICS  ~~~~~

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

# Convert the velocity to I frame from R frame
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar_CRTBP = np.array([IC[0], IC[2], vI[1], days])

# Propagate the dynamics in the CRTBP model
statesCRTBP, timesCRTBP = orbitEOMProp.statePropCRTBP(freeVar_CRTBP, mu_star)
posCRTBP = statesCRTBP[:, 0:3]
velCRTBP = statesCRTBP[:, 3:6]

# Preallocate space
r_PEM_CRTBP = np.zeros([len(timesCRTBP), 3])
r_EarthEM_CRTBP = np.zeros([len(timesCRTBP), 3])
r_MoonEM_CRTBP = np.zeros([len(timesCRTBP), 3])

# sim time in mjd
timesCRTBP_mjd = timesCRTBP + t_mjd

# DCM for G frame and I frame
C_B2G = frameConversion.body2geo(t_mjd, t_mjd, mu_star)
C_G2B = C_B2G.T

# # Obtain Moon and Earth positions for CRTBP
# for ii in np.arange(len(timesCRTBP)):
#     time = timesCRTBP_mjd[ii]
#
#     # Positions of the Moon and EM barycenter relative SS barycenter in H frame
#     r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU').value
#     EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)
#     r_EMO = EMO[0].get_xyz().to('AU').value
#
#     # Convert from H frame to GCRS frame
#     r_EMG = frameConversion.icrs2gcrs(r_EMO * u.AU, time)
#     r_MoonG = frameConversion.icrs2gcrs(r_MoonO * u.AU, time)
#
#     # Change the origin to the EM barycenter, G frame
#     r_EarthEM = -r_EMG
#     r_MoonEM = r_MoonG - r_EMG
#
#     # Convert from G frame to I frame
#     r_EarthEM_CRTBP[ii, :] = C_G2B @ r_EarthEM.to('AU')
#     r_MoonEM_CRTBP[ii, :] = C_G2B @ r_MoonEM.to('AU')
#
#     # Convert to AU
#     r_PEM_CRTBP[ii, :] = (unitConversion.convertPos_to_dim(posCRTBP[ii, :])).to('AU')


# Convert position from I frame to H frame [AU]
pos_H, vel_H, Tp_dim = orbitEOMProp.convertIC_R2H(posCRTBP[0], velCRTBP[0], t_mjd, timesCRTBP[-1], mu_star)

# Define the initial state array
state0 = np.append(np.append(pos_H.value, vel_H.value), days)   # Tp_dim.value

# Propagate the dynamics in the full force model (H frame) [AU]
statesFF, timesFF = orbitEOMProp.statePropFF(state0, t_mjd)
posFF = statesFF[:, 0:3]
velFF = statesFF[:, 3:6]

# Preallocate space
r_PEM_r = np.zeros([len(timesFF), 3])
r_SunEM_r = np.zeros([len(timesFF), 3])
r_EarthEM_r = np.zeros([len(timesFF), 3])
r_MoonEM_r = np.zeros([len(timesFF), 3])

# sim time in mjd
timesFF_mjd = timesFF + t_mjd

# Obtain Moon, Earth, and Sun positions for FF
for ii in np.arange(len(timesFF)):
    time = timesFF_mjd[ii]

    # Positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)
    r_EMO = EMO[0].get_xyz().to('AU').value

    # Convert from H frame to GCRS frame
    r_PG = frameConversion.icrs2gcrs(posFF[ii]*u.AU, time)
    r_EMG = frameConversion.icrs2gcrs(r_EMO*u.AU, time)
    r_SunG = frameConversion.icrs2gcrs(r_SunO*u.AU, time)
    r_MoonG = frameConversion.icrs2gcrs(r_MoonO*u.AU, time)

    # Change the origin to the EM barycenter, G frame
    r_PEM = r_PG - r_EMG
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG

    # Convert from G frame to I frame
    r_PEM_r[ii, :] = C_G2B@r_PEM.to('AU')
    r_SunEM_r[ii, :] = C_G2B@r_SunEM.to('AU')
    r_EarthEM_r[ii, :] = C_G2B@r_EarthEM.to('AU')
    r_MoonEM_r[ii, :] = C_G2B@r_MoonEM.to('AU')


# # ~~~~~PLOT CRTBP SOLUTION AND GMAT FILE IN THE INERTIAL FRAME~~~~
#
# # Obtain CRTBP data from GMAT
# file_name = "gmatFiles/CRTBP_ECEP.txt"
# gmat_CRTBP = []
# with open(file_name) as file:
#     next(file)
#     for line in file:
#         row = line.split()
#         row = [float(x) for x in row]
#         gmat_CRTBP.append(row)
#
# gmat_x_km = list(map(lambda x: x[0], gmat_CRTBP)) * u.km
# gmat_y_km = list(map(lambda x: x[1], gmat_CRTBP)) * u.km
# gmat_z_km = list(map(lambda x: x[2], gmat_CRTBP)) * u.km
# gmat_time = Time(list(map(lambda x: x[3], gmat_CRTBP)), format='mjd', scale='utc')
#
# # Convert to AU and put in a single matrix
# gmat_xrot = gmat_x_km.to(u.AU)
# gmat_yrot = gmat_y_km.to(u.AU)
# gmat_zrot = gmat_z_km.to(u.AU)
# gmat_posrot = np.array([gmat_xrot.value, gmat_yrot.value, gmat_zrot.value]).T
#
# # Preallocate space
# gmat_posinert = np.zeros([len(gmat_time), 3])
#
# # Convert to I frame from R frame
# for ii in np.arange(len(gmat_time)):
#     gmat_posinert[ii, :] = frameConversion.rot2inertP(gmat_posrot[ii, :], gmat_time[ii], t_mjd)
#
# # Plot
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(r_PEM_CRTBP[:, 0], r_PEM_CRTBP[:, 1], r_PEM_CRTBP[:, 2], color='blue', label='Propagated CRTBP')
# ax.plot(r_EarthEM_CRTBP[:, 0], r_EarthEM_CRTBP[:, 1], r_EarthEM_CRTBP[:, 2], color='green', label='Earth')
# ax.plot(r_MoonEM_CRTBP[:, 0], r_MoonEM_CRTBP[:, 1], r_MoonEM_CRTBP[:, 2], color='gray', label='Moon')
# ax.plot(gmat_posinert[:, 0], gmat_posinert[:, 1], gmat_posinert[:, 2], color='red', label='GMAT Orbit')
# ax.set_xlabel('X [AU]')
# ax.set_ylabel('Y [AU]')
# ax.set_zlabel('Z [AU]')
# plt.title('CRTBP in the Inertial (I) Frame')
# plt.legend()
# plt.show()


# ~~~~~PLOT FF SOLUTION AND GMAT FILE IN THE INERTIAL FRAME~~~~
# NEEDS FIXING

# Obtain FF data from GMAT
file_name = "gmatFiles/FF_ECNP.txt"
gmat_FF = []
with open(file_name) as file:
    next(file)
    for line in file:
        row = line.split()
        row = [float(x) for x in row]
        gmat_FF.append(row)

gmat_x_kmFF = list(map(lambda x: x[0], gmat_FF)) * u.km
gmat_y_kmFF = list(map(lambda x: x[1], gmat_FF)) * u.km
gmat_z_kmFF = list(map(lambda x: x[2], gmat_FF)) * u.km
gmat_timeFF = Time(list(map(lambda x: x[3], gmat_FF)), format='mjd', scale='utc')

# Convert to AU and put in a single matrix
gmat_xrotFF = gmat_x_kmFF.to(u.AU)
gmat_yrotFF = gmat_y_kmFF.to(u.AU)
gmat_zrotFF = gmat_z_kmFF.to(u.AU)
gmat_posrotFF = np.array([gmat_xrotFF.value, gmat_yrotFF.value, gmat_zrotFF.value]).T

# Preallocate space
gmat_posinertFF = np.zeros([len(gmat_timeFF), 3])

# Convert to I frame from R frame
for ii in np.arange(len(gmat_timeFF)):
    gmat_posinertFF[ii, :] = frameConversion.rot2inertP(gmat_posrotFF[ii, :], gmat_timeFF[ii], t_mjd)

# Plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_PEM_r[:, 0], r_PEM_r[:, 1], r_PEM_r[:, 2], color='blue', label='Propagated FF')
ax.plot(r_EarthEM_r[:, 0], r_EarthEM_r[:, 1], r_EarthEM_r[:, 2], color='green', label='Earth')
ax.plot(r_MoonEM_r[:, 0], r_MoonEM_r[:, 1], r_MoonEM_r[:, 2], color='gray', label='Moon')
ax.plot(r_SunEM_r[:, 0], r_SunEM_r[:, 1], r_SunEM_r[:, 2], color='orange', label='Sun')
ax.plot(gmat_posinertFF[:, 0], gmat_posinertFF[:, 1], gmat_posinertFF[:, 2], color='red', label='GMAT Orbit')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
ax.set_xlim3d(min(r_PEM_r[:, 0]), max(r_PEM_r[:, 0]))
ax.set_ylim3d(min(r_PEM_r[:, 1]), max(r_PEM_r[:, 1]))
ax.set_zlim3d(min(r_PEM_r[:, 2]), max(r_PEM_r[:, 2]))
ax.set_box_aspect([1.0, 1.0, 1.0])
plot_tools.set_axes_equal(ax)
plt.title('FF Model in the Inertial (I) Frame')
plt.legend()
plt.show()


# ~~~~~ANIMATIONS~~~~~

# # Animate the CRTBP model
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# # Collect animation data for CRTBP
# N_CRTBP = len(r_PEM_CRTBP[:, 0])  # number of frames in animation
# P_CRTBP = 50  # number of points plotted per frame
#
# data_CRTBP = np.array([r_PEM_CRTBP[:, 0], r_PEM_CRTBP[:, 1], r_PEM_CRTBP[:, 2]])
# data_Earth = np.array([r_EarthEM_CRTBP[:, 0], r_EarthEM_CRTBP[:, 1], r_EarthEM_CRTBP[:, 2]])
# data_Moon = np.array([r_MoonEM_CRTBP[:, 0], r_MoonEM_CRTBP[:, 1], r_MoonEM_CRTBP[:, 2]])
#
# # Initialize the first point for each body
# line_CRTBP, = ax.plot(data_CRTBP[0, 0:1], data_CRTBP[1, 0:1], data_CRTBP[2, 0:1], color='blue', label='Orbit')
# line_Earth, = ax.plot(data_Earth[0, 0:1], data_Earth[1, 0:1], data_Earth[2, 0:1], color='green', label='Earth')
# line_Moon, = ax.plot(data_Moon[0, 0:1], data_Moon[1, 0:1], data_Moon[2, 0:1], color='gray', label='Moon')
#
#
# def animate_CRTBP(i):
#     line_CRTBP.set_data(data_CRTBP[0, :i*P_CRTBP], data_CRTBP[1, :i*P_CRTBP])  # Set the x and y positions
#     line_CRTBP.set_3d_properties(data_CRTBP[2, :i*P_CRTBP])  # Set the z position
#     line_Earth.set_data(data_Earth[0, :i*P_CRTBP], data_Earth[1, :i*P_CRTBP])
#     line_Earth.set_3d_properties(data_Earth[2, :i * P_CRTBP])
#     line_Moon.set_data(data_Moon[0, :i*P_CRTBP], data_Moon[1, :i*P_CRTBP])
#     line_Moon.set_3d_properties(data_Moon[2, :i * P_CRTBP])
#
#
# ani_CRTBP = animation.FuncAnimation(fig, animate_CRTBP, frames=N_CRTBP//P_CRTBP, interval=1, repeat=False)
#
# # Set axes limits
# ax.set_xlim3d(min(data_CRTBP[0]), max(data_CRTBP[0]))
# ax.set_ylim3d(min(data_CRTBP[1]), max(data_CRTBP[1]))
# ax.set_zlim3d(min(data_CRTBP[2]), max(data_CRTBP[2]))
# ax.set_box_aspect([1.0, 1.0, 1.0])
# plot_tools.set_axes_equal(ax)
#
# # Set labels
# ax.set_xlabel('X [AU]')
# ax.set_ylabel('Y [AU]')
# ax.set_zlabel('Z [AU]')
# plt.legend()
# plt.title('CRTBP model in the I frame')


# # Animate the full force model NEEDS FIXING
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# # Collect animation data for full force
# N_FF = len(r_PEM_r[:, 0])  # number of frames in animation
# P_FF = 1  # number of points plotted per frame
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
# def animate_FF(i):
#     line_FF.set_data(data_FF[0, :i*P_FF], data_FF[1, :i*P_FF])
#     line_FF.set_3d_properties(data_FF[2, :i*P_FF])
#     line_EarthFF.set_data(data_EarthFF[0, :i*P_FF], data_EarthFF[1, :i*P_FF])
#     line_EarthFF.set_3d_properties(data_EarthFF[2, 0:i*P_FF])
#     line_MoonFF.set_data(data_MoonFF[0, :i*P_FF], data_MoonFF[1, :i*P_FF])
#     line_MoonFF.set_3d_properties(data_MoonFF[2, 0:i*P_FF])
#     line_SunFF.set_data(data_SunFF[0, :i*P_FF], data_SunFF[1, :i*P_FF])
#     line_SunFF.set_3d_properties(data_SunFF[2, 0:i*P_FF])
#
#
# ani_FF = animation.FuncAnimation(fig, animate_FF, frames=N_FF//P_FF, interval=10, repeat=False)
#
# # Set axes limits
# ax.set_xlim3d(min(data_FF[0]), max(data_FF[0]))
# ax.set_ylim3d(min(data_FF[1]), max(data_FF[1]))
# ax.set_zlim3d(min(data_FF[2]), max(data_FF[2]))
# ax.set_box_aspect([1.0, 1.0, 1.0])
# plot_tools.set_axes_equal(ax)
#
# # Set labels
# ax.set_xlabel('X [AU]')
# ax.set_ylabel('Y [AU]')
# ax.set_zlabel('Z [AU]')
# plt.legend()
# plt.title('Full force model in the I frame')

# ~~~~~ NORMAL PLOTS~~~~~

# # Plot CRTBP and FF solutions
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(posCRTBP[:, 0], posCRTBP[:, 1], posCRTBP[:, 2], 'r', label='CRTBP')
# # ax.plot(posFF[:, 0], posFF[:, 1], posFF[:, 2], 'b', label='Full Force')
# # ax.scatter(r_PEM_r[0, 0], r_PEM_r[0, 1], r_PEM_r[0, 2], marker='*', label='FF Start')
# # ax.scatter(r_PEM_r[-1, 0], r_PEM_r[-1, 1], r_PEM_r[-1, 2], label='FF End')
# ax.set_xlabel('X [AU]')
# ax.set_ylabel('Y [AU]')
# ax.set_zlabel('Z [AU]')
# plt.title('Orbital Motion in the Inertial Frame')
# plt.legend()
#
#
# # Plot the bodies and the FF solution
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(r_EarthEM_r[:, 0], r_EarthEM_r[:, 1], r_EarthEM_r[:, 2], 'g', label='Earth')
# ax.plot(r_MoonEM_r[:, 0], r_MoonEM_r[:, 1], r_MoonEM_r[:, 2], 'r', label='Moon')
# ax.plot(r_SunEM_r[:, 0], r_SunEM_r[:, 1], r_SunEM_r[:, 2], 'y', label='Sun')
# ax.plot(r_PEM_r[:, 0], r_PEM_r[:, 1], r_PEM_r[:, 2], 'b', label='Full Force')
# ax.set_xlabel('X [AU]')
# ax.set_ylabel('Y [AU]')
# ax.set_zlabel('Z [AU]')
# plt.legend()

plt.show()
# breakpoint()

