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
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d + 1*u.yr/4
t_mjd = Time(57680, format='mjd', scale='utc')
days = 200
days_can = unitConversion.convertTime_to_canonical(days * u.d)
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in non dimensional units in rotating frame R [pos, vel]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period

# Convert the velocity to I frame from R frame (position is the same in both)
vO = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# DCM for G frame and I frame
C_I2G = frameConversion.inert2geo(t_mjd, t_equinox, t_veq)
C_G2I = C_I2G.T

# Get position of the moon at the epoch in the inertial frame
sun_I, earth_I, moon_I = frameConversion.getSunEarthMoon(t_mjd, C_I2G)  # I frame [AU]
moon_I_can = unitConversion.convertPos_to_canonical(moon_I)

# Transform position ICs to the epoch moon
ideal_moon = [1-mu_star, 0, 0]
IC_x = (IC[0] - ideal_moon[0]) + moon_I_can[0]
IC_y = (IC[1] - ideal_moon[1]) + moon_I_can[1]
IC_z = (IC[2] - ideal_moon[2]) + moon_I_can[2]
IC[0:3] = [IC_x, IC_y, IC_z]

# Rotate velocity vector to match the epoch moon (I frame)
theta = np.arccos((np.dot(moon_I_can, ideal_moon))/(np.linalg.norm(moon_I_can)*np.linalg.norm(ideal_moon)))
if theta > np.pi/2:
    theta = -theta
rot_matrix = frameConversion.rot(theta, 3)
vI = rot_matrix @ vO

# Convert ICs to H frame (AU and AU/d) from I frame (canonical)
pos_H, vel_H = frameConversion.convertSC_I2H(IC[0:3], vI, t_mjd, C_I2G, Tp_can=None)

# Define the initial state array
state0 = np.append(np.append(pos_H.value, vel_H.value), days_can)

# Propagate the dynamics (states in AU or AU/day, times in DU)
states, times = orbitEOMProp.statePropFF(state0, t_mjd)  # State is in the H frame
pos = states[:, 0:3]
vel = states[:, 3:6]

# Convert to canonical
pos_can = unitConversion.convertPos_to_canonical(pos * u.AU)
vel_can = unitConversion.convertVel_to_canonical(vel * u.AU/u.d)

# Sim time in mjd
times_dim = unitConversion.convertTime_to_dim(times)  # Days from zero
times_mjd = times_dim + t_mjd  # Days from mission start time

# Preallocate space
pos_SC = np.zeros([len(times_mjd), 3])
vel_SC = np.zeros([len(times_mjd), 3])
pos_Sun = np.zeros([len(times_mjd), 3])
pos_Earth = np.zeros([len(times_mjd), 3])
pos_Moon = np.zeros([len(times_mjd), 3])

# Obtain celestial body positions in the I frame [AU]
for ii in np.arange(len(times_mjd)):
    pos_SC[ii, :], vel_SC[ii, :] = frameConversion.convertSC_H2I(pos_can[ii, :], vel_can[ii, :], times_mjd[ii], C_I2G)
    pos_Sun[ii, :], pos_Earth[ii, :], pos_Moon[ii, :] = frameConversion.getSunEarthMoon(times_mjd[ii], C_I2G)


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
ax.plot(pos_SC[:, 0], pos_SC[:, 1], pos_SC[:, 2], color='blue', label='Propagated FF')
ax.plot(pos_Earth[:, 0], pos_Earth[:, 1], pos_Earth[:, 2], color='green', label='Earth')
ax.plot(pos_Moon[:, 0], pos_Moon[:, 1], pos_Moon[:, 2], color='gray', label='Moon')
ax.scatter(unitConversion.convertPos_to_dim([0, 1-mu_star]).to('AU'), 0, s=50, label='Initial ideal moon position')
ax.scatter(moon_I[0], moon_I[1], moon_I[2], s=50, label='Initial actual moon position')
scale = 500
ax.plot([0, vO[0]/scale], [0, vO[1]/scale], [0, vO[2]/scale], label='Unrotated velocity vector')
ax.plot([0, vI[0]/scale], [0, vI[1]/scale], [0, vI[2]/scale], label='Rotated velocity vector')
# ax.plot(pos_Sun[:, 0], pos_Sun[:, 1], pos_Sun[:, 2], color='orange', label='Sun')
# ax.plot(gmat_posinert[:, 0], gmat_posinert[:, 1], gmat_posinert[:, 2], color='red', label='GMAT Orbit')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
# ax.set_box_aspect([1.0, 1.0, 1.0])
# plot_tools.set_axes_equal(ax)
limit = 0.002
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
plt.title('FF Model in the Inertial (I) Frame')
plt.legend()


# # ~~~~~ANIMATION~~~~~
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# # Collect animation data
# data_sc = np.array([pos_SC[:, 0], pos_SC[:, 1], pos_SC[:, 2]])
# data_Earth = np.array([pos_Earth[:, 0], pos_Earth[:, 1], pos_Earth[:, 2]])
# data_Moon = np.array([pos_Moon[:, 0], pos_Moon[:, 1], pos_Moon[:, 2]])
# data_Sun = np.array([pos_Sun[:, 0], pos_Sun[:, 1], pos_Sun[:, 2]])
#
# # Initialize the first point for each body
# line_sc, = ax.plot(data_sc[0, 0:1], data_sc[1, 0:1], data_sc[2, 0:1], color='blue', label='Orbit')
# line_Earth, = ax.plot(data_Earth[0, 0:1], data_Earth[1, 0:1], data_Earth[2, 0:1], color='green', label='Earth')
# line_Moon, = ax.plot(data_Moon[0, 0:1], data_Moon[1, 0:1], data_Moon[2, 0:1], color='gray', label='Moon')
# line_Sun, = ax.plot(data_Sun[0, 0:1], data_Sun[1, 0:1], data_Sun[2, 0:1], color='orange', label='Sun')
#
# interval = unitConversion.convertTime_to_canonical(days * u.d) / 100  # Fixed time interval for each frame
#
#
# def next_frame(times, interval):
#     t0 = 0
#     idx = 1
#     frame_indices = []
#     while idx < len(times):
#         diff = times[idx] - t0
#         if diff >= interval:
#             frame_indices.append(idx)
#             t0 = times[idx]
#         idx += 1
#     return frame_indices
#
#
# def animate(i):
#     idx = frame_indices[i]
#     if idx == 0:
#         return
#     line_sc.set_data(data_sc[0, :idx], data_sc[1, :idx])  # Set the x and y positions
#     line_sc.set_3d_properties(data_sc[2, :idx])  # Set the z position
#     line_Earth.set_data(data_Earth[0, :idx], data_Earth[1, :idx])
#     line_Earth.set_3d_properties(data_Earth[2, :idx])
#     line_Moon.set_data(data_Moon[0, :idx], data_Moon[1, :idx])
#     line_Moon.set_3d_properties(data_Moon[2, :idx])
#     line_Sun.set_data(data_Sun[0, :idx], data_Sun[1, :idx])
#     line_Sun.set_3d_properties(data_Sun[2, :idx])
#
#
# frame_indices = next_frame(times, interval)
# ani = animation.FuncAnimation(fig, animate, frames=len(frame_indices), interval=1, repeat=True)
#
# limit_ani = 0.002
# ax.set_xlim([-limit_ani, limit_ani])
# ax.set_ylim([-limit_ani, limit_ani])
# ax.set_zlim([-limit_ani, limit_ani])
# ax.set_xlabel('X [AU]')
# ax.set_ylabel('Y [AU]')
# ax.set_zlabel('Z [AU]')
# plt.legend()
# plt.title('Full Force model in the I frame')
#
# # # Save
# # writergif = animation.PillowWriter(fps=30)
# # ani.save('CRTBP L2.gif', writer=writergif)

plt.show()

breakpoint()
