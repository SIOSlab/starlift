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
t_mjd = Time(57727, format='mjd', scale='utc')
days = 200
days_can = unitConversion.convertTime_to_canonical(days * u.d)
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in non dimensional units in rotating frame R [pos, vel]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period

# Generate new ICs using the free variable and constraint method
X = [IC[0], IC[2], IC[4], IC[6]]
max_iter = 1000
error = 10
ctr = 0
eps = 4E-6
while error > eps and ctr < max_iter:
    Fx = orbitEOMProp.calcFx_R(X, mu_star)

    error = np.linalg.norm(Fx)
    dFx = orbitEOMProp.calcdFx_CRTBP(X, mu_star, m1, m2)

    X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)

    ctr = ctr + 1

IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2 * X[3]])  # Canonical, rotating frame

# DCM for G frame and I frame
C_I2G = frameConversion.inert2geo(t_mjd, t_veq)
C_G2I = C_I2G.T

# Get position of the moon at the epoch in the inertial frame
sun_I, earth_I, moon_I = frameConversion.getSunEarthMoon(t_mjd, C_I2G)  # I frame [AU]
moon_I_can = unitConversion.convertPos_to_canonical(moon_I)

# Transform position ICs to the epoch moon
ideal_moon = [1-mu_star, 0, 0]
IC_x = (IC[0] - ideal_moon[0]) + moon_I_can[0]
IC_y = (IC[1] - ideal_moon[1]) + moon_I_can[1]
IC_z = (IC[2] - ideal_moon[2]) + moon_I_can[2]
IC[0:3] = [IC_x, IC_y, IC_z]  # Canonical, I frame

# Convert the velocity to I frame from R frame (position is the same in both)
vO = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Rotate velocity vector to match the epoch moon (I frame)
theta = np.arccos((np.dot(moon_I_can, ideal_moon))/(np.linalg.norm(moon_I_can)*np.linalg.norm(ideal_moon)))
if theta > np.pi/2:
    theta = -theta
rot_matrix = frameConversion.rot(theta, 3)
IC[3:6] = rot_matrix @ vO  # Canonical, I frame

# # Convert IC to dimensional, rotating frame (for GMAT)
# pos_dim = unitConversion.convertPos_to_dim(IC[0:3]).to('km')
# vel_dim = unitConversion.convertVel_to_dim(IC[3:6]).to('km/s')
# C_I2R = frameConversion.inert2rot(t_mjd, t_mjd)
# pos_dimrot = C_I2R @ pos_dim
# vel_dimrot = C_I2R @ vel_dim
# print('Dimensional position IC in the rotating frame: ', pos_dimrot)
# print('Dimensional velocity IC in the rotating frame: ', vel_dimrot)

# Convert ICs to H frame (AU and AU/d) from I frame (canonical)
pos_H, vel_H = frameConversion.convertSC_I2H(IC[0:3], IC[3:6], t_mjd, C_I2G, Tp_can=None)

# Define the initial state array
state0 = np.append(np.append(pos_H.value, vel_H.value), days_can)

# Propagate the dynamics (states in AU or AU/day, times in DU)
states, times = orbitEOMProp.statePropFF(state0, t_mjd)  # State is in the H frame
pos = states[:, 0:3]
vel = states[:, 3:6]

# Convert to canonical
pos_can = unitConversion.convertPos_to_canonical(pos * u.AU)
vel_can = unitConversion.convertVel_to_canonical(vel * u.AU/u.d)

# Simulation time in mjd
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

# Obtain FF rotating data from GMAT
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
title = 'Full Force Model in the Inertial (I) Frame'
body_names = ['Propagated FF', 'Earth', 'Moon', 'Sun', 'GMAT Orbit']
fig, ax = plot_tools.plot_bodies(pos_SC, pos_Earth, pos_Moon, pos_Sun, gmat_posinert, body_names=body_names, title=title)

# # Save
# fig.savefig('FF L2.png')


# ~~~~~ANIMATION~~~~~

desired_duration = 3  # seconds
body_names = ['Spacecraft', 'Earth', 'Moon', 'Sun']
animate_func, ani_object = plot_tools.create_animation(times, days, desired_duration,
                                                       [pos_SC, pos_Earth, pos_Moon, pos_Sun], body_names=body_names,
                                                       title=title)

# # Save
# writergif = animation.PillowWriter(fps=30)
# ani_object.save('FF L2.gif', writer=writergif)
