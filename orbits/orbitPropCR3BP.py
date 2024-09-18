import numpy as np
import sys
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
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
coord.solar_system.solar_system_ephemeris.set('de440')

# Parameters
t_mjd = Time(57727, format='mjd', scale='utc')
days = 30
days_can = unitConversion.convertTime_to_canonical(days * u.d)
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star
moon_r_can = 1-mu_star  # Radius of the Moon
moon_r = (unitConversion.convertPos_to_dim(moon_r_can)).to('AU')
earth_r_can = mu_star  # Radius of the Earth
earth_r = (unitConversion.convertPos_to_dim(earth_r_can)).to('AU')

# Initial condition in non-dimensional units in rotating frame R [pos, vel, T/2]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period
# IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]  # DRO, 0.9697497-day period

# Convert ICs to dimensional, rotating frame (for GMAT)
pos_dim = unitConversion.convertPos_to_dim(np.array(IC[0:3])).to('km')
vel_dim = unitConversion.convertVel_to_dim(np.array(IC[3:6])).to('km/s')
print('Dimensional [km] position IC in the rotating frame: ', pos_dim)
print('Dimensional [km/s] velocity IC in the rotating frame: ', vel_dim)

# Convert the velocity to inertial from R (still canonical)
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar = np.array([IC[0], IC[2], vI[1], days_can])

# Propagate the dynamics in the CRTBP model
states, times = orbitEOMProp.statePropCRTBP(freeVar, mu_star)  # Canonical
pos = states[:, 0:3]
vel = states[:, 3:6]

# Convert to AU
pos_au = np.array(unitConversion.convertPos_to_dim(pos).to('AU'))

# Define vectors for the Earth and the Moon
pos_moon = np.zeros([len(times), 3])
pos_earth = np.zeros([len(times), 3])

# Define a phase shift for the starting point in the case that y0 is not 0
phi_moon = np.arcsin(IC[1]/moon_r_can)
phi_earth = phi_moon + np.pi
for ii in np.arange(len(times)):
    pos_moon[ii, :] = [np.array(moon_r*np.cos(times[ii]+phi_moon)), np.array(moon_r*np.sin(times[ii]+phi_moon)), 0]
    pos_earth[ii, :] = [np.array(earth_r*np.cos(times[ii]+phi_earth)), np.array(earth_r*np.sin(times[ii]+phi_earth)), 0]


# ~~~~~PLOT SOLUTION AND GMAT FILE IN THE INERTIAL FRAME~~~~

# Obtain CRTBP data from GMAT
file_name = "gmatFiles/CRTBP_rot.txt"
gmat_km, gmat_time = gmatTools.extract_pos(file_name)
gmat_posrot = np.array((gmat_km * u.km).to('AU'))

# Convert to I frame from R frame
gmat_posinert = np.zeros([len(gmat_time), 3])
for ii in np.arange(len(gmat_time)):
    C_I2R = frameConversion.inert2rot(gmat_time[ii], gmat_time[0])
    C_R2I = C_I2R.T
    gmat_posinert[ii, :] = C_R2I @ gmat_posrot[ii, :]

# Plot
title = 'CRTBP Model in the Inertial (I) Frame'
body_names = ['Propagated CRTBP', 'Earth', 'Moon', 'GMAT Orbit']
fig, ax = plot_tools.plot_bodies(pos_au, pos_earth, pos_moon, gmat_posinert, body_names=body_names, title=title)


# ~~~~~ANIMATION~~~~~

desired_duration = 3  # seconds
body_names = ['Spacecraft', 'Earth', 'Moon']
animate_func, ani_object = plot_tools.create_animation(times, days, desired_duration,
                                                       [pos_au, pos_earth, pos_moon], body_names=body_names,
                                                       title=title)


# # ~~~~~SAVE~~~~~
#
# fig.savefig('plotFigures/CRTBP L2 I frame.png')
#
# writergif = animation.PillowWriter(fps=30)
# ani_object.save('plotFigures/CRTBP L2 I frame.gif', writer=writergif)
