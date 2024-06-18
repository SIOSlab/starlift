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

import pdb

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de432s')

# Parameters
t_mjd = Time(57727, format='mjd', scale='utc')
days = 30
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in non dimensional units in rotating frame R [pos, vel]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]

# Convert the velocity to inertial from I
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar_CRTBP = np.array([IC[0], IC[2], vI[1], days])

# propagate the dynamics in the CRTBP model
statesCRTBP, timesCRTBP = orbitEOMProp.statePropCRTBP(freeVar_CRTBP, mu_star)
posCRTBP = statesCRTBP[:, 0:3]
velCRTBP = statesCRTBP[:, 3:6]

# convert position from I frame to H frame
pos_H, vel_H, Tp_dim = orbitEOMProp.convertIC_R2H(posCRTBP[0], velCRTBP[0], t_mjd, timesCRTBP[-1], mu_star)

# Define the initial state array
state0 = np.append(np.append(pos_H.value, vel_H.value), days)   # Tp_dim.value

# propagate the dynamics
statesFF, timesFF = orbitEOMProp.statePropFF(state0, t_mjd)
posFF = statesFF[:, 0:3]
velFF = statesFF[:, 3:6]

# preallocate space
r_PEM_r = np.zeros([len(timesFF), 3])
r_SunEM_r = np.zeros([len(timesFF), 3])
r_EarthEM_r = np.zeros([len(timesFF), 3])
r_MoonEM_r = np.zeros([len(timesFF), 3])

# sim time in mjd
timesFF_mjd = timesFF + t_mjd

# DCM for G frame and I frame
C_B2G = frameConversion.body2geo(t_mjd,t_mjd,mu_star)
C_G2B = C_B2G.T
for ii in np.arange(len(timesFF)):
    time = timesFF_mjd[ii]
    
    # positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
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


# animate the CRTBP orbit
P = 50  # number of points plotted per frame
def update(num, data, line):
    line.set_data(data[0:2, 0:num*P])
    line.set_3d_properties(data[2, 0:num*P])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
data_CRTBP = np.array([posCRTBP[:, 0], posCRTBP[:, 1], posCRTBP[:, 2]])
N = len(data_CRTBP[0])  # number of frames in animation
line_CRTBP, = ax.plot(data_CRTBP[0, 0:1], data_CRTBP[1, 0:1], data_CRTBP[2, 0:1])

ani = animation.FuncAnimation(fig, update, frames=N//P, fargs=(data_CRTBP, line_CRTBP), interval=1, repeat=False,
                              cache_frame_data=False)

"""
# plot CRTBP and FF solutions
ax = plt.figure().add_subplot(projection='3d')
ax.plot(posCRTBP[:, 0], posCRTBP[:, 1], posCRTBP[:, 2], 'r', label='CRTBP')
# ax.plot(posFF[:, 0], posFF[:, 1], posFF[:, 2], 'b', label='Full Force')
# ax.scatter(r_PEM_r[0, 0], r_PEM_r[0, 1], r_PEM_r[0, 2], marker='*', label='FF Start')
# ax.scatter(r_PEM_r[-1, 0], r_PEM_r[-1, 1], r_PEM_r[-1, 2], label='FF End')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
plt.title('Orbital Motion in the Inertial Frame')
plt.legend()


# plot the bodies and the FF solution
ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_EarthEM_r[:, 0], r_EarthEM_r[:, 1], r_EarthEM_r[:, 2], 'g', label='Earth')
ax.plot(r_MoonEM_r[:, 0], r_MoonEM_r[:, 1], r_MoonEM_r[:, 2], 'r', label='Moon')
ax.plot(r_SunEM_r[:, 0], r_SunEM_r[:, 1], r_SunEM_r[:, 2], 'y', label='Sun')
ax.plot(r_PEM_r[:, 0], r_PEM_r[:, 1], r_PEM_r[:, 2], 'b', label='Full Force')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
plt.legend()
"""

plt.show()
# breakpoint()

