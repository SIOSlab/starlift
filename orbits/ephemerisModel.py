import numpy as np
import sys
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
import plot_tools
import extractTools
import spiceypy as spice
import multiShooting as ms

spice.furnsh("fullForce.txt")

# Parameters
gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])

orbs = 1
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d  # + 1*u.yr/4
#t_start = Time(57727, format='mjd', scale='utc')
t_start = Time(58070, format='mjd', scale='utc')
days = 14*orbs
days_can = unitConversion.convertTime_to_canonical(days * u.d)
#mu_star = 0.012150584269940
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in canonical units in rotating frame R [pos, vel]
#IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period
# IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]  # DRO, 0.9697497-day period
#IC = [0.429519110229904, 0, 0, 0, 1.440796689672539, 0, 3.051133070334277]
#IC = [1.165130674583613, 0, -0.110699848144854, 0, 0.201519926517907, 0, 1.652428300688599]
IC = [1.114959432252717, 0, -0.027057507726036, 0, 0.191674660415012, 0, 3.403442494940593/2]   # matlab
# Generate new ICs using the free variable and constraint method
X = [IC[0], IC[2], IC[4], IC[6]]
max_iter = 1000
error = 10
ctr = 0
eps = 1
while error > eps and ctr < max_iter:
    Fx = orbitEOMProp.calcFx_R(X, mu_star)

    error = np.linalg.norm(Fx)
    dFx = orbitEOMProp.calcdFx_CRTBP(X, mu_star, m1, m2)

    X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)

    ctr = ctr + 1

IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])  # Canonical, rotating frame

# Propagate the dynamics (states in AU or AU/day, times in days starting from 0)
freeVar0CRTBP_R = X.copy()
freeVar0CRTBP_R[-1] = 2*freeVar0CRTBP_R[-1]*orbs
statesCRTBP_R, timesCRTBP_R = orbitEOMProp.statePropCRTBP_R(freeVar0CRTBP_R, mu_star)  # State is in the R frame
posCRTBP_R = statesCRTBP_R[:, 0:3]
velCRTBP_R = statesCRTBP_R[:, 3:6]
timesCRTBP_d = unitConversion.convertTime_to_dim(timesCRTBP_R).to('d')
timesCRTBP_mjd = t_start + timesCRTBP_d

N = 9
dt_int = (timesCRTBP_d[-1]-timesCRTBP_d[0])/(N-1)
taus = Time(np.zeros(N), format='mjd', scale='utc')
posvel = np.array([])
for ii in np.arange(N):
    time_i = ii*dt_int

    difference_array_i = np.absolute(timesCRTBP_d-time_i).value
    index_i = difference_array_i.argmin()
    
    taus[ii] = timesCRTBP_mjd[index_i]
    
    state_i = np.append(posCRTBP_R[index_i], velCRTBP_R[index_i])
    
    # change to be relative the moon (just subtract mu from positions)
    pos_i = np.array(state_i[0:3]) - np.array([1-mu_star, 0, 0])
    vel_i = np.array(state_i[3:6])
    
    # convert to km and s
    pos_R = unitConversion.convertPos_to_dim(pos_i).to_value(u.km)
    vel_R = unitConversion.convertVel_to_dim(vel_i).to_value(u.km/u.s)
    
    state_R = np.append(pos_R, vel_R)
    posvel = np.append(posvel, state_R)
posvel = np.reshape(posvel,(N,6))

# R frame, relative to the moon (MCR frame)
posCRTBPM =  posvel[:, 0:3]

# Convert to MCI frame
initialEphmerisEpoch = spice.str2et(t_start.iso)
times_dim = (taus - t_start).to_value(u.s)
initialEphmerisEpoches = initialEphmerisEpoch + times_dim
Crv_R2I = spice.sxform('MCR','MCI',initialEphmerisEpoches)
initialEphemerisMCI = np.zeros((len(times_dim), 6))
for ii in np.arange(len(times_dim)):
    initialEphemerisMCI[ii,:] = Crv_R2I[ii,:,:] @ posvel[ii]
    
positionTolerance = 0.01    # km
velocityTolerance = 0.0001  # km/s

correctedInitialEpoches, correctedInitialStates, exitflag = ms.multipleShooting(initialEphmerisEpoches, initialEphemerisMCI, positionTolerance, velocityTolerance, GM)

breakpoint()
