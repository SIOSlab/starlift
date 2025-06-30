import numpy as np
import sys
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.io import loadmat
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
import singleShooting as ss

spice.furnsh("fullForce.txt")

# Parameters
gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])

orbs = 1
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d
t_start = Time(57727, format='mjd', scale='utc')
#t_start = Time(58070, format='mjd', scale='utc')
mu_star = gmMoon/(gmEarth + gmMoon)
m1 = (1 - mu_star)
m2 = mu_star

et_start = spice.str2et(t_start.iso)
rvMoon = spice.spkezr('Moon', et_start, 'J2000', 'None', 'Earth')[0]
Tp_m = spice.oscltx(rvMoon, et_start, gmEarth)[-1]
omega_m = 2*np.pi/Tp_m

# Initial condition in canonical units in rotating frame R [pos, vel, time, U]
mat_data = loadmat('TrajI_1265.mat')['TrajI']
#mat_data = loadmat('TrajExample.mat')['TrajI']
posCRTBP_R = mat_data[:,0:3]
velCRTBP_R = mat_data[:,3:6]
timesCRTBP_R = mat_data[:,6]
uT = mat_data[:,7:]
mu_cstar = 0.01215059

# Convert from nondimensional units to dimensional
posCRTBP_R_dim = unitConversion.convertPos_to_dim(posCRTBP_R - np.array([1-mu_star, 0, 0])).to_value(u.km)
velCRTBP_R_dim = unitConversion.convertVel_to_dim(velCRTBP_R).to_value(u.km/u.s)
timesCRTBP_d = unitConversion.convertTime_to_dim(timesCRTBP_R).to('d')
timesCRTBP_mjd = t_start + timesCRTBP_d
etCRTBP_mjd = spice.str2et(timesCRTBP_mjd.iso)

Ts = np.array([etCRTBP_mjd[0], etCRTBP_mjd[-1]])
state0R = np.append(posCRTBP_R_dim[0,:], velCRTBP_R_dim[0,:])
times, FF_R_dim = ms.statePropFFR(Ts, state0R, GM, omega_m)

stateCRTBP_I_dim = np.zeros((len(etCRTBP_mjd), 6))
for ii in np.arange(len(etCRTBP_mjd)):
    Crv_R2I = spice.sxform('MCR','MCI',etCRTBP_mjd[ii])
    state_R = np.append(posCRTBP_R_dim[ii], velCRTBP_R_dim[ii])
    stateCRTBP_I_dim[ii,:] = Crv_R2I@state_R
    
state0I = stateCRTBP_I_dim[0,:]
times, FF_I_dim = ms.statePropFFI(Ts, state0I, GM)

Crv_I2R = spice.sxform('MCI','MCR',times)
FF_R_dim = np.zeros((len(times), 6))
for ii in np.arange(len(times)):
    FF_R_dim[ii,:] = Crv_I2R[ii,:,:]@FF_I_dim[ii,:]

ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(FF_R_dim[:,0], FF_R_dim[:,1], FF_R_dim[:,2], 'b', label='Ephemeris Model Inertial')
ax1.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP Model')
ax1.plot(FF_R_dim[:,0], FF_R_dim[:,1], FF_R_dim[:,2], 'g-.', label='Ephemeris Model Rotating')
plt.legend()



plt.show()
breakpoint()
