import numpy as np
import os.path
import pickle
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp

import pdb

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de432s')

# Parameters
t_mjd = Time(57727,format='mjd',scale='utc')
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
statesCRTBP, timesCRTBP = orbitEOMProp.statePropCRTBP(freeVar_CRTBP,mu_star)
posCRTBP = statesCRTBP[:,0:3]
velCRTBP = statesCRTBP[:,3:6]

# preallocate space
r_PEM_r = np.zeros([len(timesCRTBP),3])
r_SunEM_r = np.zeros([len(timesCRTBP),3])
r_EarthEM_r = np.zeros([len(timesCRTBP),3])
r_MoonEM_r = np.zeros([len(timesCRTBP),3])

# sim time in mjd
timesCRTBP_mjd = timesCRTBP + t_mjd

# DCM for G frame and I frame
C_B2G = frameConversion.body2geo(t_mjd,t_mjd,mu_star)
C_G2B = C_B2G.T
for ii in np.arange(len(timesCRTBP)):
    time = timesCRTBP_mjd[ii]
    
    # positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',time)
    r_EMO = EMO[0].get_xyz().to('AU').value
    
    # convert from H frame to GCRS frame
    r_EMG = frameConversion.icrs2gcrs(r_EMO*u.AU,time)
    r_SunG = frameConversion.icrs2gcrs(r_SunO*u.AU,time)
    r_MoonG = frameConversion.icrs2gcrs(r_MoonO*u.AU,time)
    
    # change the origin to the EM barycenter, G frame
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG
    
    # convert from G frame to I frame
    r_SunEM_r[ii,:] = C_G2B@r_SunEM.to('AU')
    r_EarthEM_r[ii,:] = C_G2B@r_EarthEM.to('AU')
    r_MoonEM_r[ii,:] = C_G2B@r_MoonEM.to('AU')

# plots
#ax = plt.figure().add_subplot(projection='3d')

# plot the bodies and the FF solution
#ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2],'g',label='Earth')
#ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2],'r',label='Moon')
#ax.plot(r_SunEM_r[:,0],r_SunEM_r[:,1],r_SunEM_r[:,2],'y',label='Sun')
#ax.plot(posCRTBP[:,0],posCRTBP[:,1],posCRTBP[:,2],'b',label='CRTBP')
#ax.set_xlabel('X [AU]')
#ax.set_ylabel('Y [AU]')
#ax.set_zlabel('Z [AU]')
#plt.legend()

#plt.show()
breakpoint()
