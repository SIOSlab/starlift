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
#GM = np.array([gmMoon, gmEarth])

orbs = 1
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d
#t_start = Time(57727, format='mjd', scale='utc')
t_start = Time(58070, format='mjd', scale='utc')
days = 14*orbs
days_can = unitConversion.convertTime_to_canonical(days * u.d)
#mu_star = 0.012150584269940
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

et_start = spice.str2et(t_start.iso)
rvMoon = spice.spkezr('Moon', et_start, 'J2000', 'None', 'Earth')[0]
omega_m = spice.oscltx(rvMoon,et_start,gmEarth)[-1]

# Initial condition in canonical units in rotating frame R [pos, vel]
#IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period
IC = [1.01103506347211, 0, -0.17315001039682773, 0, -0.07801414771853428, 0, 1.363209636932144/2]    #1E-7 error version of ^
#IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]   # DRO, 0.9697497-day period
#IC = [0.429519110229904, 0, 0, 0, 1.440796689672539, 0, 3.051133070334277]
#IC = [1.165130674583613, 0, -0.110699848144854, 0, 0.201519926517907, 0, 1.652428300688599]
#IC = [1.114959432252717, 0, 0.027057507726036, 0, 0.191674660415012, 0, 3.403442494940593/2]   # matlab
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
    print('Error is: '+str(error))

IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])  # Canonical, rotating frame
print('Number of attempts: '+str(ctr))
# Propagate the dynamics (states in AU or AU/day, times in days starting from 0)
freeVar0CRTBP_R = X.copy()
freeVar0CRTBP_R[-1] = 2*freeVar0CRTBP_R[-1]*orbs
statesCRTBP_R, timesCRTBP_R = orbitEOMProp.statePropCRTBP_R(freeVar0CRTBP_R, mu_star)  # State is in the R frame
posCRTBP_R = statesCRTBP_R[:, 0:3]
velCRTBP_R = statesCRTBP_R[:, 3:6]
posCRTBP_R_dim = unitConversion.convertPos_to_dim(posCRTBP_R - np.array([1-mu_star, 0, 0])).to_value(u.km)
velCRTBP_R_dim = unitConversion.convertVel_to_dim(velCRTBP_R).to_value(u.km/u.s)
timesCRTBP_d = unitConversion.convertTime_to_dim(timesCRTBP_R).to('d')
timesCRTBP_mjd = t_start + timesCRTBP_d
etCRTBP_mjd = spice.str2et(timesCRTBP_mjd.iso)

posCRTBP_I_dim = np.zeros((len(etCRTBP_mjd), 3))
for ii in np.arange(len(etCRTBP_mjd)):
    Crv_R2I = spice.sxform('MCR','MCI',etCRTBP_mjd[ii])
    state_R = np.append(posCRTBP_R_dim[ii], velCRTBP_R_dim[ii])
    state_I = Crv_R2I@state_R
    posCRTBP_I_dim[ii,:] = state_I[0:3]

N = 9*orbs
dt_int = (timesCRTBP_d[-1]-timesCRTBP_d[0])/(N-1)
taus = Time(np.zeros(N), format='mjd', scale='utc')
posvel = np.array([])
for ii in np.arange(N):
    time_i = ii*dt_int

    difference_array_i = np.absolute(timesCRTBP_d-time_i).value
    index_i = difference_array_i.argmin()
    
    taus[ii] = timesCRTBP_mjd[index_i]
    
    # change to be relative the moon (just subtract mu from positions)
    pos_i = posCRTBP_R[index_i] - np.array([1-mu_star, 0, 0])
    vel_i = velCRTBP_R[index_i]
    
    # convert to km and s
    pos_R = unitConversion.convertPos_to_dim(pos_i).to_value(u.km)
    vel_R = unitConversion.convertVel_to_dim(vel_i).to_value(u.km/u.s)
    
    state_R = np.append(pos_R, vel_R)
    posvel = np.append(posvel, state_R)
#    state_R = np.append(posCRTBP_R_dim[index_i], velCRTBP_R_dim[index_i])
#    posvel = np.append(posvel, state_R)
posvel = np.reshape(posvel,(N,6))

# R frame, relative to the moon (MCR frame)
posCRTBPM =  posvel[:, 0:3]

ax1 = plt.figure().add_subplot(projection='3d')
for ii in np.arange(N):
    ax1.scatter(posvel[ii,0], posvel[ii,1], posvel[ii,2], c='g', marker='o')
ax1.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r', label='CRTBP')
ax1.set_xlabel('X [km]')
ax1.set_ylabel('Y [km]')
ax1.set_zlabel('Z [km]')
ax1.set_title('Patch Points in Moon Centered Rotating Frame')

# Convert to MCI frame
initialEphmerisEpoch = spice.str2et(t_start.iso)
times_dim = (taus - t_start).to_value(u.s)
initialEphmerisEpoches = initialEphmerisEpoch + times_dim
Crv_R2I = spice.sxform('MCR','MCI',initialEphmerisEpoches)
initialEphemerisMCI = np.zeros((len(times_dim), 6))
for ii in np.arange(len(times_dim)):
    initialEphemerisMCI[ii,:] = Crv_R2I[ii,:,:] @ posvel[ii]
    
positionTolerance = 0.01    # km
velocityTolerance = 0.0001*orbs  # km/s

#correctedInitialEpoches, correctedInitialStates, exitflag = ms.multipleShootingI(initialEphmerisEpoches, initialEphemerisMCI, positionTolerance, velocityTolerance, GM)
correctedInitialEpoches, correctedInitialStates, exitflag = ms.multipleShootingR(initialEphmerisEpoches, initialEphemerisMCI, positionTolerance, velocityTolerance, GM, omega_m)
#correctedInitialEpoches, correctedInitialStates, exitflag = ms.multipleShootingR(initialEphmerisEpoches, posvel, positionTolerance, velocityTolerance, GM, omega_m)

# Plot in MCI and MCR
ax10 = plt.figure().add_subplot(projection='3d')
ax11 = plt.figure().add_subplot(projection='3d')
for ii in np.arange(N-1):
    Ts = correctedInitialEpoches[ii:ii+2]
    times, states = ms.statePropFFI(Ts, correctedInitialStates[ii,:], GM)
    
    ax11.plot(states[:, 0], states[:, 1], states[:, 2], 'b', label='Multi Segment')
    ax11.scatter(states[0,0], states[0,1], states[0,2], c='g', marker='o')
    ax11.scatter(states[-1,0], states[-1,1], states[-1,2], c='y', marker='*')
    
    Crv_I2R = spice.sxform('MCI','MCR',times)
    rotatedStates = np.zeros((len(times), 6))
    for jj in np.arange(len(times)):
        rotatedStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]

    ax10.plot(rotatedStates[:, 0], rotatedStates[:, 1], rotatedStates[:, 2], 'b', label='Multi Segment')
    ax10.scatter(rotatedStates[0,0], rotatedStates[0,1], rotatedStates[0,2], c='g', marker='o')
    ax10.scatter(rotatedStates[-1,0], rotatedStates[-1,1], rotatedStates[-1,2], c='y', marker='*')
ax11.plot(posCRTBP_I_dim[:,0], posCRTBP_I_dim[:,1], posCRTBP_I_dim[:,2], 'r-.', label='CRTBP')
ax11.set_xlabel('X [km]')
ax11.set_ylabel('Y [km]')
ax11.set_zlabel('Z [km]')
ax11.set_title('Moon Centered Inertial Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

ax10.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP')
ax10.set_xlabel('X [km]')
ax10.set_ylabel('Y [km]')
ax10.set_zlabel('Z [km]')
ax10.set_title('Moon Centered Rotating Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

#ax10 = plt.figure().add_subplot(projection='3d')
#ax11 = plt.figure().add_subplot(projection='3d')
#for ii in np.arange(N-1):
#    Ts = correctedInitialEpoches[ii:ii+2]
#    times, states = ms.statePropFFR(Ts, correctedInitialStates[ii,:], GM, omega_m)
#    
#    ax11.plot(states[:, 0], states[:, 1], states[:, 2], 'b', label='Multi Segment')
#    ax11.scatter(states[0,0], states[0,1], states[0,2], c='g', marker='o')
#    ax11.scatter(states[-1,0], states[-1,1], states[-1,2], c='y', marker='*')
#    
#    Crv_R2I = spice.sxform('MCR','MCI',times)
#    inertialStates = np.zeros((len(times), 6))
#    for jj in np.arange(len(times)):
#        inertialStates[jj,:] = Crv_R2I[jj,:,:]@states[jj,:]
#
#    ax10.plot(inertialStates[:, 0], inertialStates[:, 1], inertialStates[:, 2], 'b', label='Multi Segment')
#    ax10.scatter(inertialStates[0,0], inertialStates[0,1], inertialStates[0,2], c='g', marker='o')
#    ax10.scatter(inertialStates[-1,0], inertialStates[-1,1], inertialStates[-1,2], c='y', marker='*')
#ax10.plot(posCRTBP_I_dim[:,0], posCRTBP_I_dim[:,1], posCRTBP_I_dim[:,2], 'r-.', label='CRTBP')
#ax10.set_xlabel('X [km]')
#ax10.set_ylabel('Y [km]')
#ax10.set_zlabel('Z [km]')
#ax10.set_title('Moon Centered Inertial Frame')
#plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)
#
#ax11.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP')
#ax11.set_xlabel('X [km]')
#ax11.set_ylabel('Y [km]')
#ax11.set_zlabel('Z [km]')
#ax11.set_title('Moon Centered Rotating Frame')
#plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)
plt.show()
breakpoint()
