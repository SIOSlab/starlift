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
import singleShooting as ss

spice.furnsh("fullForce.txt")

showPlots = False
# Parameters
gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])
#GM = np.array([gmMoon, gmEarth, 0.0])

orbs = 4
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

# Initial condition in canonical units in rotating frame R [pos, vel]
#IC = [((1 - mu_star) - 0.023413), 0, 0, 0, 0.720544, 0, 0.102081]
IC = [1.01103506347211, 0, -0.17315001039682773, 0, -0.07801414771853428, 0, 1.363209636932144/2]  #L2, 5.92773293-day period
#IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]   # DRO, 0.9697497-day period
#IC = [0.429519110229904, 0, 0, 0, 1.440796689672539, 0, 3.051133070334277] # DRO
#IC = [0.517332653163958, 0, 0, 0, 1.12965881302616, 0, 8.50664047891897] # P3DRO, fails miserably
#IC = [1.165130674583613, 0, -0.110699848144854, 0, 0.201519926517907, 0, 1.652428300688599]
#IC = [1.114959432252717, 0, 0.027057507726036, 0, 0.191674660415012, 0, 3.403442494940593/2]   # matlab
#IC = [1.11495, 0, 0.02705, 0, 0.19167, 0, 3.40344/2]   # matlab
#IC = [0.856382122325864, 0, -0.181519309916197, 0, 0.257898218422393, 0, 1.22727308466325]  # L1
#IC = [1.06896234204296, 0, 0.159599443574046, 0, -0.00769167653854165, 0, 1.66142030228280] # butterfly
#IC = [0.766044481790803, 0, 0, 0, 0.488736680662207, 0, 2.20546980585774]   # L1 lyapunov
#IC = [0.265819894849149, 0, 0, 0, 2.27750677757506, 0, 6.25588866460133]    # 2:1 resonant, fails miserably
#IC = [0.139106790847531, 0, 0, 0, 3.35999055380076, 0, 9.40977341640670]    # 2:3 resonant, fails miserably

# Generate new ICs using the free variable and constraint method
X = [IC[0], IC[2], IC[4], IC[6]]
max_iter = 50
error = 10
ctr = 0
eps = 1E-5
while error > eps and ctr < max_iter:
    if np.mod(ctr,5) == 0.0:
        pltX = [X[0], X[1], X[2], 2*X[3]]
        posvelSS, _ = orbitEOMProp.statePropCRTBP_R(pltX, mu_star)
        ax1 = plt.figure().add_subplot(projection='3d')
        ax1.plot(posvelSS[:,0], posvelSS[:,1], posvelSS[:,2])
        ax1.set_title('Single Shooting in progress...')
        if showPlots:
            plt.show()

    Fx = ss.calcFx_R(X, mu_star)

    error = np.linalg.norm(Fx)
    if error < eps:
        print('Error is: '+str(error))
        break
        
    dFx = ss.calcdFx_CRTBP(X, mu_star, m1, m2)

    X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)

    ctr = ctr + 1
    print('Error is: '+str(error))

print('Number of attempts: '+str(ctr))
# Propagate the dynamics (states in AU or AU/day, times in days starting from 0)
freeVar0CRTBP_R = X.copy()
freeVar0CRTBP_R[-1] = 2*freeVar0CRTBP_R[-1]
statesCRTBP_R, timesCRTBP_R = orbitEOMProp.statePropCRTBP_R(freeVar0CRTBP_R, mu_star)  # State is in the R frame
posCRTBP_R = statesCRTBP_R[:, 0:3]
velCRTBP_R = statesCRTBP_R[:, 3:6]

# Convert from nondimensional units to dimensional
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

timesCRTBP_dtot = timesCRTBP_d.copy()
timesCRTBP_mjdtot = timesCRTBP_mjd.copy()
posCRTBP_dimtot = posCRTBP_R_dim.copy()
velCRTBP_dimtot = velCRTBP_R_dim.copy()
for ii in np.arange(1,orbs):
    next_times = timesCRTBP_d[1:] + timesCRTBP_dtot[-1]
    timesCRTBP_dtot = np.append(timesCRTBP_dtot, next_times)
    
    next_mjd = timesCRTBP_d[1:] + timesCRTBP_mjdtot[-1]
    timesCRTBP_mjdtot = Time(np.append(timesCRTBP_mjdtot.value, next_mjd.value), format='mjd', scale='utc')

    posCRTBP_dimtot = np.vstack((posCRTBP_dimtot[:-1,:], posCRTBP_R_dim))
    velCRTBP_dimtot = np.vstack((velCRTBP_dimtot[:-1,:], velCRTBP_R_dim))
ax2 = plt.figure().add_subplot(projection='3d')
ax2.plot(posCRTBP_dimtot[:,0], posCRTBP_dimtot[:,1], posCRTBP_dimtot[:,2])
ax2.set_title('Multiple Orbits Check')

N1 = 37
posvel, taus = ms.getPatches(N1, timesCRTBP_dtot, timesCRTBP_mjdtot, posCRTBP_dimtot, velCRTBP_dimtot)

#dt_int = (timesCRTBP_dtot[-1]-timesCRTBP_dtot[0])/(N1-1)
#taus = Time(np.zeros(N1), format='mjd', scale='utc')
#posvel = np.array([])
#for ii in np.arange(N1):
#    time_i = ii*dt_int
#
#    # find the index
#    difference_array_i = np.absolute(timesCRTBP_dtot-time_i).value
#    index_i = difference_array_i.argmin()
#    
#    # index the time, position, and velocity
#    taus[ii] = timesCRTBP_mjdtot[index_i]
#    pos_R = posCRTBP_dimtot[index_i,:]
#    vel_R = velCRTBP_dimtot[index_i,:]
#    
#    # package the state
#    state_R = np.append(pos_R, vel_R)
#    posvel = np.append(posvel, state_R)
#posvel = np.reshape(posvel,(N1,6))

# R frame, relative to the moon (MCR frame)
posCRTBPM = posvel[:, 0:3]

ax3 = plt.figure().add_subplot(projection='3d')
for ii in np.arange(N1):
    ax3.scatter(posvel[ii,0], posvel[ii,1], posvel[ii,2], c='g', marker='o')
ax3.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r', label='CRTBP')
ax3.set_xlabel('X [km]')
ax3.set_ylabel('Y [km]')
ax3.set_zlabel('Z [km]')
ax3.set_title('Patch Points in Moon Centered Rotating Frame')
if showPlots:
    plt.show()

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

correctedInitialEpoches, correctedInitialStates, exitflag = ms.multipleShootingI(initialEphmerisEpoches, initialEphemerisMCI, positionTolerance, velocityTolerance, GM)
#correctedInitialEpoches, correctedInitialStates, exitflag = ms.multipleShootingR(initialEphmerisEpoches, initialEphemerisMCI, positionTolerance, velocityTolerance, GM, omega_m)

# Plot in MCI and MCR
ax4 = plt.figure().add_subplot(projection='3d')
ax5 = plt.figure().add_subplot(projection='3d')
inertialStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
rotatedStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
timesFull = np.array([])
for ii in np.arange(N1-1):
    Ts = correctedInitialEpoches[ii:ii+2]
    times, states = ms.statePropFFI(Ts, correctedInitialStates[ii,:], GM)
        
    inertialStates = np.vstack((inertialStates, states))
    
    ax5.plot(states[:, 0], states[:, 1], states[:, 2], 'b', label='Multi Segment')
    ax5.scatter(states[0,0], states[0,1], states[0,2], c='g', marker='o')
    ax5.scatter(states[-1,0], states[-1,1], states[-1,2], c='y', marker='*')
    
    Crv_I2R = spice.sxform('MCI','MCR',times)
    rStates = np.zeros((len(times), 6))
    for jj in np.arange(len(times)):
        rStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]
        
    ax4.plot(rStates[:, 0], rStates[:, 1], rStates[:, 2], 'b', label='Multi Segment')
    ax4.scatter(rStates[0,0], rStates[0,1], rStates[0,2], c='g', marker='o')
    ax4.scatter(rStates[-1,0], rStates[-1,1], rStates[-1,2], c='y', marker='*')

    rotatedStates = np.vstack((rotatedStates, rStates))
    timesFull = np.append(timesFull, times)

diff = rotatedStates[-1] - rotatedStates[1]
rDiff = np.linalg.norm(diff[0:3])
vDiff = np.linalg.norm(diff[3:6])
print('End Position Difference: '+str(rDiff)+' km')
print('End Velocity Difference: '+str(vDiff)+' km/s')

ax4.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP')
ax4.set_xlabel('X [km]')
ax4.set_ylabel('Y [km]')
ax4.set_zlabel('Z [km]')
ax4.set_title('Moon Centered Rotating Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

ax5.plot(posCRTBP_I_dim[:,0], posCRTBP_I_dim[:,1], posCRTBP_I_dim[:,2], 'r-.', label='CRTBP')
ax5.set_xlabel('X [km]')
ax5.set_ylabel('Y [km]')
ax5.set_zlabel('Z [km]')
ax5.set_title('Moon Centered Inertial Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

fig, (ax6, ax7, ax8) = plt.subplots(3, 1)
ax6.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,0])
ax6.set_ylabel('x')
ax7.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,1])
ax7.set_ylabel('y')
ax8.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,2])
ax8.set_ylabel('z')

rmag = np.linalg.norm(rotatedStates[1:,0:3], axis=1)
plt.figure(9)
plt.plot(np.arange(len(rotatedStates[1:,2])), rmag)
if showPlots:
    plt.show()

# calculate all the perilunes
rmags = np.linalg.norm(rotatedStates[1:,0:3], axis=1)

# find first perilune (new first patch point)
oneOrb = timesCRTBP_d[-1].to_value(u.s)
difference_array1 = np.absolute((timesFull - timesFull[0]) - oneOrb)
oneOrb_ind = difference_array1.argmin()
rmag1 = np.linalg.norm(rotatedStates[1:oneOrb_ind,0:3], axis=1)
rmin1 = min(rmag1)
min1_ind = np.argwhere(rmags == rmin1)[0][0]

# find last perilune (new last patch point
lastOrb = (orbs-1)*oneOrb
difference_array2 = np.absolute((timesFull - timesFull[0]) - lastOrb)
lastOrb_ind = difference_array2.argmin()
rmagLast = np.linalg.norm(rotatedStates[lastOrb_ind:,0:3], axis=1)
rminLast = min(rmagLast)
minLast_ind = np.argwhere(rmags == rminLast)[0][0]

# redo the patches
timesPeri_d = ((timesFull[min1_ind:minLast_ind] - timesFull[min1_ind])*u.s).to('d')
t_startStr = spice.et2utc(timesFull[min1_ind], 'ISOC', 23, 24)
t_startNew = Time(t_startStr, format='isot', scale='utc')
timesPeri_mjd = Time(t_startNew.mjd + timesPeri_d.value, format='mjd', scale='utc')
posPeri = inertialStates[min1_ind:minLast_ind,0:3]
velPeri = inertialStates[min1_ind:minLast_ind,3:6]

N2 = 31
posvel, taus = ms.getPatches(N2, timesPeri_d, timesPeri_mjd, posPeri, velPeri)
initialEphmerisEpoch = spice.str2et(timesPeri_mjd[0].iso)
times_dim = (taus - timesPeri_mjd[0]).to_value(u.s)
initialEphmerisEpoches = initialEphmerisEpoch + times_dim

correctedInitialEpochesNew, correctedInitialStatesNew, exitflag = ms.multipleShootingI(initialEphmerisEpoches, posvel, positionTolerance, velocityTolerance, GM)

# Plot in MCI and MCR
ax10 = plt.figure().add_subplot(projection='3d')
ax11 = plt.figure().add_subplot(projection='3d')
inertialStatesNew = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
rotatedStatesNew = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
timesFullNew = np.array([])
for ii in np.arange(N2-1):
    Ts = correctedInitialEpochesNew[ii:ii+2]
    times, states = ms.statePropFFI(Ts, correctedInitialStatesNew[ii,:], GM)
        
    inertialStatesNew = np.vstack((inertialStatesNew, states))
    
    ax11.plot(states[:, 0], states[:, 1], states[:, 2], 'b', label='Multi Segment')
    ax11.scatter(states[0,0], states[0,1], states[0,2], c='g', marker='o')
    ax11.scatter(states[-1,0], states[-1,1], states[-1,2], c='y', marker='*')
    
    Crv_I2R = spice.sxform('MCI','MCR',times)
    rStates = np.zeros((len(times), 6))
    for jj in np.arange(len(times)):
        rStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]
        
    ax10.plot(rStates[:, 0], rStates[:, 1], rStates[:, 2], 'b', label='Multi Segment')
    ax10.scatter(rStates[0,0], rStates[0,1], rStates[0,2], c='g', marker='o')
    ax10.scatter(rStates[-1,0], rStates[-1,1], rStates[-1,2], c='y', marker='*')

    rotatedStatesNew = np.vstack((rotatedStatesNew, rStates))
    timesFullNew = np.append(timesFullNew, times)

diff = rotatedStatesNew[-1] - rotatedStatesNew[1]
rDiff = np.linalg.norm(diff[0:3])
vDiff = np.linalg.norm(diff[3:6])
print('End Position Difference: '+str(rDiff)+' km')
print('End Velocity Difference: '+str(vDiff)+' km/s')

ax10.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP')
ax10.set_xlabel('X [km]')
ax10.set_ylabel('Y [km]')
ax10.set_zlabel('Z [km]')
ax10.set_title('Moon Centered Rotating Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

ax11.plot(posCRTBP_I_dim[:,0], posCRTBP_I_dim[:,1], posCRTBP_I_dim[:,2], 'r-.', label='CRTBP')
ax11.set_xlabel('X [km]')
ax11.set_ylabel('Y [km]')
ax11.set_zlabel('Z [km]')
ax11.set_title('Moon Centered Inertial Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

fig, (ax12, ax13, ax14) = plt.subplots(3, 1)
ax12.plot(np.arange(len(rotatedStatesNew[1:,2])),rotatedStatesNew[1:,0])
ax12.set_ylabel('x')
ax13.plot(np.arange(len(rotatedStatesNew[1:,2])),rotatedStatesNew[1:,1])
ax13.set_ylabel('y')
ax14.plot(np.arange(len(rotatedStatesNew[1:,2])),rotatedStatesNew[1:,2])
ax14.set_ylabel('z')

rmag = np.linalg.norm(rotatedStatesNew[1:,0:3], axis=1)
plt.figure(15)
plt.plot(np.arange(len(rotatedStatesNew[1:,2])), rmag)
plt.show()

breakpoint()
