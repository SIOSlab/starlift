import numpy as np
import sys
import os
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from scipy.io import loadmat
from scipy.integrate import cumulative_trapezoid
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
fileStr = 'TrajI_1265'
mat_data = loadmat(fileStr+'.mat')['TrajI']
mat_data2 = loadmat('TrajExample.mat')['TrajI']
posCRTBP_R = mat_data[:,0:3]
velCRTBP_R = mat_data[:,3:6]
timesCRTBP_R = mat_data[:,6]
uT = mat_data[:,7:]
mu_cstar = 0.01215059

posCRTBP_R2 = mat_data2[:,0:3]
velCRTBP_R2 = mat_data2[:,3:6]
timesCRTBP_R2 = mat_data2[:,6]
uT2 = mat_data2[:,7:]

# Convert from nondimensional units to dimensional
posCRTBP_R_dim = unitConversion.convertPos_to_dim(posCRTBP_R - np.array([1-mu_cstar, 0, 0])).to_value(u.km)
velCRTBP_R_dim = unitConversion.convertVel_to_dim(velCRTBP_R).to_value(u.km/u.s)
timesCRTBP_d = unitConversion.convertTime_to_dim(timesCRTBP_R).to('d')
timesCRTBP_mjd = t_start + timesCRTBP_d
etCRTBP_mjd = spice.str2et(timesCRTBP_mjd.iso)
uT_dim = unitConversion.convertAcc_to_dim(uT).to('km/s**2')

posCRTBP_R_dim2 = unitConversion.convertPos_to_dim(posCRTBP_R2 - np.array([1-mu_cstar, 0, 0])).to_value(u.km)
velCRTBP_R_dim2 = unitConversion.convertVel_to_dim(velCRTBP_R2).to_value(u.km/u.s)
timesCRTBP_d2 = unitConversion.convertTime_to_dim(timesCRTBP_R2).to('d')
timesCRTBP_mjd2 = t_start + timesCRTBP_d2
etCRTBP_mjd2 = spice.str2et(timesCRTBP_mjd2.iso)
uT_dim2 = unitConversion.convertAcc_to_dim(uT2).to('km/s**2')

# Calculate delta-v
uT_mag = np.linalg.norm(uT_dim, axis=1)
dVCRTBPtot = cumulative_trapezoid(uT_mag, x=etCRTBP_mjd, axis=0)*u.km/u.s
dVCRTBP = np.append(0*u.km/u.s,dVCRTBPtot)
dVCRTBP = np.diff(dVCRTBP)

uT_mag2 = np.linalg.norm(uT_dim2, axis=1)
dVCRTBPtot2 = cumulative_trapezoid(uT_mag2, x=etCRTBP_mjd2, axis=0)*u.km/u.s
dVCRTBP2 = np.append(0*u.km/u.s,dVCRTBPtot2)
dVCRTBP2 = np.diff(dVCRTBP2)

# Calculate mass profile
Isp = 1500*u.s                                          # seconds
mi = 1000*u.kg                                          # kg
Ftmax = (max(uT_mag)*mi)*1.01                       # mN
g0 = const.g0.value*const.g0.unit                     # m/s^2
mf = mi*np.exp(-dVCRTBPtot/(Isp*g0))   # kg
m_dim = np.append(mi, mf)                               # mass history

Ftmax2 = (max(uT_mag2)*mi)                       # mN
mf2 = mi*np.exp(-dVCRTBPtot2/(Isp*g0))   # kg
m_dim2 = np.append(mi, mf2)                               # mass history

# Calculate the force profile
Ft_mag = (uT_mag*m_dim).to_value(u.mN)
Ft_mag2 = (uT_mag2*m_dim2).to_value(u.mN)

posCRTBP_I_dim = np.zeros((len(etCRTBP_mjd), 3))
for ii in np.arange(len(etCRTBP_mjd)):
    Crv_R2I = spice.sxform('MCR','MCI',etCRTBP_mjd[ii])
    state_R = np.append(posCRTBP_R_dim[ii], velCRTBP_R_dim[ii])
    state_I = Crv_R2I@state_R
    posCRTBP_I_dim[ii,:] = state_I[0:3]

N = 9
dt_int = (timesCRTBP_d[-1]-timesCRTBP_d[0])/(N-1)
taus = Time(np.zeros(N), format='mjd', scale='utc')
posvel = np.array([])
for ii in np.arange(N):
    time_i = ii*dt_int

    # find the index
    difference_array_i = np.absolute(timesCRTBP_d-time_i).value
    index_i = difference_array_i.argmin()
    
    # index the time, position, and velocity
    taus[ii] = timesCRTBP_mjd[index_i]
    pos_R = posCRTBP_R_dim[index_i,:]
    vel_R = velCRTBP_R_dim[index_i,:]
    
    # package the state
    state_R = np.append(pos_R, vel_R)
    posvel = np.append(posvel, state_R)
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
velocityTolerance = 1E-6*orbs  # km/s

correctedInitialEpoches, correctedInitialStates, exitflag, correctedFinalStates = ms.multipleShootingIForced(initialEphmerisEpoches, initialEphemerisMCI, positionTolerance, velocityTolerance, GM, uT_dim.value, etCRTBP_mjd)

# Plot in MCI and MCR
ax10 = plt.figure().add_subplot(projection='3d')
ax11 = plt.figure().add_subplot(projection='3d')
inertialStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
rotatedStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
timesTot = np.array([])
vFinal = np.array([])
dVtot = 0
for ii in np.arange(N-1):
    Ts = correctedInitialEpoches[ii:ii+2]
    times, states = ms.statePropFFIForced(Ts, correctedInitialStates[ii,:], GM, uT_dim.value, etCRTBP_mjd)
        
    inertialStates = np.vstack((inertialStates, states))
    vFinal = np.append(vFinal, states[-1,3:6])
    
#    ax11.plot(states[:, 0], states[:, 1], states[:, 2], 'b', label='Multi Segment')
    ax11.scatter(states[0,0], states[0,1], states[0,2], c='g', marker='o')
    ax11.scatter(states[-1,0], states[-1,1], states[-1,2], c='y', marker='*')
    
    Crv_I2R = spice.sxform('MCI','MCR',times)
    rStates = np.zeros((len(times), 6))
    for jj in np.arange(len(times)):
        rStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]
        
#    ax10.plot(rStates[:, 0], rStates[:, 1], rStates[:, 2], 'b', label='Multi Segment')
    ax10.scatter(rStates[0,0], rStates[0,1], rStates[0,2], c='g', marker='o')
    ax10.scatter(rStates[-1,0], rStates[-1,1], rStates[-1,2], c='y', marker='*')

    rotatedStates = np.vstack((rotatedStates, rStates[:-1,:]))
    timesTot = np.append(timesTot, times[:-1])
    breakpoint()

diff = rotatedStates[-1] - rotatedStates[1]
rDiff = np.linalg.norm(diff[0:3])
vDiff = np.linalg.norm(diff[3:6])
print('End Position Difference: '+str(rDiff)+' km')
print('End Velocity Difference: '+str(vDiff)+' km/s')

vFinal = np.reshape(vFinal, (N-1,3))
dVtot = np.linalg.norm(vFinal[:-1,:] - correctedInitialStates[1:-1,3:6], axis=1)
dVpatch = sum(dVtot)
print('Additional dV patches: '+str(dVpatch)+' km/s')

correctedTp = correctedInitialEpoches[-1] - correctedInitialEpoches[0]
TpDiff = correctedTp - (etCRTBP_mjd[-1] - etCRTBP_mjd[0])
print('Orbit period difference: '+str(TpDiff)+' s')

ax10.plot(rotatedStates[:, 0], rotatedStates[:, 1], rotatedStates[:, 2], 'b', label='Multi Segment')
ax10.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP')
ax10.set_xlabel('X [km]')
ax10.set_ylabel('Y [km]')
ax10.set_zlabel('Z [km]')
ax10.set_title('Moon Centered Rotating Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

ax11.plot(inertialStates[:, 0], inertialStates[:, 1], inertialStates[:, 2], 'b', label='Multi Segment')
ax11.plot(posCRTBP_I_dim[:,0], posCRTBP_I_dim[:,1], posCRTBP_I_dim[:,2], 'r-.', label='CRTBP')
ax11.set_xlabel('X [km]')
ax11.set_ylabel('Y [km]')
ax11.set_zlabel('Z [km]')
ax11.set_title('Moon Centered Inertial Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

fig, (ax5, ax6, ax7) = plt.subplots(3, 1)
ax5.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,0])
ax5.set_ylabel('x')
ax6.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,1])
ax6.set_ylabel('y')
ax7.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,2])
ax7.set_ylabel('z')

plt.figure(8)
plt.plot(etCRTBP_mjd[:-1], dVCRTBP, label='Original Thrust Profile')
plt.scatter(correctedInitialEpoches[1:-1], dVtot, c='r', marker='*', zorder=3, label='Patch Point Burns')
plt.yscale('log')
plt.xlabel('Time since epoch [s]')
plt.ylabel('Delta-v [km/s]')
plt.title('Delta-v History')
plt.legend()

plt.figure(9)
plt.plot(etCRTBP_mjd, Ft_mag)
plt.yscale('log')
plt.xlabel('Time since epoch [s]')
plt.ylabel('Control Force [mN]')
plt.title('Thrust History')
plt.show()

filepath = '/Users/gracegenszler/Documents/Research/starlift/orbits/forcedOrbits/'+fileStr
if os.path.isdir(filepath):
    print('directory exists')
else:
    os.makedirs(filepath)
    
np.savez(filepath+'/InitialFF.npz', ICs = correctedInitialStates, FCs = correctedFinalStates, Ts = correctedInitialEpoches, times = timesTot, statesR = rotatedStates, statesI = inertialStates, Npatch = N, dVpatches = dVtot, startTime = t_start, mu_star = mu_cstar)
breakpoint()


# save things
# Convert impulsive burns into continuous thrust
sigma = 1
inds = np.array([])
dts = np.array([])
Ups = np.array([])
for ii in np.arange(1,len(correctedInitialEpoches)-2):
    uT_flag = False
    timeDiff1 = np.abs(etCRTBP_mjd - correctedInitialEpoches[ii])
    
    ind_ii = np.argwhere(timeDiff1 == min(timeDiff1))[0,0]
    uT_ii = uT_mag[ind_ii]
    while not uT_flag:
        # get uT at patch point and estimate dt for a burn with a buffer
        dt_ii = dVtot[ii]*u.km/u.s/(Ftmax/mf[ii]*sigma - uT_ii)
        if dt_ii < 0:
            breakpoint()
        # check the uT in the dt range, centered at patch point
        
        t_initial = correctedInitialEpoches[ii] - dt_ii.value/2
        t_final = correctedInitialEpoches[ii] + dt_ii.value/2
        
        timeDiff2 = np.abs(etCRTBP_mjd - t_initial)
        timeDiff3 = np.abs(etCRTBP_mjd - t_final)
        ind_i = np.argwhere(timeDiff2 == min(timeDiff2))[0,0]
        ind_f = np.argwhere(timeDiff3 == min(timeDiff3))[0,0]
        
        if ind_i != ind_f:
            uTcheck = np.argwhere(uT_mag[ind_i] + dVtot[ii]*u.km/u.s/dt_ii > Ftmax/mf[ii])
        else:
            uTcheck = np.array([])

        if np.any(uTcheck):
            # rescale time based off of max thrust force in current time interval
            uT_ii = max(uT_mag[ind_i:ind_f])
        else:
            # save info to create continuous burn
            inds = np.append(inds, np.array([ind_i, ind_f]))
            dts = np.append(dts, dt_ii.value)
            Ups = np.append(Ups, uT_ii.value)
            uT_flag = True
           
        print(ind_i)
        print(ind_f)
        print(dt_ii)

# create a new uT array from scratch
dStates = correctedFinalStates[:,3:6] - correctedInitialStates[:-1,3:6]

t_min = correctedInitialEpoches[1] - dts[0]/2
t_max = correctedInitialEpoches[1] + dts[0]/2
ind_min = np.argwhere(etCRTBP_mjd < t_min)[-1,0]
ind_old = np.argwhere(etCRTBP_mjd > t_max)[0,0]
uT_new = uT_dim[:,:ind_min].copy().value
uT_burn = Ups[0]*dStates[0]/np.linalg.norm(dStates[0])
etCRTBP_mjd_new = etCRTBP_mjd[:ind_min].copy()

if ind_min != ind_old:
    uT_patch = uT_burn + uT_dim[ind_min+1:ind_old,:][0,:].value
else:
    uT_patch = uT_burn + uT_dim[ind_min,:].value
    
uT_patch = np.reshape(np.repeat(uT_patch, repeats=3,axis=0),(3,3)).T
uT_new = np.vstack((uT_new,uT_patch))
etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, np.array([t_min, correctedInitialEpoches[1], t_max]))
    
for ii in np.arange(1,len(dts)):
    t_min = correctedInitialEpoches[ii+1] - dts[ii]/2
    t_max = correctedInitialEpoches[ii+1] + dts[ii]/2
    ind_min = np.argwhere(etCRTBP_mjd < t_min)[-1,0]
    ind_max = np.argwhere(etCRTBP_mjd > t_max)[0,0]
    uT_burn = Ups[ii]*dStates[ii]/np.linalg.norm(dStates[ii])
    
    if ind_max - ind_min > 1:
        uT_patch = uT_burn + uT_dim[ind_min+1:ind_max,:][0,:].value
    elif ind_max - ind_min == 1:
        uT_patch = uT_burn + (uT_dim[ind_min,:]+uT_dim[ind_max,:]).value/2
    else:
        uT_patch = uT_burn + uT_dim[ind_min,:].value
    uT_patch = np.reshape(np.repeat(uT_patch, repeats=3,axis=0),(3,3)).T
    uT_new = np.vstack((uT_new,uT_patch))
    etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, np.array([t_min, correctedInitialEpoches[ii+1], t_max]))
    
    ind_old = np.argwhere(etCRTBP_mjd > t_max)[0,0]
uT_new = np.vstack((uT_new, uT_dim[ind_old:,:].value))
etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, etCRTBP_mjd[ind_old:])

Ts = np.array([correctedInitialEpoches[0], correctedInitialEpoches[-1]])
times_final, states_final = ms.statePropFFIForced(Ts, correctedInitialStates[0,:], GM, uT_new, etCRTBP_mjd_new)

states_final_R = np.zeros((len(times_final), 6))
for ii in np.arange(len(times_final)):
    Crv_I2R = spice.sxform('MCI','MCR',times_final[ii])
    states_final_R[ii,:] = Crv_I2R@states_final[ii,:]
    

ax12 = plt.figure().add_subplot(projection='3d')
ax12.plot(rotatedStates[:, 0], rotatedStates[:, 1], rotatedStates[:, 2], 'b', label='Multi Segment')
ax12.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP')
ax12.plot(states_final_R[:,0], states_final_R[:,1], states_final_R[:,2], 'g', label='Final Trajectory')
ax12.set_xlabel('X [km]')
ax12.set_ylabel('Y [km]')
ax12.set_zlabel('Z [km]')
ax12.set_title('Patch Points in Moon Centered Rotating Frame')
plt.show()
breakpoint()


# separate into two files
# fix the repeat in added data
