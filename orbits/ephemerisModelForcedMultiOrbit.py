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
#fileStr = 'TrajExample'     # pole sitter
fileStr = 'TrajI_1265'      # mass optimal
mat_data = loadmat(fileStr+'.mat')['TrajI']
posCRTBP_R = mat_data[:,0:3]
velCRTBP_R = mat_data[:,3:6]
timesCRTBP_R = mat_data[:,6]
uT = mat_data[:,7:]
mu_cstar = 0.01215059

# Convert from nondimensional units to dimensional
posCRTBP_R_dim = unitConversion.convertPos_to_dim(posCRTBP_R - np.array([1-mu_cstar, 0, 0])).to_value(u.km)
velCRTBP_R_dim = unitConversion.convertVel_to_dim(velCRTBP_R).to_value(u.km/u.s)
timesCRTBP_d = unitConversion.convertTime_to_dim(timesCRTBP_R).to('d')
timesCRTBP_mjd = t_start + timesCRTBP_d
etCRTBP_mjd = spice.str2et(timesCRTBP_mjd.iso)
uT_dim = unitConversion.convertAcc_to_dim(uT).to('km/s**2')

# Calculate delta-v
uT_mag = np.linalg.norm(uT_dim, axis=1)
dVCRTBPtot = cumulative_trapezoid(uT_mag, x=etCRTBP_mjd, axis=0)*u.km/u.s
dVCRTBP = np.append(0*u.km/u.s,dVCRTBPtot)
dVCRTBP = np.diff(dVCRTBP)

posCRTBP_I_dim = np.zeros((len(etCRTBP_mjd), 3))
for ii in np.arange(len(etCRTBP_mjd)):
    Crv_R2I = spice.sxform('MCR','MCI',etCRTBP_mjd[ii])
    state_R = np.append(posCRTBP_R_dim[ii], velCRTBP_R_dim[ii])
    state_I = Crv_R2I@state_R
    posCRTBP_I_dim[ii,:] = state_I[0:3]

filepath = '/Users/gracegenszler/Documents/Research/starlift/orbits/forcedOrbits/twoOrbits/'+fileStr
if os.path.isdir(filepath):
    print('directory exists')
else:
    os.makedirs(filepath)
    
positionTolerance = 0.01    # km
velocityTolerance = 1E-2   #1E-6*orbs  # km/s
    
Nmax = 9
Ns = np.arange(Nmax, 7, -1)
patchCtr = 0
plusCtr = 0
minPatch = False
exitflag = 1
minPatchN = np.array([Nmax+1])
while not minPatch:
    if exitflag == 1:
        N = Ns[patchCtr]
        plusCtr = 0
    else:
        print('Solution not found. Increasing number of patch points by 1.')
        N = N + 1
        plusCtr = plusCtr + 1

    posvel, taus = ms.getPatches(N, timesCRTBP_d, timesCRTBP_mjd, posCRTBP_R_dim, velCRTBP_R_dim)

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
        initialEphemerisMCI[ii,:] = Crv_R2I[ii,:,:]@posvel[ii]

    correctedInitialEpoches, correctedInitialStates, exitflag, correctedFinalStates = ms.multipleShootingIForced(initialEphmerisEpoches, initialEphemerisMCI, positionTolerance, velocityTolerance, GM, uT_dim.value, etCRTBP_mjd, omega_m)
    
    if exitflag == 1:
        if np.any(Ns == N):
            patchCtr = patchCtr + 1
            minPatchN = np.append(minPatchN, len(correctedInitialEpoches))
            
            if len(minPatchN) != len(np.unique(minPatchN)):
                patchCtr = patchCtr - 1
            
        if N < Ns[-2] and N >= Ns[-1]:
            minPatch = True
        elif plusCtr <= (Ns[patchCtr-1] - Ns[patchCtr]) and plusCtr > 0:
            minPatch = True

        inertialStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        rotatedStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        timesFull = np.array([])
        for ii in np.arange(N-1):
            Ts = correctedInitialEpoches[ii:ii+2]
            times, states = ms.statePropFFI(Ts, correctedInitialStates[ii,:], GM)
                
            inertialStates = np.vstack((inertialStates, states))
            
            Crv_I2R = spice.sxform('MCI','MCR',times)
            rStates = np.zeros((len(times), 6))
            for jj in np.arange(len(times)):
                rStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]
            
            rotatedStates = np.vstack((rotatedStates, rStates))
            timesFull = np.append(timesFull, times)
                        
        np.savez(filepath+'/InitialFF_N'+str(N)+'.npz', ICs = correctedInitialStates, FCs = correctedFinalStates, Ts = correctedInitialEpoches, times = timesFull, statesR = rotatedStates, statesI = inertialStates, Npatch = N, startTime = t_start, mu_star = mu_cstar)

# solve for one orbit using code above
# calculate patch points for second orbit
N2 = 9
t_startStr = spice.et2utc(correctedInitialEpoches[-1], 'ISOC', 23, 24)
t_start2 = Time(t_startStr, format='isot', scale='utc')
timesCRTBP_mjd2 = t_start2 + timesCRTBP_d
etCRTBP_mjd2 = spice.str2et(timesCRTBP_mjd2.iso)

posvelNew, tausNew = ms.getPatches(N2, timesCRTBP_d, timesCRTBP_mjd2, posCRTBP_R_dim, velCRTBP_R_dim)

# ignore the first new patch point and use the final state from the section above
posvel2 = np.vstack((correctedFinalStates[-1,:], posvelNew[1:,:]))
taus2 = Time(np.append(t_start2.mjd, tausNew[1:].value), format='mjd', scale='utc')

# solve again
# Convert to MCI frame
initialEphmerisEpoch2 = spice.str2et(t_start2.iso)
times_dim2 = ((taus2.value - t_start2.mjd)*u.d).to_value(u.s)
initialEphmerisEpoches2 = initialEphmerisEpoch2 + times_dim2
Crv_R2I = spice.sxform('MCR','MCI',initialEphmerisEpoches2)
initialEphemerisMCI2 = np.zeros((len(times_dim2), 6))
for ii in np.arange(len(times_dim2)):
    initialEphemerisMCI2[ii,:] = Crv_R2I[ii,:,:] @ posvel2[ii]
        
correctedInitialEpoches2, correctedInitialStates2, exitflag, correctedFinalStates2 = ms.multipleShootingIForced(initialEphmerisEpoches2, initialEphemerisMCI2, positionTolerance, velocityTolerance, GM, uT_dim.value, etCRTBP_mjd2, omega_m)

inertialStates2 = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
rotatedStates2 = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
timesFull2 = np.array([])
for ii in np.arange(N2-1):
    Ts = correctedInitialEpoches2[ii:ii+2]
    times, states = ms.statePropFFI(Ts, correctedInitialStates2[ii,:], GM)
        
    inertialStates2 = np.vstack((inertialStates2, states))
    
    Crv_I2R = spice.sxform('MCI','MCR',times)
    rStates = np.zeros((len(times), 6))
    for jj in np.arange(len(times)):
        rStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]
    
    rotatedStates2 = np.vstack((rotatedStates2, rStates))
    timesFull2 = np.append(timesFull2, times)
            
    np.savez(filepath+'/SecondFF_N'+str(N2)+'.npz', ICs = correctedInitialStates2, FCs = correctedFinalStates2, Ts = correctedInitialEpoches2, times = timesFull2, statesR = rotatedStates2, statesI = inertialStates2, Npatch = N2, startTime = t_start2, mu_star = mu_cstar)
        
breakpoint()
