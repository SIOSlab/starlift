import numpy as np
import sys
import os
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from scipy.io import loadmat
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(1, 'tools')
import unitConversion
import spiceypy as spice
import multiShooting as ms
#import singleShooting as ss

spice.furnsh("fullForce.txt")

# ** USER INPUTS
fileDir = '/Users/gracegenszler/Documents/Research'     # file path to starlift repo
t_start = Time(61119, format='mjd', scale='utc')        # simulation start time in MJD
mu_star = 0.01215059            # mu value for CRTBP system
orbs = 1                        # number of orbits
N = 9                           # number of patch points
mi = 1000*u.kg                  # kg
Isp = 1500*u.s                  # s
positionTolerance = 0.01        # km
velocityTolerance = 1E-6*orbs   # km/s
sigma = .9999           # Scaling parameter for thrust budget contingency
sigma2 = 1.000          # Scaling parameter for new applied control contingency

# Initial condition in canonical units in rotating frame R [pos, vel, time, U]
# Note: assumes files are .mat
#fileStr = 'L1_Halo'                     # L1 Halo
#fileStr = 'L2_NRHO'                     # L2 NRHO
#fileStr = 'TrajI_1265_MassOptimal'      # L2 Halo
fileStr = 'TrajI_1265_EnergyOptimal'    # L2 Halo
#fileStr = 'L2_Butterfly'                # L2 Butterfly

# Parameters
gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])
g0 = const.g0.value*const.g0.unit       # m/s^2

et_start = spice.str2et(t_start.iso)
rvMoon = spice.spkezr('Moon', et_start, 'J2000', 'None', 'Earth')[0]
Tp_m = spice.oscltx(rvMoon, et_start, gmEarth)[-1]
omega_m = 2*np.pi/Tp_m

mat_data = loadmat(fileDir+'/starlift/orbits/orbitFiles/'+fileStr+'.mat')['TrajI']
posCRTBP_R = mat_data[:,0:3]
velCRTBP_R = mat_data[:,3:6]
timesCRTBP_R = mat_data[:,6]
uT = mat_data[:,7:]

# Convert from nondimensional units to dimensional
posCRTBP_R_dim = unitConversion.convertPos_to_dim(posCRTBP_R - np.array([1-mu_star, 0, 0])).to_value(u.km)
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

# Calculate mass profile
mf = mi*np.exp(-dVCRTBPtot/(Isp*g0))    # kg
m_dim = np.append(mi, mf)               # mass history

# Calculate force profile
Ftmax0 = (max(uT_mag)*mi).to_value(u.mN)
Ft_mag = (uT_mag*m_dim).to_value(u.mN)

# Convert from MCR to MCI
posCRTBP_I_dim = np.zeros((len(etCRTBP_mjd), 3))
for ii in np.arange(len(etCRTBP_mjd)):
    Crv_R2I = spice.sxform('MCR','MCI',etCRTBP_mjd[ii])
    state_R = np.append(posCRTBP_R_dim[ii], velCRTBP_R_dim[ii])
    state_I = Crv_R2I@state_R
    posCRTBP_I_dim[ii,:] = state_I[0:3]

# Define the patch point states
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

# Convert patch points in MCR to MCI frame
initialEphmerisEpoch = spice.str2et(t_start.iso)
times_dim = (taus - t_start).to_value(u.s)
initialEphmerisEpoches = initialEphmerisEpoch + times_dim
Crv_R2I = spice.sxform('MCR','MCI',initialEphmerisEpoches)
initialEphemerisMCI = np.zeros((len(times_dim), 6))
for ii in np.arange(len(times_dim)):
    initialEphemerisMCI[ii,:] = Crv_R2I[ii,:,:] @ posvel[ii]

# Run multi-segment shooting algorithm
correctedInitialEpoches, correctedInitialStates, exitflag, correctedFinalStates = ms.multipleShootingIForced(initialEphmerisEpoches, initialEphemerisMCI, positionTolerance, velocityTolerance, GM, uT_dim.value, etCRTBP_mjd, omega_m)

# Plot in MCI and MCR
inertialStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
rotatedStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
timesTot = np.array([])
vFinal = np.array([])
dVtot = 0
for ii in np.arange(N-1):
    Ts = correctedInitialEpoches[ii:ii+2]
    times, states = ms.statePropFFIForced(Ts, correctedInitialStates[ii,:], GM, uT_dim.value, etCRTBP_mjd, omega_m)
        
    inertialStates = np.vstack((inertialStates, states))
    vFinal = np.append(vFinal, states[-1,3:6])
    
    Crv_I2R = spice.sxform('MCI','MCR',times)
    rStates = np.zeros((len(times), 6))
    for jj in np.arange(len(times)):
        rStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]

    rotatedStates = np.vstack((rotatedStates, rStates[:-1,:]))
    timesTot = np.append(timesTot, times[:-1])
    
rotatedStates = rotatedStates[1:,:]
diff = rotatedStates[-1] - rotatedStates[0]
rDiff = np.linalg.norm(diff[0:3])
vDiff = np.linalg.norm(diff[3:6])
print('End Position Difference: '+str(rDiff)+' km')
print('End Velocity Difference: '+str(vDiff)+' km/s \n')

vFinal = np.reshape(vFinal, (N-1,3))
dVtot = np.linalg.norm(vFinal[:-1,:] - correctedInitialStates[1:-1,3:6], axis=1)
dVpatch = sum(dVtot)
print('Additional dV patches: '+str(dVpatch)+' km/s')

correctedTp = correctedInitialEpoches[-1] - correctedInitialEpoches[0]
TpDiff = correctedTp - (etCRTBP_mjd[-1] - etCRTBP_mjd[0])
print('Orbit period difference: '+str(TpDiff)+' s \n')

filePath = fileDir+'/starlift/orbits/forcedOrbits/'+fileStr
if os.path.isdir(filePath):
    print('Directory exists \n')
else:
    print('Creating directory \n')
    os.makedirs(filePath)

print('Saving state information post multi-segment algorithm \n')
np.savez(filePath+'/InitialFF.npz', ICs = correctedInitialStates, FCs = correctedFinalStates, Ts = correctedInitialEpoches, times = timesTot, statesR = rotatedStates, statesI = inertialStates, Npatch = N, dVpatches = dVtot, startTime = t_start, mu_star = mu_star)

# Calculate orbital parameter differences pre and post multi-segment algorithm
rmagMS = np.linalg.norm(rotatedStates[:,0:3],axis=1)
rmagCRTBP = np.linalg.norm(posCRTBP_R_dim,axis=1)

# Orbital size
rMSmin = min(rmagMS)
rMSmax = max(rmagMS)
rMSlen = rMSmax - rMSmin

rCRTBPmin = min(rmagCRTBP)
rCRTBPmax = max(rmagCRTBP)
rCRTBPlen = rCRTBPmax - rCRTBPmin

rDiff = rMSlen - rCRTBPlen
perDiffR = rDiff/rCRTBPlen*100

# Orbital period
periodMS = ((timesTot[-1]-timesTot[0])*u.s).to('min')
periodCRTBP = (timesCRTBP_d[-1]-timesCRTBP_d[0]).to('min')

periodDiff = periodMS - periodCRTBP
perDiffPeriod = periodDiff/periodCRTBP*100

print('Initial Position Conditons: '+str(posCRTBP_R[0]))
print('Initial Velocity Conditons: '+str(velCRTBP_R[0]))
print('Multi-Segment min: '+str(rMSmin))
print('Multi-Segment max: '+str(rMSmax))
print('Multi-Segment length: '+str(rMSlen))
print('CRTBP min: '+str(rCRTBPmin))
print('CRTBP max: '+str(rCRTBPmax))
print('CRTBP length: '+str(rCRTBPlen))
print('Length difference: '+str(rDiff))
print('Length percent difference: '+str(perDiffR))
print('Multi-Segment period: '+str(periodMS.to('d')))
print('CRTBP period: '+str(periodCRTBP.to('d')))
print('Period difference: '+str(periodDiff))
print('Period percent difference: '+str(perDiffPeriod))

# Calculate delta-v
uT_mag = np.linalg.norm(uT_dim, axis=1)
dVCRTBPtot = cumulative_trapezoid(uT_mag, x=etCRTBP_mjd, axis=0)*u.km/u.s
dVCRTBP = np.append(0*u.km/u.s,dVCRTBPtot)
dVCRTBP = np.diff(dVCRTBP)

dStates = correctedFinalStates - correctedInitialStates[1:,:]
Crv_I2R = spice.sxform('MCI','MCR',correctedInitialEpoches)
dStatesR = np.zeros((len(correctedInitialEpoches)-2, 6))
for kk in np.arange(1,len(correctedInitialEpoches)-1):
    dStatesR[kk-1,:] = Crv_I2R[kk,:,:]@dStates[kk-1,:]

# Convert impulsive burns into continuous thrust
inds = np.array([])
dts = np.array([])
Ups = np.array([])
for ii in np.arange(1,len(correctedInitialEpoches)-1):
    uT_flag = False
    
    # Find the nearest state info in the fpo CRTBP data to the newly calculated patch point
    timeDiff1 = np.abs(etCRTBP_mjd - correctedInitialEpoches[ii])
    ind_ii = np.argwhere(timeDiff1 == min(timeDiff1))[0,0]
    uT_ii = uT_mag[ind_ii]
    m_ii = m_dim[ind_ii]
    
    while not uT_flag:
        # Calculate the maximum force
        F_ii = uT_ii*m_ii
        tmpFt = Ftmax0
        if tmpFt >= Ftmax0:
            Ftmax = Ftmax0*u.mN
        else:
            Ftmax = tmpFt*u.mN
        # Calculate the allowable burn force magnitude
        F_burn = Ftmax*sigma - F_ii
        
        # Convert to the corresponding allowable control, delta-v, and burn time
        uT_burn = F_burn/m_ii
        dV_burn = (np.linalg.norm(dStatesR[ii-1,3:6])*(u.km/u.s))
        dt_burn = dV_burn/uT_burn

        # Check if burn conversion is possible
        if dt_burn < 0:
            print('Warning! Burn not possible! Quit simulation and tune parameters.')
            breakpoint()
            
        # Check the uT in the dt range, centered at patch point
        t_initial = correctedInitialEpoches[ii] - dt_burn.value/2
        t_final = correctedInitialEpoches[ii] + dt_burn.value/2
        
        timeDiff2 = np.abs(etCRTBP_mjd - t_initial)
        timeDiff3 = np.abs(etCRTBP_mjd - t_final)
        ind_i = np.argwhere(timeDiff2 == min(timeDiff2))[0,0]-1
        ind_f = np.argwhere(timeDiff3 == min(timeDiff3))[0,0]+1
        
        uTcheck = np.argwhere(F_burn + uT_mag[ind_i:ind_f]*m_dim[ind_i:ind_f] > Ftmax)
        if len(uTcheck) > 0:
            # rescale time based off of max thrust force in current time interval
            uT_ii = max(uT_mag[ind_i:ind_f])*sigma2

            m_ii = mi*np.exp(-(dV_burn + dVCRTBPtot[ind_i])/(Isp*g0))    # kg
        else:
            # save info to create continuous burn
            inds = np.append(inds, np.array([ind_i, ind_f]))
            dts = np.append(dts, (dt_burn.to('s')).value)
            Ups = np.append(Ups, (uT_burn.to('km/s**2')).value)
            uT_flag = True

    # Report burn duration
    dt_print_s = dt_burn.to('s')
    if dt_print_s.value < 60:
        print('Burn duration for patch '+str(ii)+' is: '+str(dt_print_s))
    elif dt_print_s.value < 60*60:
        print('Burn duration for patch '+str(ii)+' is: '+str(dt_burn.to('min')))
    elif dt_print_s.value < 60*60*24:
        print('Burn duration for patch '+str(ii)+' is: '+str(dt_burn.to('hr')))
    else:
        print('Burn duration for patch '+str(ii)+' is: '+str(dt_burn.to('d')))

# create a new uT array with the newly solved for low thrust burns
# define the start and end times to be added in
t_min = correctedInitialEpoches[1] - dts[0]/2
t_max = correctedInitialEpoches[1] + dts[0]/2

ind_min = np.argwhere(etCRTBP_mjd < t_min)[-1,0]
ind_old = np.argwhere(etCRTBP_mjd > t_max)[0,0]
uT_burn = Ups[0]*dStatesR[0,3:6]/np.linalg.norm(dStatesR[0,3:6])

etCRTBP_mjd_new = etCRTBP_mjd[:ind_min].copy()
uT_new = uT_dim[:ind_min,:].copy().value

# define the control output interpolant for a standardized time scale
points_uT = (etCRTBP_mjd, np.array([0, 1, 2]))
interp_fc_uT = RegularGridInterpolator(points_uT, uT_dim, method='linear')

# initialize the new control output history
if ind_old - ind_min >= 1:
    # if the difference in time indices is greater than 1, interpolate the bounds
    uT_patch = uT_burn + uT_dim[ind_min+1:ind_old,:].value
    uT_patch_i = uT_burn + interp_fc_uT((t_min, np.array([0, 1, 2])))
    uT_patch_f = uT_burn + interp_fc_uT((t_max, np.array([0, 1, 2])))
    uT_patch = np.vstack((np.vstack((uT_patch_i, uT_patch)), uT_patch_f))

    et_patch = np.append(np.append(t_min, etCRTBP_mjd[ind_min+1:ind_old]), t_max)
    etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, et_patch)
else:
    # if difference is 1, addatitively apply the newly solved control
    uT_patch = uT_burn + uT_dim[ind_min,:].value
    uT_patch = np.reshape(np.repeat(uT_patch, repeats=3,axis=0),(3,3)).T
    etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, np.array([t_min, correctedInitialEpoches[1], t_max]))

# repeat the process for the remainder of the control output history
uT_new = np.vstack((uT_new,uT_patch))
for ii in np.arange(1,len(dts)):
    t_min = correctedInitialEpoches[ii+1] - dts[ii]/2
    t_max = correctedInitialEpoches[ii+1] + dts[ii]/2
    
    ind_min = np.argwhere(etCRTBP_mjd < t_min)[-1,0]
    ind_max = np.argwhere(etCRTBP_mjd > t_max)[0,0]
    uT_burn = Ups[ii]*dStatesR[ii,3:6]/np.linalg.norm(dStatesR[ii,3:6])

    etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, etCRTBP_mjd[ind_old:ind_min])
    uT_new = np.vstack((uT_new, uT_dim[ind_old:ind_min,:].value))
    if ind_max - ind_min > 1:
        uT_patch = uT_burn + uT_dim[ind_min:ind_max,:].value
        uT_patch_i = uT_burn + interp_fc_uT((t_min, np.array([0, 1, 2])))
        uT_patch_f = uT_burn + interp_fc_uT((t_max, np.array([0, 1, 2])))
        uT_patch = np.vstack((np.vstack((uT_patch_i, uT_patch)), uT_patch_f))

        et_patch = np.append(np.append(t_min, etCRTBP_mjd[ind_min:ind_max]), t_max)
        etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, et_patch)
    else:
        uT_patch = uT_burn + uT_dim[ind_min,:].value
        uT_patch = np.reshape(np.repeat(uT_patch, repeats=3,axis=0),(3,3)).T
        
        etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, np.array([t_min, correctedInitialEpoches[ii+1], t_max]))
    uT_new = np.vstack((uT_new,uT_patch))

    ind_old = np.argwhere(etCRTBP_mjd > t_max)[0,0]
uT_new = np.vstack((uT_new, uT_dim[ind_old:,:].value))
etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, etCRTBP_mjd[ind_old:])

Ts = np.array([correctedInitialEpoches[0], correctedInitialEpoches[-1]])

# propagate the dynamics with new control only, no multi-segment algorithm to validate the solution
times_final, states_final = ms.statePropFFIForced(Ts, correctedInitialStates[0,:], GM, uT_new, etCRTBP_mjd_new, omega_m)

# rotate from MCI to MCR
states_final_R = np.zeros((len(times_final), 6))
for ii in np.arange(len(times_final)):
    Crv_I2R = spice.sxform('MCI','MCR',times_final[ii])
    states_final_R[ii,:] = Crv_I2R@states_final[ii,:]

# define position interpolant to compare pre and post propulsion system standardization
points_R = (times_final, np.array([0, 1, 2]))
interp_fc_R = RegularGridInterpolator(points_R, states_final_R[:,0:3], method='linear')

# interpolate the states
statesR_interp = np.zeros((len(timesTot), 3))
for ii in np.arange(len(timesTot)):
    statesR_interp[ii,:] = interp_fc_R((timesTot[ii], np.array([0, 1, 2])))

# define orbital quantities
# define total burn time
burnTimeTot = sum(dts)
if burnTimeTot < 60:
    print('Total burn time is: '+str(burnTimeTot)+' s')
elif burnTimeTot < 60*60:
    print('Total burn time is: '+str(burnTimeTot/60)+' min')
elif burnTimeTot < 60*60*24:
    print('Total burn time is: '+str(burnTimeTot/60/60)+' hr')
else:
    print('Total burn time is: '+str(burnTimeTot/60/60/24)+' d')
    
# position magnitude difference
finalR_mag = np.linalg.norm(rotatedStates[:,0:3], axis = 1)
interpR_mag = np.linalg.norm(statesR_interp[:,0:3], axis = 1)

statesR_diff = statesR_interp[:,0:3] - rotatedStates[:,0:3]
diffR_mag = interpR_mag - finalR_mag
print('Max deviation is '+str(max(diffR_mag))+' km')

# delta-v history
uTNew_mag = np.linalg.norm(uT_new, axis=1)*u.km/u.s**2
dVCRTBPtotNew = cumulative_trapezoid(uTNew_mag, x=etCRTBP_mjd_new, axis=0)*u.km/u.s
dVCRTBPNew = np.append(0*u.km/u.s,dVCRTBPtotNew)
dVCRTBPNew = np.diff(dVCRTBPNew)

# mass history
mfNew = mi*np.exp(-dVCRTBPtotNew/(Isp*g0))    # kg
mNew_dim = np.append(mi, mfNew)               # mass history

print('Final mass is : '+str(mNew_dim[-1]))

# define additional plotting quantities
ffNew_time = ((etCRTBP_mjd_new-etCRTBP_mjd_new[0])*u.s).to_value(u.d)
FtMaxPlt = (Ftmax).to_value(u.mN)*np.array([1, 1])

np.savez(filePath+'/plotVariables.npz', posvel = posvel, posCRTBP_R_dim = posCRTBP_R_dim, rStates = rStates, rotatedStates = rotatedStates, dVCRTBP = dVCRTBP, dVtot = dVtot, Ft_mag = Ft_mag, states_final_R = states_final_R, statesR_diff = statesR_diff, uT_mag = uT_mag, uTNew_mag = uTNew_mag, mNew_dim = mNew_dim, m_dim = m_dim, dVCRTBPNew = dVCRTBPNew, FtMaxPlt = FtMaxPlt, etCRTBP_mjd = etCRTBP_mjd, correctedInitialEpoches = correctedInitialEpoches, ffNew_time = ffNew_time, diffR_mag = diffR_mag)
