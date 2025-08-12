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
from scipy.interpolate import RegularGridInterpolator
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

fileDirectory = '/Users/gracegenszler/Documents/Research/starlift/orbits/forcedOrbits/'
#fileStr = 'TrajI_1265'
fileStr = 'TrajExample'
folders = [fileStr+'/']

for ii in np.arange(len(folders)):
    data = np.load(fileDirectory+folders[ii]+'InitialFF.npz', allow_pickle=True)

    t_start = data['startTime']+0
    patchStates = data['ICs']
    patchStatesFinal = data['FCs']
    patchTimes = data['Ts']
    statesR = data['statesR'][1:,:]
    statesI = data['statesI'][1:,:]
    timesFF = data['times']
    N = data['Npatch']
    dVtot = data['dVpatches']
    mu_cstar = 0.01215059

    mat_data = loadmat(fileStr+'.mat')['TrajI']
    posCRTBP_R = mat_data[:,0:3]
    velCRTBP_R = mat_data[:,3:6]
    timesCRTBP_R = mat_data[:,6]
    uT = mat_data[:,7:]

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
#    breakpoint()

    # Electric Propulsion Systems
    ## MET-MAX Electrospray https://www.busek.com/electrospray-thrusters
    #Isp = 2300*u.s       # 850 low, 2300 high
    #Ftmax = 150*u.mN    # 55 low, 150 high

    ## BHT-1500 Hall thruster https://www.busek.com/bht1500
    #Isp = 1710*u.s
    #Ftmax = 101*u.mN

#    # ST-100 Hall thruster https://sets.space/wp-content/themes/sets-space/images/product-sheet/2023/ST-100.pdf
#    Isp = 1500*u.s                  # 1500 low, 1700 high
#    #Ftmax = 50.10969584*u.mN     # 50.10969584 low (more than the data max), 107 high
#    Ftmax = 50.125*u.mN

    ## SPS-100 Hall thruster https://sets.space/wp-content/themes/sets-space/images/product-sheet/2024/SPS-100.pdf
    #Isp = 1800*u.s
    #Ftmax = 90*u.mN
    
    # Chemical Propulsion Systems
#    # B20 Thruster - Green Chemical https://catalog.orbitaltransports.com/b20-thruster-green-propulsion/
#    Isp = 285*u.s
#    Ftmax = 19.3*u.N    # 6.6N low, 19.3N high, temperature dependent
    
#    # 200N HPGP Thruster https://satsearch.co/products/ecaps-200n-hpgp-thruster
#    Isp = 206*u.s       # 206-234s RCS mode, 243-255 delta-v mode
#    Ftmax = 50*u.N      # 50-200N RCS mode, 55-220N delta-v mode

    # 50N GPGP Thruster https://satsearch.co/products/ecaps-50n-hpgp-thruster
#    Isp = 243*u.s       # 243-255s
#    Ftmax = 12.5*u.N    # 12.5-50N
    
    ## Nuclear Thermal Propulsion https://www1.grc.nasa.gov/research-and-engineering/nuclear-thermal-propulsion-systems/
    Isp = 900*u.s
    Ftmax = 44482.22*u.N   # 10-100kblf
    
    # Default
    Isp = 200*u.s
    mi = 1000*u.kg                          # kg
    Ftmax = (max(uT_mag)*mi)                # mN

    
    g0 = const.g0.value*const.g0.unit       # m/s^2
    mf = mi*np.exp(-dVCRTBPtot/(Isp*g0))    # kg
    m_dim = np.append(mi, mf)               # mass history
    breakpoint()
    # Convert impulsive burns into continuous thrust
    sigma = 1.000
    inds = np.array([])
    dts = np.array([])
    Ups = np.array([])
    for ii in np.arange(1,len(patchTimes)-2):
        uT_flag = False
        timeDiff1 = np.abs(etCRTBP_mjd - patchTimes[ii])
        
        ind_ii = np.argwhere(timeDiff1 == min(timeDiff1))[0,0]
        uT_ii = uT_mag[ind_ii]
        while not uT_flag:
            # get uT at patch point and estimate dt for a burn with a buffer
            dt_ii = dVtot[ii]*u.km/u.s/(Ftmax/mf[ii]*sigma - uT_ii)
            if dt_ii < 0:
                breakpoint()
            # check the uT in the dt range, centered at patch point
            
            t_initial = patchTimes[ii] - dt_ii.value/2
            t_final = patchTimes[ii] + dt_ii.value/2
            
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
                dts = np.append(dts, (dt_ii.to('s')).value)
                Ups = np.append(Ups, (uT_ii.to('km/s**2')).value)
                uT_flag = True
            
            dt_print_s = dt_ii.to('s')
            if dt_print_s.value < 60:
                print('Burn duration for patch '+str(ii)+' is: '+str(dt_print_s))
            elif dt_print_s.value < 60*60:
                print('Burn duration for patch '+str(ii)+' is: '+str(dt_ii.to('min')))
            elif dt_print_s.value < 60*60*24:
                print('Burn duration for patch '+str(ii)+' is: '+str(dt_ii.to('hr')))
            else:
                print('Burn duration for patch '+str(ii)+' is: '+str(dt_ii.to('d')))

    # create a new uT array from scratch
    dStates = patchStatesFinal[:,3:6] - patchStates[:-1,3:6]

    # take all the times before the start of the burn
    # add in the times during the burn
    # add in the times after the burn and before the next burn
    t_min = patchTimes[1] - dts[0]/2
    t_max = patchTimes[1] + dts[0]/2
    ind_min = np.argwhere(etCRTBP_mjd < t_min)[-1,0]
    ind_old = np.argwhere(etCRTBP_mjd > t_max)[0,0]
    uT_burn = Ups[0]*dStates[0]/np.linalg.norm(dStates[0])

    etCRTBP_mjd_new = etCRTBP_mjd[:ind_min].copy()
    uT_new = uT_dim[:ind_min,:].copy().value

    points_uT = (etCRTBP_mjd, np.array([0, 1, 2]))
    interp_fc_uT = RegularGridInterpolator(points_uT, uT_dim, method='linear')

    if ind_old - ind_min >= 1:
        uT_patch = uT_burn + uT_dim[ind_min+1:ind_old,:].value
        uT_patch_i = uT_burn + interp_fc_uT((t_min, np.array([0, 1, 2])))
        uT_patch_f = uT_burn + interp_fc_uT((t_max, np.array([0, 1, 2])))
        uT_patch = np.vstack((np.vstack((uT_patch_i, uT_patch)), uT_patch_f))

        et_patch = np.append(np.append(t_min, etCRTBP_mjd[ind_min+1:ind_old]), t_max)
        etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, et_patch)
    else:
        uT_patch = uT_burn + uT_dim[ind_min,:].value
        uT_patch = np.reshape(np.repeat(uT_patch, repeats=3,axis=0),(3,3)).T
        etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, np.array([t_min, patchTimes[1], t_max]))

    uT_new = np.vstack((uT_new,uT_patch))
    for ii in np.arange(1,len(dts)):
        t_min = patchTimes[ii+1] - dts[ii]/2
        t_max = patchTimes[ii+1] + dts[ii]/2
        ind_min = np.argwhere(etCRTBP_mjd < t_min)[-1,0]
        ind_max = np.argwhere(etCRTBP_mjd > t_max)[0,0]
        uT_burn = Ups[ii]*dStates[ii]/np.linalg.norm(dStates[ii])

        etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, etCRTBP_mjd[ind_old:ind_min])
        uT_new = np.vstack((uT_new, uT_dim[ind_old:ind_min,:].value))
        if ind_max - ind_min >= 1:
            uT_patch = uT_burn + uT_dim[ind_min+1:ind_max,:].value
            uT_patch_i = uT_burn + interp_fc_uT((t_min, np.array([0, 1, 2])))
            uT_patch_f = uT_burn + interp_fc_uT((t_max, np.array([0, 1, 2])))
            uT_patch = np.vstack((np.vstack((uT_patch_i, uT_patch)), uT_patch_f))

            et_patch = np.append(np.append(t_min, etCRTBP_mjd[ind_min+1:ind_max]), t_max)
            etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, et_patch)
        else:
            uT_patch = uT_burn + uT_dim[ind_min,:].value
            uT_patch = np.reshape(np.repeat(uT_patch, repeats=3,axis=0),(3,3)).T
            
            etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, np.array([t_min, patchTimes[ii+1], t_max]))
        uT_new = np.vstack((uT_new,uT_patch))
        tmp1 = np.shape(etCRTBP_mjd_new)
        tmp2 = np.shape(uT_new)
        
        ind_old = np.argwhere(etCRTBP_mjd > t_max)[0,0]
    uT_new = np.vstack((uT_new, uT_dim[ind_old:,:].value))
    etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, etCRTBP_mjd[ind_old:])

    Ts = np.array([patchTimes[0], patchTimes[-1]])
    #breakpoint()
    times_final, states_final = ms.statePropFFIForced(Ts, patchStates[0,:], GM, uT_new, etCRTBP_mjd_new)

    states_final_R = np.zeros((len(times_final), 6))
    for ii in np.arange(len(times_final)):
        Crv_I2R = spice.sxform('MCI','MCR',times_final[ii])
        states_final_R[ii,:] = Crv_I2R@states_final[ii,:]
        
    ax1 = plt.figure().add_subplot(projection='3d')
    ax1.plot(statesR[:, 0], statesR[:, 1], statesR[:, 2], 'b', label='Multi Segment')
    ax1.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP')
    ax1.plot(states_final_R[:,0], states_final_R[:,1], states_final_R[:,2], 'g-.', label='Final Trajectory')
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_zlabel('Z [km]')
    ax1.set_title('Moon Centered Rotating Frame')

    points_R = (times_final, np.array([0, 1, 2]))
    interp_fc_R = RegularGridInterpolator(points_R, states_final_R[:,0:3], method='linear')

    statesR_interp = np.zeros((len(timesFF), 3))
    for ii in np.arange(len(timesFF)):
        statesR_interp[ii,:] = interp_fc_R((timesFF[ii], np.array([0, 1, 2])))

    finalR_mag = np.linalg.norm(statesR[:,0:3], axis = 1)
    interpR_mag = np.linalg.norm(statesR_interp[:,0:3], axis = 1)

    statesR_diff = statesR_interp[:,0:3] - statesR[:,0:3]
    diffR_mag = interpR_mag - finalR_mag
    print('Max deviation is '+str(max(diffR_mag))+' km')

    burnTimeTot = sum(dts)
    if burnTimeTot < 60:
        print('Total burn time is: '+str(burnTimeTot)+' s')
    elif burnTimeTot < 60*60:
        print('Total burn time is: '+str(burnTimeTot/60)+' min')
    elif burnTimeTot < 60*60*24:
        print('Total burn time is: '+str(burnTimeTot/60/60)+' hr')
    else:
        print('Total burn time is: '+str(burnTimeTot/60/60/24)+' d')

    plot_time = (timesFF - timesFF[0])/60/60/24
    fig, (ax2, ax3, ax4, ax5) = plt.subplots(4, 1)
    ax2.plot(plot_time, abs(statesR_diff[:,0]))
    ax2.set_ylabel('x [km]')
    ax2.set_title('Absolute value differences')
    ax3.plot(plot_time, abs(statesR_diff[:,1]))
    ax3.set_ylabel('y [km]')
    ax4.plot(plot_time, abs(statesR_diff[:,2]))
    ax4.set_ylabel('z [km]')
    ax5.plot(plot_time, abs(diffR_mag))
    ax5.set_ylabel('magnitude [km]')
    ax5.set_xlabel('time [days]')

    plt.show()
    breakpoint()

