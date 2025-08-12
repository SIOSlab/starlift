import sys
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
import astropy.units as u
sys.path.insert(1, 'tools')
import unitConversion

def jacobiConstFFR(state_scM, time, GM):

    GM_M = GM[0]
    GM_E = GM[1]
    GM_S = GM[2]
    
    state_baryM = spice.spkezr('Earth Moon Barycenter', time, 'J2000', 'None', 'Moon')[0]
    r_Moon = spice.spkpos('Moon', time, 'J2000', 'None', 'Moon')[0]
    r_Earth = spice.spkpos('Earth', time, 'J2000', 'None', 'Moon')[0]
    r_Sun = spice.spkpos('Sun', time, 'J2000', 'None', 'Moon')[0]
    
    Crv_I2R = (spice.sxform('MCI','MCR', time))
    state_baryM = Crv_I2R@state_baryM
    r_Moon = Crv_I2R[0:3,0:3]@r_Moon
    r_Earth = Crv_I2R[0:3,0:3]@r_Earth
    r_Sun = Crv_I2R[0:3,0:3]@r_Sun
    
    state_scBary = state_scM - state_baryM
    
    pos_Mbary = -state_baryM[0:3]
    pos_Ebary = r_Earth - state_baryM[0:3]
    pos_Sbary = r_Sun - state_baryM[0:3]
    
    s1 = (pos_Sbary[0] - state_scBary[0])**2
    s5 = (pos_Sbary[1] - state_scBary[1])**2
    s4 = (pos_Sbary[2] - state_scBary[2])**2
        
    s2 = (pos_Mbary[0] - state_scBary[0])**2
    s7 = (pos_Mbary[1] - state_scBary[1])**2
    s6 = (pos_Mbary[2] - state_scBary[2])**2
    
    s3 = (pos_Ebary[0] - state_scBary[0])**2
    s9 = (pos_Ebary[1] - state_scBary[1])**2
    s8 = (pos_Ebary[2] - state_scBary[2])**2
    
    s10 = (pos_Ebary[0] - pos_Mbary[0])**2
    s11 = (pos_Ebary[1] - pos_Mbary[1])**2
    s12 = (pos_Ebary[2] - pos_Mbary[2])**2
    
    s13 = (pos_Mbary[0] - pos_Sbary[0])**2
    s14 = (pos_Mbary[1] - pos_Sbary[1])**2
    s15 = (pos_Mbary[2] - pos_Sbary[2])**2
    
    
    t1 = GM_E/np.sqrt(s9 + s8 + s3)
    t2 = GM_M/np.sqrt(s7 + s6 + s2)
    t3 = GM_S/np.sqrt(s5 + s4 + s1)
    
    t4 = GM_E*state_scBary[0]*np.sqrt(s10)/(s10 + s11 + s12)**(3/2)
    t5 = GM_S*state_scBary[0]*np.sqrt(s13)/(s13 + s14 + s15)**(3/2)

    t6 = GM_E*state_scBary[1]*np.sqrt(s11)/(s10 + s11 + s12)**(3/2)
    t7 = GM_S*state_scBary[1]*np.sqrt(s14)/(s13 + s14 + s15)**(3/2)
    
    t8 = GM_E*state_scBary[2]*np.sqrt(s12)/(s10 + s11 + s12)**(3/2)
    t9 = GM_S*state_scBary[2]*np.sqrt(s15)/(s13 + s14 + s15)**(3/2)
    
    Ux = -(-t1 - t2 - t3 - t4 + t5)
    Uy = -(-t1 - t2 - t3 - t6 + t7)
    Uz = -(-t1 - t2 - t3 - t8 + t9)
    
    KE1 = np.dot(state_scBary[3:6], state_scBary[3:6])/2

    C1 = KE1 + Ux + Uy + Uz
#    C2 = unitConversion.convertVel_to_canonical(C1*u.km/u.s)
#    breakpoint()

    return C1

def jacobiConstCRTBPR(pos, vel, mu_star):

    r_Mbary = np.array([1-mu_star, 0, 0])
    r_Ebary = np.array([-mu_star, 0, 0])
    
    KE = np.dot(vel, vel)/2
    U1 = -(pos[0]**2 + pos[1]**2)/2
    U2 = -((1-mu_star)/np.linalg.norm(pos - r_Ebary[0:3]) + (mu_star)/np.linalg.norm(pos - r_Mbary))

    C = KE + U1 + U2
    
#    breakpoint()
    return C
    
spice.furnsh("/Users/gracegenszler/Documents/Research/starlift/orbits/fullForce.txt")

gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])
#mu_star = gmMoon/(gmEarth + gmMoon)

# for each orbit period folder
baseFileDirectory = '/Users/gracegenszler/Documents/Research/starlift/orbits/graveyardOrbits/L2S/*days/'
fileDirectories = glob(baseFileDirectory)
for ii in np.arange(len(fileDirectories)):
    propTimeMax = 0
    fileDirMax = ''
    fileDir_ii = fileDirectories[ii]
    startTimeDirectories = glob(fileDir_ii + 'startTime_*/')
    
    if len(startTimeDirectories) == 0:
        print('No data. Rerun.')
        print(fileDir_ii)
        print('')
        continue
        
    for jj in np.arange(len(startTimeDirectories)):
        fileDir_jj = startTimeDirectories[jj]
        attemptDirectories = glob(fileDir_jj + 'attemptStartTime_*')
        
        for kk in np.arange(len(attemptDirectories)):
            propTime = float(attemptDirectories[kk].split("_")[-1][:-4])
            if propTime > propTimeMax:
                propTimeMax = propTime
                fileDirMax = attemptDirectories[kk]

    folders = fileDirMax.split('/')
    print('Orbit period: '+str(round(float(folders[9][:-5]), 3))+' days')
    print('Start time: '+str(round(float(folders[11].split('_')[1]), 3))+' mjd')
    print('Prop time: '+str(round(propTimeMax, 3))+' days')

    fileDir_kk = os.path.split(fileDirMax)[0] + '/'
    initialFF = np.load(fileDir_kk+'InitialFF.npz')
    apoluneData = np.load(fileDir_kk+'ApoluneData.npz')
    crtbpData = np.load(fileDir_kk+'CRTBPData.npz', allow_pickle=True)
    periluneData = np.load(fileDir_kk+'PeriluneData.npz')
    event = np.load(fileDirMax)
    
    stateApo0 = initialFF['ICs']
    timeApo0 = initialFF['Ts']
    NOrbApo = initialFF['Norbit']
    NPatchApo = initialFF['Npatch']
    
    statesApoR = apoluneData['RICs']
    statesApoI = apoluneData['IICs']
    timesApo = apoluneData['Ts']
    patchesApoR = apoluneData['patchStatesR']
    patchesApoI = apoluneData['patchStatesI']
    epochesApo = apoluneData['patchTimes']
        
    posCRTBPR = crtbpData['RPs']
    posCRTBPI = crtbpData['IPs']
    velCRTBPR = crtbpData['RVs']
    timesCRTBP = crtbpData['Ts']
    mu_star = crtbpData['mu_star']
    
    statesPeriR = periluneData['RICs']
    statesPeriI = periluneData['IICs']
    timesPeri = periluneData['Ts']
    patchesPeriR = periluneData['patchStatesR']
    patchesPeriI = periluneData['patchStatesI']
    epochesPeri = periluneData['patchTimes']

    states100I = event['states']
#    states100R = event['statesR']
    times100 = event['Ts']
    
    states100R = np.zeros((len(times100), 6))
    for jj in np.arange(len(times100)):
        Crv_I2R = spice.sxform('MCI','MCR',times100[jj])
        states100R[jj,:] = Crv_I2R@states100I[jj,:]

    # Original plot
    ax1 = plt.figure().add_subplot(projection='3d')
    ax1.plot(statesApoR[:,0], statesApoR[:,1], statesApoR[:,2], 'b', label='Ephemeris Orbit')
    ax1.plot(posCRTBPR[:,0], posCRTBPR[:,1], posCRTBPR[:,2], 'r-.', label='CRTBP Orbit')
    ax1.scatter(patchesApoR[:,0], patchesApoR[:,1], patchesApoR[:,2], c='g', marker='o', s=50, alpha=1, label='Patch Points')
    ax1.set_title('First Multi Shooting Pass')
    ax1.set_xlabel('x [km]')
    ax1.set_ylabel('y [km]')
    ax1.set_zlabel('z [km]')
    plt.legend()

    # Pretty perilune plot
    ax2 = plt.figure().add_subplot(projection='3d')
    ax2.plot(statesPeriR[:,0], statesPeriR[:,1], statesPeriR[:,2], 'b', label='Ephemeris Orbit')
    ax2.plot(posCRTBPR[:,0], posCRTBPR[:,1], posCRTBPR[:,2], 'r-.', label='CRTBP Orbit')
    ax2.scatter(patchesPeriR[:,0], patchesPeriR[:,1], patchesPeriR[:,2], c='g', marker='o', s=50, alpha=1, label='Patch Points')
    ax2.set_title('Multi Segment Ephemeris Orbit')
    ax2.set_xlabel('x [km]')
    ax2.set_ylabel('y [km]')
    ax2.set_zlabel('z [km]')
    plt.legend()
            
    # Event plot
    ax3 = plt.figure().add_subplot(projection='3d')
    ax3.plot(states100R[:,0], states100R[:,1], states100R[:,2], 'b', label='Ephemeris Orbit')
    ax3.plot(posCRTBPR[:,0], posCRTBPR[:,1], posCRTBPR[:,2], 'r-.', label='CRTBP Orbit')
    ax3.set_title('MCR Frame '+str(round(propTimeMax, 3))+' day propagation')
    ax3.set_xlabel('x [km]')
    ax3.set_ylabel('y [km]')
    ax3.set_zlabel('z [km]')
    plt.legend()

    ax4 = plt.figure().add_subplot(projection='3d')
    ax4.plot(states100I[:,0], states100I[:,1], states100I[:,2], 'b', label='Ephemeris Orbit')
    ax4.plot(posCRTBPI[:,0], posCRTBPI[:,1], posCRTBPI[:,2], 'r-.', label='CRTBP Orbit')
    ax4.set_title('MCI Frame '+str(round(propTimeMax, 3))+' day propagation')
    ax4.set_xlabel('x [km]')
    ax4.set_ylabel('y [km]')
    ax4.set_zlabel('z [km]')
    plt.legend()
        
    # Jacobi constant with FF
    CFF = np.zeros(len(times100))
    CFF2 = np.zeros(len(times100))
    for jj in np.arange(len(times100)):
        states100R[jj,0:3]*u.km
        pos_can = unitConversion.convertPos_to_canonical(states100R[jj,0:3]*u.km) + np.array([1-mu_star, 0, 0])
        vel_can = unitConversion.convertVel_to_canonical(states100R[jj,3:6]*u.km/u.s)
        CFF[jj] = jacobiConstCRTBPR(pos_can, vel_can, mu_star)
        CFF2[jj] = jacobiConstFFR(states100R[jj,:], times100[jj], GM)
    CFF2 = unitConversion.convertVel_to_canonical(CFF2*u.km/u.s)
    
    # Jacobi constant CRTBP
    CCRTBP = np.zeros(len(timesCRTBP))
    timesCRTBP_dim = np.array([])
    for jj in np.arange(len(timesCRTBP)):
        pos_can = unitConversion.convertPos_to_canonical(posCRTBPR[jj,:]*u.km) + np.array([1-mu_star, 0, 0])
        vel_can = unitConversion.convertVel_to_canonical(velCRTBPR[jj,:]*u.km/u.s)
        CCRTBP[jj] = jacobiConstCRTBPR(pos_can, vel_can, mu_star)
        timesCRTBP_dim = np.append(timesCRTBP_dim, timesCRTBP[jj].value)
    timesCRTBP_dim = (timesCRTBP_dim - timesCRTBP_dim[0])
    
    plt.figure(6)
    plt.plot(times100, CFF2, label = 'FF - FF Method')
    plt.plot(times100, CFF, label = 'FF - CRTBP Method')
    plt.plot(timesCRTBP_dim+times100[0], CCRTBP, label = 'CRTBP')
    plt.xlabel('time [days]')
    plt.ylabel('Jacobi constant [nd]')
    plt.legend()
#    plt.xlim(0,7)
#    plt.ylim(-1.525,-1.515)

    
#    if flagEvent == 0:
#        print('Crashes into Moon \n')
#    elif flagEvent == 1:
#        print('Crashes into Earth \n')
#    elif flagEvent == 2:
#        print('Crashes into Sun \n')
#    elif flagEvent == 3:
#        print('Orbit loses shape \n')
    
#    plt.show()
#    breakpoint()
