import sys
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
import astropy.units as u
sys.path.insert(1, 'tools')
import unitConversion

def jacobiConstCRTBPR(pos, vel, mu_star):

    r_Mbary = np.array([1-mu_star, 0, 0])
    r_Ebary = np.array([-mu_star, 0, 0])
    
    KE = np.dot(vel, vel)/2
    U1 = -(pos[0]**2 + pos[1]**2)/2
    U2 = -((1-mu_star)/np.linalg.norm(pos - r_Ebary[0:3]) + (mu_star)/np.linalg.norm(pos - r_Mbary))

    C = KE + U1 + U2
    
    return C
    
spice.furnsh("/Users/gracegenszler/Documents/Research/starlift/orbits/fullForce.txt")

gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])

radiiMoon = spice.bodvrd( 'Moon', 'RADII', 3 )[1][0]

# for each orbit period folder
baseFileDirectory = '/Users/gracegenszler/Documents/Research/starlift/orbits/graveyardOrbits/L2S_full100run/*days/'
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
    
    if len(attemptDirectories) == 0:
        print('No data. Rerun.')
        print(fileDir_ii)
        print('')
        continue

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
    times100 = event['Ts']
    try:
        states100R = event['statesR']
        flagEvent = event['eventFlag']
    except:
        states100R = np.zeros((len(times100), 6))
        for jj in np.arange(len(times100)):
            Crv_I2R = spice.sxform('MCI','MCR',times100[jj])
            states100R[jj,:] = Crv_I2R@states100I[jj,:]
        
        rmag_final = np.linalg.norm(states100R[-1,0:3])
        if rmag_final < radiiMoon:
            flagEvent = 0
        else:
            flagEvent = 3

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
        
    # Jacobi constant FF
    CFF = np.zeros(len(times100))
    for jj in np.arange(len(times100)):
        states100R[jj,0:3]*u.km
        pos_can = unitConversion.convertPos_to_canonical(states100R[jj,0:3]*u.km) + np.array([1-mu_star, 0, 0])
        vel_can = unitConversion.convertVel_to_canonical(states100R[jj,3:6]*u.km/u.s)
        CFF[jj] = jacobiConstCRTBPR(pos_can, vel_can, mu_star)
    
    # Jacobi constant CRTBP
    CCRTBP = np.zeros(len(timesCRTBP))
    timesCRTBP_dim = np.array([])
    for jj in np.arange(len(timesCRTBP)):
        pos_can = unitConversion.convertPos_to_canonical(posCRTBPR[jj,:]*u.km) + np.array([1-mu_star, 0, 0])
        vel_can = unitConversion.convertVel_to_canonical(velCRTBPR[jj,:]*u.km/u.s)
        CCRTBP[jj] = jacobiConstCRTBPR(pos_can, vel_can, mu_star)
        timesCRTBP_dim = np.append(timesCRTBP_dim, timesCRTBP[jj].value)
    timesCRTBP_dim = (timesCRTBP_dim - timesCRTBP_dim[0])

    pltTimes100 = ((times100 - times100[0])*u.s).to_value(u.d)
    plt.figure(6)
    plt.plot(pltTimes100, CFF, label = 'FF - CRTBP Method')
    plt.plot(timesCRTBP_dim, CCRTBP, label = 'CRTBP')
    plt.xlabel('time [days]')
    plt.ylabel('Jacobi constant [nd]')
    plt.legend()
    
    if flagEvent == 0:
        print('Crashes into Moon \n')
    elif flagEvent == 1:
        print('Crashes into Earth \n')
    elif flagEvent == 2:
        print('Crashes into Sun \n')
    elif flagEvent == 3:
        print('Orbit loses shape \n')
    else:
        print('Event flag not available \n')
    
#    plt.close('all')
    plt.show()
    breakpoint()
