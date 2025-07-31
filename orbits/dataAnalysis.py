import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
import astropy.units as u
sys.path.insert(1, 'tools')
import unitConversion

def jacobiConstFFR(state_scM, time):

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
    
    pos_scE = state_scM[0:3] - r_Earth
    pos_scS = state_scM[0:3] - r_Sun
    pos_Mbary = -state_baryM[0:3]
    pos_Ebary = r_Earth - state_baryM[0:3]
    pos_Sbary = r_Sun - state_baryM[0:3]
    
    KE1 = np.dot(state_scBary[3:6], state_scBary[3:6])/2
    U11 = -(state_scBary[0]**2 + state_scBary[1]**2)/2
    U21 = -(np.linalg.norm(pos_Mbary)/np.linalg.norm(state_scM[0:3]) + np.linalg.norm(pos_Ebary)/np.linalg.norm(pos_scE) + np.linalg.norm(pos_Sbary)/np.linalg.norm(pos_scS))

    C1 = KE1 + U11 + U21

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

fileDirectory = '/Users/gracegenszler/Documents/Research/starlift/orbits/graveyardOrbits/L2Nx/'
#folders = ['6.019359389530944_days/', '6.889032355209055_days/', '7.802188969171071_days/', '8.671861934849183_days/', '9.541534900527276_days/', '10.454691514489271_days/', '11.324364480167363_days/', '12.194037445845456_days/', '13.107194059807451_days/', '13.976867025485543_days/']
folders = ['6.889032355209055_days/']

for ii in np.arange(len(folders)):
    centuryData = np.load(fileDirectory+folders[ii]+'CenturyData.npz')
    event = np.load(fileDirectory+folders[ii]+'Event.npz')
    initialFF = np.load(fileDirectory+folders[ii]+'InitialFF.npz')
    apoluneData = np.load(fileDirectory+folders[ii]+'ApoluneData.npz')
    perilune = np.load(fileDirectory+folders[ii]+'Perilune.npz')
    periluneData = np.load(fileDirectory+folders[ii]+'PeriluneData.npz')
    crtbpData = np.load(fileDirectory+folders[ii]+'CRTBPData.npz', allow_pickle=True)
    
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

    statePeri0 = perilune['ICs']
    timePeri0 = perilune['Ts']
    NOrbPeri = perilune['Norbit']
    NPatchPeri = perilune['Npatch']
    
    statesPeriR = periluneData['RICs']
    statesPeriI = periluneData['IICs']
    timesPeri = periluneData['Ts']
    patchesPeriR = periluneData['patchStatesR']
    patchesPeriI = periluneData['patchStatesI']
    epochesPeri = periluneData['patchTimes']
    
    states100R = centuryData['RICs']
    states100I = centuryData['IICs']
    times100 = centuryData['Ts']
    
    stateEvent = event['ICs']
    timeEvent = event['Ts']
    flagEvent = event['flag']
    indEvent = np.argwhere(timeEvent == times100)
        
    posCRTBPR = crtbpData['RPs']
    posCRTBPI = crtbpData['IPs']
    velCRTBPR = crtbpData['RVs']
    timesCRTBP = crtbpData['Ts']
    mu_star = crtbpData['mu_star']

    if len(timesApo) > 1:
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

        if len(timesPeri) > 1:
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
            
            if np.any(indEvent):
                indEvent = indEvent[0,0]
                # Event plot
                ax3 = plt.figure().add_subplot(projection='3d')
                ax3.plot(states100R[:indEvent,0], states100R[:indEvent,1], states100R[:indEvent,2], 'b', label='Ephemeris Orbit')
                ax3.plot(posCRTBPR[:,0], posCRTBPR[:,1], posCRTBPR[:,2], 'r-.', label='CRTBP Orbit')
                ax3.set_title(str((times100[indEvent]-times100[0])/60/60/24)+' day propagation')
                ax3.set_xlabel('x [km]')
                ax3.set_ylabel('y [km]')
                ax3.set_zlabel('z [km]')
                plt.legend()
                
                # Century plot, if applicable
                ax4 = plt.figure().add_subplot(projection='3d')
                ax4.plot(states100R[:,0], states100R[:,1], states100R[:,2], 'b', label='Ephemeris Orbit')
                ax4.set_title('Moon Centered Rotating Frame: Propogate for 100 years')
                ax4.set_xlabel('x [km]')
                ax4.set_ylabel('y [km]')
                ax4.set_zlabel('z [km]')
                plt.legend()
                
                ax5 = plt.figure().add_subplot(projection='3d')
                ax5.plot(states100I[:,0], states100I[:,1], states100I[:,2], 'b', label='Ephemeris Orbit')
                ax5.set_title('Moon Centered Inertial Frame: Propogate for 100 years')
                ax5.set_xlabel('x [km]')
                ax5.set_ylabel('y [km]')
                ax5.set_zlabel('z [km]')
                plt.legend()
                
#                CFF = np.array([])
#                ctr = 0
#                while (times100[ctr] - times100[0])/24/60/60 < 7*24*60*60:
#                    tmp = jacobiConstFFR(states100R[ctr,:], times100[ctr])
#                    CFF = np.append(CFF, tmp)
#                    ctr = ctr + 1
                    
                # Jacobi constant with ephemeris data, but CRTBP calculation
#                CFF = np.zeros(len(times100[:indEvent]))
#                for jj in np.arange(len(times100[:indEvent])):
#                    states100R[jj,0:3]*u.km
#                    pos_can = unitConversion.convertPos_to_canonical(states100R[jj,0:3]*u.km) + np.array([1-mu_star, 0, 0])
#                    vel_can = unitConversion.convertVel_to_canonical(states100R[jj,3:6]*u.km/u.s)
#                    CFF[jj] = jacobiConstCRTBPR(pos_can, vel_can, mu_star)
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

#                times100plt = times100[:indEvent] - times100[0]
                times100plt = times100 - times100[0]
#                breakpoint()
                plt.figure(6)
#                plt.plot(times100plt, CFF[:indEvent], label = 'FF')
                plt.plot(times100plt, CFF, label = 'FF')
                plt.plot(timesCRTBP_dim, CCRTBP, label = 'CRTBP')
                plt.xlabel('time [days]')
                plt.ylabel('Jacobi constant [nd]')
                plt.legend()
                plt.xlim(0,7)
                plt.ylim(-1.525,-1.515)
                plt.show()
                breakpoint()
                if flagEvent == 0:
                    print('Crashes into Moon')
                elif flagEvent == 1:
                    print('Crashes into Earth')
                elif flagEvent == 2:
                    print('Crashes into Sun')
                elif flagEvent == 3:
                    print('Orbit loses shape')
            else:
                print('No event')
        else:
            print('No perilune data')
    else:
        print('No data')
        
    plt.show()
    breakpoint()
