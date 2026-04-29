import numpy as np
import spiceypy as spice
from matplotlib import pyplot as plt
from matplotlib import animation

plt.rcParams.update({'font.size': 22})
spice.furnsh("fullForce.txt")

# Parameters
gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])

fileDirectory = '/starlift/orbits/forcedOrbits/'
#fileStr = 'L1_Halo'                     # L1 Halo
#fileStr = 'L2_NRHO'                     # L2 NRHO
fileStr = 'TrajI_1265_MassOptimal'      # L2 Halo
#fileStr = 'TrajI_1265_EnergyOptimal'    # L2 Halo
#fileStr = 'L2_Butterfly'                # L2 Butterfly

mat_data = loadmat(fileStr+'.mat')['TrajI']
posCRTBP_R = mat_data[:,0:3]
velCRTBP_R = mat_data[:,3:6]
timesCRTBP_R = mat_data[:,6]
uT = mat_data[:,7:]
mu_cstar = 0.01215059

# load initialFF data
initialFFData = np.load(fileDirectory+fileStr+'/InitialFF.npz', allow_pickle=True)
correctedInitialEpoches = initialFFData['correctedInitialEpoches']
ff_time = initialFFData['timesTot']

# load plot data
fpoData = np.load(fileDirectory+fileStr+'/plotVariables.npz', allow_pickle=True)
posvel = fpoData['posvel']
posCRTBP_R_dim = fpoData['posCRTBP_R_dim']
rStates = fpoData['rStates']
rotatedStates = fpoData['rotatedStates']
dVCRTBP = fpoData['dVCRTBP']
dVtot = fpoData['dVtot']
Ft_mag = fpoData['Ft_mag']
states_final_R = fpoData['states_final_R']
statesR_diff = fpoData['statesR_diff']
uT_mag = fpoData['uT_mag']
uTNew_mag = fpoData['uTNew_mag']
mNew_di = fpoData['mNew_di']
m_dim = fpoData['m_dim']
dVCRTBPNew = fpoData['dVCRTBPNew']
FtMaxPlt = fpoData['FtMaxPlt']
etCRTBP_mjd = fpoData['etCRTBP_mjd']
ffNew_time = fpoData['ffNew_time']

crtbp_time = (etCRTBP_mjd[:-1] - etCRTBP_mjd[0])/60/60/24
scatterTimes = (correctedInitialEpoches[1:-1] - etCRTBP_mjd[0])/60/60/24

# Position plot with patch points in MCR Frame
ax1 = plt.figure().add_subplot(projection='3d')
for ii in np.arange(N):
    ax1.scatter(posvel[ii,0], posvel[ii,1], posvel[ii,2], c='g', marker='o')
ax1.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r', label='CRTBP')
ax1.set_xlabel('X [km]')
ax1.set_ylabel('Y [km]')
ax1.set_zlabel('Z [km]')
ax1.set_title('Patch Points in Moon Centered Rotating Frame')

# Position plot in MCR Frame post multi-segment algorithm
ax2 = plt.figure().add_subplot(projection='3d')
ax2.plot(rStates[:, 0], rStates[:, 1], rStates[:, 2], 'b', label='Multi Segment')
ax2.scatter(rStates[0,0], rStates[0,1], rStates[0,2], c='g', marker='o')
ax2.scatter(rStates[-1,0], rStates[-1,1], rStates[-1,2], c='y', marker='*')
ax2.plot(rotatedStates[:, 0], rotatedStates[:, 1], rotatedStates[:, 2], 'b', label='Multi Segment')
ax2.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='CRTBP')
ax2.set_xlabel('X [km]', labelpad = 30)
ax2.set_ylabel('Y [km]', labelpad = 30)
ax2.set_zlabel('Z [km]', labelpad = 30)
ax2.set_title('Moon Centered Rotating Frame')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

# Position component plots in MCR Frame post multi-segment algorithm
fig, (ax3, ax4, ax5) = plt.subplots(3, 1)
ax3.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,0])
ax3.set_ylabel('x')
ax4.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,1])
ax4.set_ylabel('y')
ax5.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,2])
ax5.set_ylabel('z')

# Delta-v profile over time
plt.figure(8)
plt.plot(crtbp_time, dVCRTBP, label='Original Thrust Profile')
plt.scatter(scatterTimes, dVtot, c='r', marker='*', zorder=3, label='Patch Point Burns')
plt.yscale('log')
plt.xlabel('Time [d]')
plt.ylabel('Delta-v [km/s]')
plt.title('Delta-v History')
plt.legend()

# Original thrust profile
plt.figure(9)
plt.plot(etCRTBP_mjd, Ft_mag)
plt.yscale('log')
plt.xlabel('Time since epoch [s]')
plt.ylabel('Control Force [mN]')
plt.title('Thrust History')

# Position plot in MCR Frame after propulsion system standardization
ax10 = plt.figure(figsize=(16, 12)).add_subplot(projection='3d')
ax10.plot(statesR[:, 0], statesR[:, 1], statesR[:, 2], 'b', label='Multi Segment - Ephemeris Model')
ax10.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='Single Segment - CRTBP')
ax10.plot(states_final_R[:,0], states_final_R[:,1], states_final_R[:,2], 'y-.', label='Final Trajectory')
ax10.set_xlabel('X [km]', labelpad = 30)
ax10.set_ylabel('Y [km]', labelpad = 30)
ax10.set_zlabel('Z [km]', labelpad = 30)
ax10.set_title('Moon Centered Rotating Frame')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')

# Position component comparison pre and post propulsion system standardization
fig, (ax12, ax13, ax14, ax15) = plt.subplots(4, 1, figsize=(10, 8))
ax12.plot(ff_time, abs(statesR_diff[:,0])*100)
ax12.set_ylabel('x [km]'
ax12.set_xlim(0, ff_time[-1])
ax12.get_xaxis().set_visible(False)
ax12.set_title('Absolute Value Differences')
ax13.plot(ff_time, abs(statesR_diff[:,1]))
ax13.set_ylabel('y [km]')
ax13.set_xlim(0, ff_time[-1])
ax13.get_xaxis().set_visible(False)
ax14.plot(ff_time, abs(statesR_diff[:,2]))
ax14.set_ylabel('z [km]')
ax14.set_xlim(0, ff_time[-1])
ax14.get_xaxis().set_visible(False)
ax15.plot(ff_time, abs(diffR_mag))
ax15.set_ylabel('Position Magnitude [km]')
ax15.set_xlabel('Time [d]')
ax15.set_xlim(0, ff_time[-1])

# Thrust profile comparison pre and post propulsion system standardization
fig, (ax16, ax17) = plt.subplots(2, 1)
ax16.plot(timesCRTBP_d.value, (uT_mag*m_dim).to_value(u.mN), 'b', label='Original Thrust Profile')
ax16.plot(ffNew_time, (uTNew_mag*mNew_dim).to_value(u.mN), 'r-.', label='Recreated Thrust Profile')
ax16.plot(np.array([timesCRTBP_d[0].value, timesCRTBP_d[-1].value]), FtMaxPlt, 'k', label='Max Thrust')
ax16.set_ylabel('Thrust Force [mN]')
ax16.set_xlim(0, ffNew_time[-1])
ax16.set_ylim(49.8, 50.2)
ax16.get_xaxis().set_visible(False)
ax17.plot(timesCRTBP_d.value, (uT_mag*m_dim).to_value(u.mN), 'b', label='Original Thrust Profile')
ax17.plot(ffNew_time, (uTNew_mag*mNew_dim).to_value(u.mN), 'r-.', label='Recreated Thrust Profile')
ax17.plot(np.array([timesCRTBP_d[0].value, timesCRTBP_d[-1].value]), FtMaxPlt, 'k', label='Max Thrust')
ax17.set_xlabel('Time [d]')
ax17.set_ylabel('Thrust Force [mN]')
ax17.set_xlim(0, ffNew_time[-1])
plt.legend()

# Mass profile comparison pre and post propulsion system standardization
plt.figure(18)
plt.plot(ffNew_time, mNew_dim, label='Recreated Mass Profile')
plt.plot(timesCRTBP_d.value, m_dim, label='Original Mass Profile')
plt.legend()
plt.xlabel('Time [d]')
plt.ylabel('Mass [kg]')

# Delta-v profile comparison pre and post propulsion system standardization
plt.figure(19)
plt.plot(ffNew_time[:-1], dVCRTBPNew.to_value(u.m/u.s), 'b', label='Recreated Delta-v Profile')
plt.plot(timesCRTBP_d[:-1].value, (dVCRTBP).to_value(u.m/u.s), 'r-.', label='Original Delta-v Profile')
plt.scatter(((patchTimes[1:-1]-patchTimes[0])*u.s).to_value(u.d), dVtot, c='g', marker='*', zorder=3, label='Patch Point Burns')
plt.legend()
plt.xlabel('time [d]')
plt.ylabel('delta-v [m/s]')
plt.yscale('log')
breakpoint()

plt.show()
