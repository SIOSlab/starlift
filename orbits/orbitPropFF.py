import numpy as np
import sys
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
import plot_tools
import extractTools
import pdb

# ~~~~~PROPAGATE THE DYNAMICS~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de440')

# Parameters
orbs = 1
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d  # + 1*u.yr/4
t_start = Time(57727, format='mjd', scale='utc')
days = 14*orbs
days_can = unitConversion.convertTime_to_canonical(days * u.d)
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in canonical units in rotating frame R [pos, vel]
#IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period
# IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]  # DRO, 0.9697497-day period
#IC = [0.429519110229904, 0, 0, 0, 1.440796689672539, 0, 3.051133070334277]
#IC = [1.165130674583613, 0, -0.110699848144854, 0, 0.201519926517907, 0, 1.652428300688599]
IC = [1.114959432252717, 0, -0.027057507726036, 0, 0.191674660415012, 0, 3.403442494940593/2]
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

IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])  # Canonical, rotating frame

# Propagate the dynamics (states in AU or AU/day, times in days starting from 0)
freeVar0CRTBP_R = X.copy()
freeVar0CRTBP_R[-1] = 2*freeVar0CRTBP_R[-1]*orbs
statesCRTBP_R, timesCRTBP_R = orbitEOMProp.statePropCRTBP_R(freeVar0CRTBP_R, mu_star)  # State is in the R frame
posCRTBP_R = statesCRTBP_R[:, 0:3]
velCRTBP_R = statesCRTBP_R[:, 3:6]
posCRTBP_R_dim = unitConversion.convertPos_to_dim(posCRTBP_R).to('AU')
#breakpoint()
# DCM for G frame and I frame
C_I2G = frameConversion.inert2geo(t_start, t_veq)
C_G2I = C_I2G.T

# sim time in mjd
times_dim = unitConversion.convertTime_to_dim(timesCRTBP_R).to('d')
timesCRTBP_mjd = Time(times_dim.value + t_start.value, format='mjd', scale='utc')

# collect states in H at nodes
N = 7*orbs
dt_int = (timesCRTBP_mjd[-1]-timesCRTBP_mjd[0]).value/(N-1)
taus = Time(np.zeros(N), format='mjd', scale='utc')
posvel = np.array([])
nodeInds = np.array([])
for ii in np.arange(N):
    time_i = ii*dt_int

    difference_array_i = np.absolute(times_dim.value-time_i)
    index_i = difference_array_i.argmin()
    nodeInds = np.append(nodeInds,index_i)
    
    taus[ii] = timesCRTBP_mjd[index_i]
    
    state_i = np.append(posCRTBP_R[index_i], velCRTBP_R[index_i])

    _, _, state_ii = frameConversion.convertSC_R2I(t_start, taus[ii], C_I2G, state_i, mu_star)
    
    pos_i = np.array(state_ii[0:3])
    vel_i = np.array(state_ii[3:6])
    
    # Convert ICs to H frame (AU and AU/d) from I frame (canonical)
    pos_H, vel_H = frameConversion.convertSC_I2H(pos_i, vel_i, taus[ii], C_I2G)

    # Define the initial state array
    state_H = np.append(pos_H.value, vel_H.value)
    posvel = np.append(posvel,state_H)
posvel_H = np.reshape(posvel,(N,6))

# Propagate the dynamics (states in AU or AU/day, times in days starting from 0)
state0 = np.append(posvel_H[0],times_dim[-1].value)
states, times = orbitEOMProp.statePropFF(state0, t_start)  # State is in the H frame
pos = states[:, 0:3]
vel = states[:, 3:6]
#
# Convert to canonical
pos_can = unitConversion.convertPos_to_canonical(pos * u.AU)
vel_can = unitConversion.convertVel_to_canonical(vel * u.AU/u.d)

# Simulation time in mjd
times_mjd = times + t_start  # Days from mission start time

# Preallocate space
pos_SC = np.zeros([len(times_mjd), 3])
vel_SC = np.zeros([len(times_mjd), 3])
pos_Sun = np.zeros([len(times_mjd), 3])
pos_Earth = np.zeros([len(times_mjd), 3])
pos_Moon = np.zeros([len(times_mjd), 3])
pos_Sun_H = np.zeros([len(times_mjd), 3])
pos_Earth_H = np.zeros([len(times_mjd), 3])
pos_Moon_H = np.zeros([len(times_mjd), 3])

# Obtain celestial body positions in the I frame [AU] and convert state to I frame
for ii in np.arange(len(times_mjd)):
    pos_SC[ii, :], vel_SC[ii, :] = frameConversion.convertSC_H2I(pos_can[ii, :], vel_can[ii, :], times_mjd[ii], C_I2G)
    pos_Sun[ii, :], pos_Earth[ii, :], pos_Moon[ii, :] = frameConversion.getSunEarthMoon(times_mjd[ii], C_I2G)
    pos_Sun_H[ii, :] = get_body_barycentric_posvel('Sun', times_mjd[ii])[0].get_xyz().to('AU').value
    pos_Earth_H[ii, :] = get_body_barycentric_posvel('Earth', times_mjd[ii])[0].get_xyz().to('AU').value
    pos_Moon_H[ii, :] = get_body_barycentric_posvel('Moon', times_mjd[ii])[0].get_xyz().to('AU').value


# set tolerances for optimizers
eps1 = ((100*u.m).to('AU')).value
eps2 = (.1*u.m/u.s).to('AU/day').value  # total velocity within .01km/s
error2 = 10

# initialize initial and final states and epoches
initialEpoches = taus
simTimes = initialEpoches.value - t_start.value
initialStates = posvel_H[0:-1,:].copy()
finalStates = posvel_H[1:,:].copy()
STMs = np.zeros((N-1,6,6))
stm_0 = np.eye(6)
stm_0 = np.reshape(stm_0,(1,36))
while error2 > eps2 and error2 < 11:
    print("layer 1")
    
#    ax1 = plt.figure().add_subplot(projection='3d')
    for ii in np.arange(0,N-1):
        print("segment " + str(ii))
        error1 = 10
        ctr = 0
        Rstar = finalStates[ii,0:3]
        
        while error1 > eps1 and error1 < 11:
            state0 = np.append(initialStates[ii,:],stm_0)
            state0 = np.append(state0,simTimes[ii:ii+2])
            
            states, times = orbitEOMProp.prop_FF_J(state0, t_start, C_I2G)
            
            Rk = states[-1,0:3]
            
            stm_ii = states[-1,6:]
            phi = np.reshape(stm_ii, (6,6))

            invB = np.linalg.inv(phi[0:3,3:6])
            
            Rdiff = (Rstar - Rk)*.618
            dvk = invB@Rdiff
            
            error1 = np.linalg.norm(Rdiff)
            print(error1)

            initialStates[ii,3:6] = initialStates[ii,3:6] + dvk
            
            ctr = ctr + 1
#        breakpoint()
        STMs[ii,:,:] = phi
        finalStates[ii,:] = states[-1,0:6].copy()
        print(ctr)
        
#        ax1.plot(states[:,0],states[:,1],states[:,2],'r')
#        ax1.scatter(states[0,0],states[0,1],states[0,2],'*')
#        ax1.scatter(states[-1,0],states[-1,1],states[-1,2],'+')
#    plt.show()
#    breakpoint()
        
#    breakpoint()
    print("layer 2")
    dv = np.array([])
    
    dInitialEpoches, dInitialPos, dFinalPos, minus_dv = orbitEOMProp.multiShooting2(initialEpoches, initialStates, finalStates, C_G2I, STMs, t_start)

    initialEpoches = dInitialEpoches*u.d + initialEpoches
    initialStates[:,0:3] = dInitialPos + initialStates[:,0:3]
    finalStates[:,0:3] = dFinalPos + finalStates[:,0:3]

    error2 = np.linalg.norm(minus_dv)

    print(error2)
#    breakpoint()

# Propagate the dynamics for each segment and convert to I and R
pos_MS_H = np.array([np.nan, np.nan, np.nan])
pos_MS_I = np.array([np.nan, np.nan, np.nan])
pos_MS_R = np.array([np.nan, np.nan, np.nan])
NI_MS_H = np.array([np.nan, np.nan, np.nan])
NI_MS_I = np.array([np.nan, np.nan, np.nan])
NI_MS_R = np.array([np.nan, np.nan, np.nan])
NF_MS_H = np.array([np.nan, np.nan, np.nan])
NF_MS_I = np.array([np.nan, np.nan, np.nan])
NF_MS_R = np.array([np.nan, np.nan, np.nan])
for ii in np.arange(N-1):

    # fix some stacking indexing here
    dt = (initialEpoches[ii+1] - initialEpoches[ii]).value
    state0 = np.append(initialStates[ii],dt)
    states, times = orbitEOMProp.statePropFF(state0, initialEpoches[ii])  # State is in the H frame
    pos = states[:, 0:3]
    vel = states[:, 3:6]
    pos_MS_H = np.vstack((pos_MS_H, pos))
    NI_MS_H = np.vstack((NI_MS_H, pos[0]))
    NF_MS_H = np.vstack((NF_MS_H,finalStates[ii,0:3]))
            
    MS_I = np.array([np.nan, np.nan, np.nan])
    MS_R = np.array([np.nan, np.nan, np.nan])
    for jj in np.arange(len(times)):
        pos_can = unitConversion.convertPos_to_canonical(pos[jj]*u.AU)
        vel_can = unitConversion.convertVel_to_canonical(vel[jj]*u.AU/u.d)
        t_mjd = times[jj] + initialEpoches[ii]
        pos_jj_I, vel_jj_I = frameConversion.convertSC_H2I(pos_can, vel_can, t_mjd, C_I2G)
        MS_I = np.vstack((MS_I, pos_jj_I))
        
        C_I2R = frameConversion.inert2rot(t_mjd,t_start)
        pos_jj_R = C_I2R@pos_jj_I
        MS_R = np.vstack((MS_R, pos_jj_R))
    
    pos_f = finalStates[ii,0:3]
    vel_f = finalStates[ii,3:6]
    pos_can = unitConversion.convertPos_to_canonical(pos_f*u.AU)
    vel_can = unitConversion.convertVel_to_canonical(vel_f*u.AU/u.d)
    pos_I, vel_I = frameConversion.convertSC_H2I(pos_can, vel_can, initialEpoches[ii+1], C_I2G)
    NF_MS_I = np.vstack((NF_MS_I, pos_jj_I))
    pos_MS_I = np.vstack((pos_MS_I, MS_I))
    
    C_I2R = frameConversion.inert2rot(initialEpoches[ii+1],t_start)
    pos_R = C_I2R@pos_I
    NF_MS_R = np.vstack((NF_MS_R, pos_jj_R))
    pos_MS_R = np.vstack((pos_MS_R, MS_R))
        
    pos_MS_I = np.vstack((pos_MS_I, pos_jj_I))
    NI_MS_I = np.vstack((NI_MS_I, MS_I[1]))
    
    pos_MS_R = np.vstack((pos_MS_R, pos_jj_R))
    NI_MS_R = np.vstack((NI_MS_R, MS_R[1]))
    
ax10 = plt.figure().add_subplot(projection='3d')
ax10.plot(pos_MS_H[:, 0], pos_MS_H[:, 1], pos_MS_H[:, 2], 'b', label='H frame')
ax10.scatter(NI_MS_H[:, 0], NI_MS_H[:, 1], NI_MS_H[:, 2], marker='s')
ax10.scatter(NF_MS_H[:, 0], NF_MS_H[:, 1], NF_MS_H[:, 2], marker='o')

ax20 = plt.figure().add_subplot(projection='3d')
ax20.plot(pos_MS_I[:, 0], pos_MS_I[:, 1], pos_MS_I[:, 2], 'b', label='Ephemeris Model')
ax20.plot(pos_SC[:, 0], pos_SC[:, 1], pos_SC[:, 2], 'r', label='CRTBP Model')
ax20.scatter(NI_MS_I[:, 0], NI_MS_I[:, 1], NI_MS_I[:, 2], marker='s')
ax20.scatter(NF_MS_I[:, 0], NF_MS_I[:, 1], NF_MS_I[:, 2], marker='o')
ax20.set_xlabel('X [AU]')
ax20.set_ylabel('Y [AU]')
ax20.set_zlabel('Z [AU]')
ax20.set_title('Earth-Moon Inertial Frame')

ax30 = plt.figure().add_subplot(projection='3d')
ax30.plot(pos_MS_R[:, 0], pos_MS_R[:, 1], pos_MS_R[:, 2], 'b', label='R frame')
ax30.plot(posCRTBP_R_dim[:,0],posCRTBP_R_dim[:,1],posCRTBP_R_dim[:,2],'r', label='CRTBP')
ax30.scatter(NI_MS_R[:, 0], NI_MS_R[:, 1], NI_MS_R[:, 2], marker='s')
ax30.scatter(NF_MS_R[:, 0], NF_MS_R[:, 1], NF_MS_R[:, 2], marker='o')

plt.show()
breakpoint()

# ~~~~~OBTAIN STK DATA~~~~

# Obtain FF rotating data from STK
file_path = "gmatSTKFiles/L2Orbit_Full_Force_State_earthpoint100.txt"
stk_posrot, stk_times = extractTools.extractSTK(file_path)

# Convert to I frame from R frame
stk_posinert = np.zeros([len(stk_times), 3])
for ii in np.arange(len(stk_times)):
    C_I2R = frameConversion.inert2rot(stk_times[ii], stk_times[0])
    C_R2I = C_I2R.T
    stk_posinert[ii, :] = C_R2I @ stk_posrot[ii, :]


# ~~~~~PLOT~~~~

title = 'Ephemeris Model in the Inertial Earth-Moon Frame'
body_names = ['Propagated Ephemeris Model', 'Earth', 'Moon', 'STK Orbit']
fig_I, ax_I = plot_tools.plot_bodies(pos_SC, pos_Earth, pos_Moon, stk_posinert, body_names=body_names, title=title)

# title = 'Full Force Model in the H Frame'
# body_names = ['Spacecraft', 'Moon', 'Sun']
# fig_H, ax_H = plot_tools.plot_bodies(pos, pos_Moon_H, pos_Sun_H, body_names=body_names, title=title)


# ~~~~~ANIMATION~~~~~


def interpolate_positions(stk_pos, stk_times, target_times):
    # Create interpolation functions for each position component (x, y, z)
    interp_func_x = interp1d(stk_times.value, stk_pos[:, 0], kind='linear', fill_value="extrapolate")
    interp_func_y = interp1d(stk_times.value, stk_pos[:, 1], kind='linear', fill_value="extrapolate")
    interp_func_z = interp1d(stk_times.value, stk_pos[:, 2], kind='linear', fill_value="extrapolate")

    # Interpolate stk_posrot to match target_times
    interp_x = interp_func_x(target_times.value)
    interp_y = interp_func_y(target_times.value)
    interp_z = interp_func_z(target_times.value)

    # Combine interpolated components into a new position array
    interpolated_posrot = np.vstack((interp_x, interp_y, interp_z)).T

    return interpolated_posrot


interp_stk_posinert = interpolate_positions(stk_posinert, stk_times, times_mjd)

desired_duration = 5  # seconds
title = 'Full Force Model in the Inertial (I) Frame'
body_names = ['Propagated FF', 'Earth', 'Moon', 'STK Orbit']
animate_func_I, ani_object_I = plot_tools.create_animation(times, days, desired_duration,
                                                       [pos_SC, pos_Earth, pos_Moon, interp_stk_posinert],
                                                       body_names=body_names, title=title)

# title = 'Full Force Model in the H Frame'
# body_names = ['Spacecraft', 'Earth', 'Moon', 'Sun']
# animate_func_H, ani_object_H = plot_tools.create_animation(times, days, desired_duration,
#                                                            [pos, pos_Earth_H, pos_Moon_H, pos_Sun_H],
#                                                            body_names=body_names, title=title)


# # ~~~~~SAVE~~~~~
#
# fig_I.savefig('plotFigures/FF STK earth point mass 100 days.png')
# # fig_H.savefig('plotFigures/FF DRO H frame.png')
#
# writergif = animation.PillowWriter(fps=30)
# ani_object_I.save('plotFigures/FF STK earth point mass 100 days.gif', writer=writergif)
# # ani_object_H.save('plotFigures/FF DRO H frame.gif', writer=writergif)
