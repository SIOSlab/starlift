import numpy as np
import os.path
import pickle
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import sys
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
#import plot_tools

#import tools.unitConversion as unitConversion
#import tools.frameConversion as frameConversion
#import tools.orbitEOMProp as orbitEOMProp
#import tools.plot_tools as plot_tools
import pdb

# ~~~~~PROPAGATE THE DYNAMICS  ~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set("ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de440.bsp")

# Parameters
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d + 1*u.yr/4
t_start = Time(57727, format='mjd', scale='utc')
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

C_I2G = frameConversion.inert2geo(t_start,t_veq)
C_G2I = C_I2G.T

# Initial condition in non dimensional units in rotating frame R [pos, vel]
#IC = [0.8497294463740502, 0, 0, 0, 0.47923580202109567, 0, 1.1515074327754002]       # DRO
IC = [1.0110350593505575, 0, -0.17315000084377485, 0, -0.0780142664611386, 0, 0.6816048399338378]   # NRHO L2
#IC = [1.1093213406579072, 0, -0.19463236063796546, 0, -0.22111944917599072, 0, 1.3816048399182301]  # L2
X = [IC[0], IC[2], IC[4], IC[6]]

max_iter = 1000

orbT = unitConversion.convertTime_to_dim(2*IC[6])
step = 1E-3
eps = 4E-6
z = np.array([0, 0, 0, 1])
while orbT.value < 12.1:
    print(orbT)
    error = 10
    ctr = 0
    
    while error > eps and ctr < max_iter:
        Fx = orbitEOMProp.calcFx_R(X, mu_star)

        error = np.linalg.norm(Fx)
        print(error)
        dFx = orbitEOMProp.calcdFx_CRTBP(X,mu_star,m1,m2)

        X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
        
        ctr = ctr + 1
    
#    breakpoint()
    # Generate new z and X for another orbit
    solp = X + z * step
    ss = fsolve(orbitEOMProp.fsolve_eqns, X, args=(z, solp, mu_star), full_output=True, xtol=1E-12)
    X = ss[0]
    Q = ss[1]['fjac']
    Rs = ss[1]['r']
    R = np.zeros((4, 4))
    idx, col = np.triu_indices(4, k=0)
    R[idx, col] = Rs
    J = Q.T @ R

    z = np.linalg.inv(J) @ z
    z = z / np.linalg.norm(z)

    orbT = unitConversion.convertTime_to_dim(2*X[-1])
    
IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])   # Canonical, rotating frame
freeVar_CRTBP = np.array([X[0], X[1], X[2], 2*X[3]])

# Propagate the dynamics in the CRTBP model
statesCRTBP, timesCRTBP = orbitEOMProp.statePropCRTBP(freeVar_CRTBP, mu_star)
posCRTBP = statesCRTBP[:, 0:3]
velCRTBP = statesCRTBP[:, 3:6]
times_dim = unitConversion.convertTime_to_dim(timesCRTBP).to('d')

# sim time in mjd
timesCRTBP_mjd = Time(times_dim.value + t_start.value, format='mjd', scale='utc')

N = 9

dt_epoch = (timesCRTBP_mjd[-1]-timesCRTBP_mjd[0]).value/(N)
dt_int = (timesCRTBP_mjd[-1]-timesCRTBP_mjd[0]).value/(N-1)
taus = Time(np.zeros(N), format='mjd', scale='utc')
Ts = Time(np.ones(N-1)*dt_int, format='jd', scale='utc')
posvel = np.array([])
nodeInds = np.array([])
for ii in np.arange(N):
    time_i = ii*dt_epoch

    difference_array_i = np.absolute(times_dim.value-time_i)
    index_i = difference_array_i.argmin()
    nodeInds = np.append(nodeInds,index_i)

    taus[ii] = timesCRTBP_mjd[index_i]
    pos_i = (unitConversion.convertPos_to_dim(posCRTBP[index_i])).to('AU')
    vel_i = (unitConversion.convertVel_to_dim(velCRTBP[index_i])).to('AU/d')

    posvel = np.append(posvel,np.append(pos_i.value, vel_i.value))
posvel = np.reshape(posvel,(N,6))

#eps = (((N-1)*100*u.m).to('AU')).value
#error = 10
#X = np.append(np.append(posvel,Ts.value),taus.value)
#print(eps)


initialEpoches = taus
initialStates = posvel[0:N-1,:]
finalStates = posvel[1:N,:]
state0 = np.eye(6)
state0 = np.reshape(state0,(1,36))[0]
state0 = np.append(state0,taus[0:2].value)
STMs = np.zeros((N-1,6,6))
#for ii in np.arange(N-1):
#    states, times = orbitEOMProp.jPropFF(state0, t_start, C_G2I)
#    
#    stm = states[-1,:]
#    STMs[ii,:,:] = np.reshape(stm, (6,6))
#
#    state0 = np.append(stm,initialEpoches[ii:ii+2].value)

simTimes = initialEpoches.value - t_start.value

eps1 = (((N-1)*100*u.m).to('AU')).value
print(eps1)
error2 = 10
eps2 = (10*u.m/u.s).to('AU/day').value
initialStates = posvel[0:-1,:]
finalStates = np.zeros((N-1,6))
finalStates[:,0:3] = posvel[1:,0:3]

STMs = np.zeros((N-1,6,6))
while error2 > eps2 and error2 < 11:
    print("layer 1")
    error1 = 10
    while error1 > eps1 and error1 < 11:
        dR = np.array([])
        
        stm_ii = np.eye(6)
        stm_ii = np.reshape(stm_ii,(1,36))[0]

        for ii in np.arange(0,N-1):
            state0 = np.append(initialStates[ii,:],stm_ii)
            state0 = np.append(state0,simTimes[ii:ii+2])
            
            states, times = orbitEOMProp.prop_FF_J(state0, t_start, C_I2G)
            
            Rk = states[-1,0:3]
            Rstar = posvel[ii,0:3]
            
            stm_ii = states[-1,6:]

            STMs[ii,:,:] = np.reshape(stm_ii, (6,6))
        
            phi = STMs[ii,:,:]
            invB = np.linalg.inv(phi[0:3,3:6])
            
            Rdiff = Rstar - Rk
            dvk = invB@Rdiff
            
            dR = np.append(dR, np.linalg.norm(Rdiff))
            
            initialStates[ii,3:6] = initialStates[ii,3:6] + dvk
            
            finalStates[ii,:] = states[-1,0:6]

        error1 = np.sum(dR)
        print(error1)

    print("layer 2")
    dv = np.array([])
    
    dInitialEpoches, dInitialPos, dFinalPos = orbitEOMProp.multiShooting2(initialEpoches, initialStates, finalStates, C_G2I, STMs, t_start)

    initialEpoches = dInitialEpoches*u.d + initialEpoches
    initialStates[:,0:3] = dInitialPos + initialStates[:,0:3]
    finalStates[:,0:3] = dFinalPos + finalStates[:,0:3]
    
    stm_ii = np.eye(6)
    stm_ii = np.reshape(stm_ii,(1,36))[0]
    simTimes = (initialEpoches - initialEpoches[0]).value
    for ii in np.arange(0,N-2):
#        state0 = np.append(initialStates[ii,:],initialEpoches[ii:ii+2].value)
#        states, times = orbitEOMProp.statePropFF(state0, t_start, C_I2G)
        state0 = np.append(initialStates[ii,:],stm_ii)
        state0 = np.append(state0,simTimes[ii:ii+2])
        states, times = orbitEOMProp.prop_FF_J(state0, t_start, C_I2G)
        
        Vminus = states[-1,3:6]
        Vplus = initialStates[ii+1,3:6]
        dvk = Vplus - Vminus
        
        dv = np.append(dv, np.linalg.norm(dvk))
        stm_ii = states[-1,6:]
#        STMs[ii,:,:] = np.reshape(stm_ii,(6,6))
        
#    state0 = np.append(initialStates[-1,:],stm_ii)
#    state0 = np.append(state0,simTimes[-2:])
#    states, times = orbitEOMProp.prop_FF_J(state0, t_start, C_I2G)
#    
#    Vminus = states[-1,3:6]
#    Vplus = initialStates[0,3:6]
#    dvk = Vplus - Vminus
#    
#    dv = np.append(dv, dvk)
    
    stm_ii = states[-1,6:]
#    STMs[-1,:,:] = np.reshape(stm_ii,(6,6))
    
    error2 = np.sum(dv)
    print(error2)

breakpoint()
dp = np.array([])
for ii in np.arange(1,N-1):
    dpk = initialStates[ii,0:3] - finalStates[ii-1,0:3]
    
    dp = np.append(dp,dpk)
    
dpk = initialStates[0,0:3] - finalStates[-1,0:3]
dp = np.append(dp,dpk)
error3 = np.linalg.norm(dp)
print(error3)

ax1 = plt.figure().add_subplot(projection='3d')
for ii in np.arange(0,N-2):
    state0 = np.append(initialStates[ii,:],initialEpoches[ii:ii+2].value)
    states, times = orbitEOMProp.statePropFF(state0, t_start, C_I2G)
    ax1.plot(states[0,:],states[1,:], states[2,:])
    
state0 = np.append(initialStates[-1,:],initialEpoches[-2:].value)
states, times = orbitEOMProp.statePropFF(state0, t_start, C_I2G)
ax1.plot(states[0,:],states[1,:], states[2,:])
plt.show()
    
breakpoint()

