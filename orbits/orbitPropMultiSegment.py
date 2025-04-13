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
import pdb

# ~~~~~PROPAGATE THE DYNAMICS  ~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set("ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de440.bsp")

# Parameters
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d + 1*u.yr/4
t_start = Time(57727, format='mjd', scale='utc')
#t_start = Time(58070, format='mjd', scale='utc')
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

C_I2G = frameConversion.inert2geo(t_start,t_veq)
C_G2I = C_I2G.T

# Initial condition in non dimensional units in rotating frame R [pos, vel]
#IC = [0.8497294463740502, 0, 0, 0, 0.47923580202109567, 0, 1.1515074327754002]       # DRO
#IC = [1.0110350593505575, 0, -0.17315000084377485, 0, -0.0780142664611386, 0, 0.6816048399338378]   # NRHO L2
IC = [1.114959432252717, 0, -0.027057507726036, 0, 0.191674660415012, 0, 1.701721247470296]
#IC = [1.1093213406579072, 0, -0.19463236063796546, 0, -0.22111944917599072, 0, 1.3816048399182301]  # L2

#Phi0 = np.eye(6)
#Phi0 = np.reshape(Phi0,(36,1))
#X = np.append(IC[0:6],Phi0)
#X = np.append(X,IC[-1])
X = np.array([IC[0], IC[2], IC[4], 1*IC[-1]])

max_iter = 1000

orbT0 = unitConversion.convertTime_to_dim(2*IC[6])
orbT = unitConversion.convertTime_to_dim(2*IC[6])
step = 1E-2
eps = 1E-6
z = np.array([0, 0, 0, 1])
#while orbT.value < orbT0.value+.01:
print(orbT)
error = 10

Fx = orbitEOMProp.calcFx_R(X, mu_star)

error = np.linalg.norm(Fx)
print(error)
while error > eps and ctr < max_iter:
    dFx = orbitEOMProp.calcdFx_CRTBP(X, mu_star, m1, m2)

    X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
    
    ctr = ctr + 1
    
    Fx = orbitEOMProp.calcFx_R(X, mu_star)

    error = np.linalg.norm(Fx)
    print(error)
        
    # Generate new z and X for another orbit    *******THIS IS MAYBE BROKEN***********
#    solp = X + z * step
#    ss = fsolve(orbitEOMProp.fsolve_eqns, X, args=(z, solp, mu_star), full_output=True, xtol=1E-12)
#    X = ss[0]
#    Q = ss[1]['fjac']
#    Rs = ss[1]['r']
#    R = np.zeros((4, 4))
#    idx, col = np.triu_indices(4, k=0)
#    R[idx, col] = Rs
#    J = Q.T @ R
#
#    z = np.linalg.inv(J) @ z
#    z = z / np.linalg.norm(z)
#
#    orbT = unitConversion.convertTime_to_dim(2*X[-1])
    
IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])   # Canonical, rotating frame
freeVar_CRTBP = np.array([X[0], X[1], X[2], 2*X[3]])

# Propagate the dynamics in the CRTBP model
statesCRTBP, timesCRTBP = orbitEOMProp.statePropCRTBP_R(freeVar_CRTBP, mu_star)
posCRTBP = statesCRTBP[:, 0:3]
velCRTBP = statesCRTBP[:, 3:6]
times_dim = unitConversion.convertTime_to_dim(timesCRTBP).to('d')

posDim = unitConversion.convertPos_to_dim(posCRTBP).to('AU').value
# Get position of the moon at the epoch in the inertial frame
sun_I, earth_I, moon_I = frameConversion.getSunEarthMoon(t_start, C_I2G)  # I frame [AU]
moon_I_can = unitConversion.convertPos_to_canonical(moon_I)

# Transform position ICs to the epoch moon
ideal_moon = [1-mu_star, 0, 0]
IC_x = (IC[0] - ideal_moon[0]) + moon_I_can[0]
IC_y = (IC[1] - ideal_moon[1]) + moon_I_can[1]
IC_z = (IC[2] - ideal_moon[2]) + moon_I_can[2]
IC[0:3] = [IC_x, IC_y, IC_z]  # Canonical, I frame

# Convert the velocity to I frame from R frame (position is the same in both)
vO = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Rotate velocity vector to match the epoch moon (I frame)
theta = np.arccos((np.dot(moon_I_can, ideal_moon))/(np.linalg.norm(moon_I_can)*np.linalg.norm(ideal_moon)))
if theta > np.pi/2:
    theta = -theta
rot_matrix = frameConversion.rot(theta, 3)
IC[0:3] = rot_matrix @ IC[0:3]  # Canonical, I frame
IC[3:6] = rot_matrix @ vO  # Canonical, I frame

# Convert IC to dimensional, rotating frame (for STK)
C_I2R = frameConversion.inert2rot(t_start, t_start)
pos_canrot = C_I2R @ IC[0:3]  # Canonical, R frame
vel_canrot = frameConversion.inert2rotV(pos_canrot, IC[3:6], 0)  # Canonical, R frame
pos_dimrot = unitConversion.convertPos_to_dim(pos_canrot).to('AU')  # R frame, dimensional
vel_dimrot = unitConversion.convertVel_to_dim(vel_canrot).to('AU/d')  # R frame, dimensional

stm0 = np.eye(6)
stm0 = np.reshape(stm0,(1,36))[0]

ts = np.array([times_dim[0].value, times_dim[-1].value])
state0 = np.append(np.append(np.append(pos_dimrot.value,vel_dimrot.value),stm0), ts)
statesFF, times = orbitEOMProp.prop_FF_J(state0, t_start, C_I2G)

ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(posDim[:,0],posDim[:,1],posDim[:,2])
ax1.plot(statesFF[:,0],statesFF[:,1],statesFF[:,2])
plt.show()
breakpoint()


# sim time in mjd
timesCRTBP_mjd = Time(times_dim.value + t_start.value, format='mjd', scale='utc')

N = 16

dt_epoch = (timesCRTBP_mjd[-1]-timesCRTBP_mjd[0]).value/(N)
dt_int = (timesCRTBP_mjd[-1]-timesCRTBP_mjd[0]).value/(N-1)
taus = Time(np.zeros(N), format='mjd', scale='utc')
Ts = Time(np.ones(N-1)*dt_int, format='jd', scale='utc')
posvel = np.array([])
nodeInds = np.array([])
for ii in np.arange(N):
    time_i = ii*dt_int

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

#eps1 = (((N-1)*100*u.m).to('AU')).value
eps1 = 1E-4 #((100*u.m).to('AU')).value
print(eps1)
error2 = 10
eps2 = (10*u.m/u.s).to('AU/day').value  # total velocity within .01km/s
initialStates = posvel[0:-1,:].copy()
finalStates = posvel[1:,:].copy()

STMs = np.zeros((N-1,6,6))
stm_0 = np.eye(6)
stm_0 = np.reshape(stm_0,(1,36))

while error2 > eps2 and error2 < 11:
    print("layer 1")
    
    ax1 = plt.figure().add_subplot(projection='3d')
    ax1.plot(posDim[:,0],posDim[:,1],posDim[:,2],'b')
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
        
        ax1.plot(states[:,0],states[:,1],states[:,2],'r')
        ax1.scatter(states[0,0],states[0,1],states[0,2],'*')
        ax1.scatter(states[-1,0],states[-1,1],states[-1,2],'+')
    plt.show()
    breakpoint()
        
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

