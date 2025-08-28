import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp
from astropy.time import Time
from matplotlib import pyplot as plt


def multipleShootingI(initialEpoches, initialStates, positionTolerance, velocityTolerance, GM):

    iterationNumberLevelTwoMax = 20

    N = len(initialEpoches)

    iterationNumberLevelTwo = 1
    correctedInitialEpoches = initialEpoches.copy()
    correctedInitialStates = initialStates.copy()
    correctedFinalStates = initialStates[1:,:].copy()
    
    deltaV = 1
    while deltaV > velocityTolerance:
        stateTransitionMatrixes = np.zeros((N,6,6))
        exitflag = np.zeros((N,1))
        exitFlagLevel1 = np.zeros((N-1,1))
        correctedFinalStates = np.zeros((N-1,6))
        STMs = np.zeros((N-1,6,6))
        print('Inner loop: position shooting')
        for ii in np.arange(N-1):
            cInitial, cFinal, STM, exitFlag1 = positionShooting(correctedInitialEpoches[ii], correctedInitialStates[ii,:], correctedInitialEpoches[ii+1], correctedInitialStates[ii+1,:], positionTolerance, GM)
            correctedInitialStates[ii,:] = cInitial
            correctedFinalStates[ii,:] = cFinal
            STMs[ii,:,:] = STM
            exitFlagLevel1[ii] = exitFlag1
            if not exitFlagLevel1[ii]:
                print('     Segment '+str(ii)+'/'+str(N-2)+' fails at position shooting')
                deltaV = -1
                break
            else:
                print('     Segment '+str(ii)+'/'+str(N-2)+' done')
        
        # test failure
        if np.any(exitFlagLevel1 != 1):
            exitflag = -2
            break
        
        #---- level-2 shooting ----
        print('Outer loop: velocity matching')
        # collcect the target error
        deltaVelocity = correctedFinalStates[:-1,3:6] - correctedInitialStates[1:-1,3:6]
        deltaVelocity = np.reshape(deltaVelocity,(1,3*(N-2)))[0]
        deltaV = np.linalg.norm(deltaVelocity)
        print('     Iteration '+str(iterationNumberLevelTwo)+' norm: '+str(deltaV))
        
        # test early stop
        if deltaV < velocityTolerance:
            exitflag = 1
            print('Multi-shooting success! \n')
            break
        
        # modify epoch and velocity of all segments at once
        # after one modification, shooting position again
        dVdu = np.zeros((N-2,3,12))         # for all the interior patch points 1 to N-2
        for ii in np.arange(1,N-1):
            # generate state relationship matrix
            stm21 = STMs[ii-1, :, :]
            stm12 = np.linalg.inv(stm21)
            stm32 = STMs[ii, :, :]

            v1plus  = correctedInitialStates[ii-1, 3:6]
            v2minus = correctedFinalStates[ii-1, 3:6]
            v2plus  = correctedInitialStates[ii, 3:6]
            v3minus = correctedFinalStates[ii, 3:6]

            a2minus = ffInertial(correctedInitialEpoches[ii], correctedFinalStates[ii-1, :], GM)
            a2minus = a2minus[3:6]
            a2plus  = ffInertial(correctedInitialEpoches[ii], correctedInitialStates[ii, :], GM)
            a2plus = a2plus[3:6]

            dVdu1 = -np.linalg.inv(stm12[0:3,3:6])
            dVdu2 = np.linalg.inv(stm12[0:3,3:6])@v1plus
            dVdu3 = -np.linalg.inv(stm32[0:3,3:6])@stm32[0:3,0:3] + np.linalg.inv(stm12[0:3,3:6])@stm12[0:3,0:3]
            dVdu4 = (a2plus-a2minus) + (np.linalg.inv(stm32[0:3,3:6])@stm32[0:3,0:3]@v2plus - np.linalg.inv(stm12[0:3,3:6])@stm12[0:3,0:3]@v2minus)
            dVdu5 = np.linalg.inv(stm32[0:3,3:6])
            dVdu6 = -np.linalg.inv(stm32[0:3,3:6])@v3minus
            
            dVdu[ii-1,:,0:3] = dVdu1
            dVdu[ii-1,:,3] = dVdu2
            dVdu[ii-1,:,4:7] = dVdu3
            dVdu[ii-1,:,7] = dVdu4
            dVdu[ii-1,:,8:11] = dVdu5
            dVdu[ii-1,:,11] = dVdu6
          
        bb = deltaVelocity
        M = np.zeros((len(bb), 4*(N)))
        for ii in np.arange(0,N-2):
            M[3*(ii):3*(ii+1),4*ii:4*(ii+3)] = dVdu[ii,:,:]
        deltas = M.T@np.linalg.inv(M@M.T)@bb
        deltas = np.reshape(deltas, (N,4))
        sigma = 1
        
        correctedInitialEpoches = correctedInitialEpoches + sigma*deltas[:,3]
        correctedInitialStates[:,0:3] = correctedInitialStates[:,0:3] + sigma*deltas[:,0:3]

        iterationNumberLevelTwo = iterationNumberLevelTwo + 1
        # stop after too many iterations
        if iterationNumberLevelTwo > iterationNumberLevelTwoMax:
            exitflag = -1
            print('Outer loop shooting exceeds maximum iteration number '+str(iterationNumberLevelTwoMax))
            break

    return correctedInitialEpoches, correctedInitialStates, exitflag


def positionShooting(initialEpoch, initialState, targetEpoch, targetState, positionTolerance, GM):

    iterationNumberMax = 50
    iterationNumber = 1
    deltaR = 1
    phi0 = np.identity(6)
    phi0 = np.reshape(phi0, (36,1))
    while np.linalg.norm(deltaR) > positionTolerance:
        # calculate state transition matrix
        state0 = np.append(initialState, phi0)
        times, states = statePropFFI(np.array([initialEpoch, targetEpoch]), state0, GM)

        finalState = states[-1, 0:6]
        STM = np.reshape(states[-1, 6:], (6,6))
        
        # check if target is reached
        Rstar = targetState[0:3]
        deltaR = Rstar - finalState[0:3]

        # test early stop
        if np.linalg.norm(deltaR) < positionTolerance:
            exitflag = 1
            break

        B = STM[0:3, 3:6]

        correctionAtInitialState = np.linalg.inv(B)@deltaR

        # update state for next iteration
        sigma = 1
        initialState[3:6] = initialState[3:6] + sigma * correctionAtInitialState[0:3]
        iterationNumber = iterationNumber + 1
        
        # stop after too many iterations
        if iterationNumber > iterationNumberMax:
            exitflag = -1
            print('Position shooting maximum iteration reached')
            break

    return initialState, finalState, STM, exitflag


def ffInertial(tt, w, GM, radii=None, uT=None, times=None):

    x = w[0]
    y = w[1]
    z = w[2]
    
    r_sc = w[0:3]
    v_sc = w[3:6]
    phi = w[6:]

    r_Moon = spice.spkpos('Moon', tt, 'J2000', 'None', 'Moon')[0]
    r_Earth = spice.spkpos('Earth', tt, 'J2000', 'None', 'Moon')[0]
    r_Sun = spice.spkpos('Sun', tt, 'J2000', 'None', 'Moon')[0]

    r_bodies = np.vstack((r_sc, np.vstack((-r_Earth, -r_Sun))))
    
    f_Moon = -GM[0]*(r_sc - r_Moon)/np.linalg.norm(r_sc - r_Moon)**3
    f_Earth = -GM[1]*(r_sc - r_Earth)/np.linalg.norm(r_sc - r_Earth)**3 - GM[1]*r_Earth/np.linalg.norm(r_Earth)**3
    f_Sun = -GM[2]*(r_sc - r_Sun)/np.linalg.norm(r_sc - r_Sun)**3 - GM[2]*r_Sun/np.linalg.norm(r_Sun)**3
    
    Fg = f_Moon + f_Earth + f_Sun

    if len(w) > 6:
        Z = np.zeros((3,3))
        I = np.identity(3)
        U = np.zeros((3,3))
        for ii in np.arange(len(GM)):
            P = r_bodies[ii]
            px = P[0]
            py = P[1]
            pz = P[2]
            p3 = ((px)**2 + (py)**2 + (pz)**2)**(3/2)
            p5 = ((px)**2 + (py)**2 + (pz)**2)**(5/2)
            GM3 = 3*GM[ii]
            
            U[0,0] = U[0,0] + GM3*(px)**2/(p5) - GM[ii]/p3
            U[0,1] = U[0,1] + GM3*(py)*(px)/p5
            U[0,2] = U[0,2] + GM3*(pz)*(px)/p5
            U[1,0] = U[0,1]
            U[1,1] = U[1,1] + GM3*(py)**2/(p5) - GM[ii]/p3
            U[1,2] = U[1,2] + GM3*(pz)*(py)/p5
            U[2,0] = U[0,2]
            U[2,1] = U[1,2]
            U[2,2] = U[2,2] + GM3*(pz)**2/(p5) - GM[ii]/p3

        J1 = np.hstack((Z, I))
        J2 = np.hstack((U, Z))
        J = np.vstack((J1, J2))
        
        dPhi = J@np.reshape(phi, (6,6))
        dPhi = np.reshape(dPhi, (1,36))[0]

        dw = np.hstack((np.hstack((v_sc, Fg)), dPhi))
    else:
        dw = np.hstack((v_sc, Fg))
        
    return dw


def statePropFFI(Ts,state0,GM):
    ti = Ts[0]
    tf = Ts[1]
    
    sol_int = solve_ivp(ffInertial, [ti, tf], state0, args=(GM,), rtol=1E-12, atol=1E-12, method='LSODA')

    states = sol_int.y.T
    times = sol_int.t
    
    return times, states


def multipleShootingIForced(initialEpoches, initialStates, positionTolerance, velocityTolerance, GM, uT, timesInterp, omega_m):

    iterationNumberLevelTwoMax = 10

    N = len(initialEpoches)

    iterationNumberLevelTwo = 1
    correctedInitialEpoches = initialEpoches.copy()
    correctedInitialStates = initialStates.copy()
    correctedFinalStates = initialStates[1:,:].copy()
    
    deltaV = 1
    while deltaV > velocityTolerance:
        stateTransitionMatrixes = np.zeros((N,6,6))
        exitflag = np.zeros((N,1))
        exitFlagLevel1 = np.zeros((N-1,1))
        correctedFinalStates = np.zeros((N-1,6))
        STMs = np.zeros((N-1,6,6))
        print('Inner loop: position shooting')
        for ii in np.arange(N-1):
            cInitial, cFinal, STM, exitFlag1 = positionShootingForced(correctedInitialEpoches[ii], correctedInitialStates[ii,:], correctedInitialEpoches[ii+1], correctedInitialStates[ii+1,:], positionTolerance, GM, uT, timesInterp, omega_m)
            correctedInitialStates[ii,:] = cInitial
            correctedFinalStates[ii,:] = cFinal
            STMs[ii,:,:] = STM
            exitFlagLevel1[ii] = exitFlag1
            if not exitFlagLevel1[ii]:
                print('     Segment '+str(ii)+'/'+str(N-2)+' fails at position shooting')
                deltaV = -1
                break
            else:
                print('     Segment '+str(ii)+'/'+str(N-2)+' done')

        
        # test failure
        if np.any(exitFlagLevel1 != 1):
            exitflag = -2
            break
        
        #---- level-2 shooting ----
        print('Outer loop: velocity matching')
        # collcect the target error
        deltaVelocity1 = correctedFinalStates[:-1,3:6] - correctedInitialStates[1:-1,3:6]
        deltaVelocity = np.reshape(deltaVelocity1,(1,3*(N-2)))[0]
        deltaVelocities = np.linalg.norm(deltaVelocity1, axis=1)
        deltaV = sum(deltaVelocities)
        print('     Iteration '+str(iterationNumberLevelTwo)+' norm: '+str(deltaV))
        
        # test early stop
        if deltaV < velocityTolerance:
            exitflag = 1
            print('Multi-shooting success!')
            break
        
        # modify epoch and velocity of all segments at once
        # after one modification, shooting position again
        dVdu = np.zeros((N-2,3,12))         # for all the interior patch points 1 to N-2
        for ii in np.arange(1,N-1):
            # generate state relationship matrix
            stm21 = STMs[ii-1, :, :]
            stm12 = np.linalg.inv(stm21)
            stm32 = STMs[ii, :, :]

            v1plus  = correctedInitialStates[ii-1, 3:6]
            v2minus = correctedFinalStates[ii-1, 3:6]
            v2plus  = correctedInitialStates[ii, 3:6]
            v3minus = correctedFinalStates[ii, 3:6]

            a2minus = ffInertialForced(correctedInitialEpoches[ii], correctedFinalStates[ii-1, :], GM, uT, timesInterp, omega_m)
            a2minus = a2minus[3:6]
            a2plus  = ffInertialForced(correctedInitialEpoches[ii], correctedInitialStates[ii, :], GM, uT, timesInterp, omega_m)
            a2plus = a2plus[3:6]

            dVdu1 = -np.linalg.inv(stm12[0:3,3:6])
            dVdu2 = np.linalg.inv(stm12[0:3,3:6])@v1plus
            dVdu3 = -np.linalg.inv(stm32[0:3,3:6])@stm32[0:3,0:3] + np.linalg.inv(stm12[0:3,3:6])@stm12[0:3,0:3]
            dVdu4 = (a2plus-a2minus) + (np.linalg.inv(stm32[0:3,3:6])@stm32[0:3,0:3]@v2plus - np.linalg.inv(stm12[0:3,3:6])@stm12[0:3,0:3]@v2minus)
            dVdu5 = np.linalg.inv(stm32[0:3,3:6])
            dVdu6 = -np.linalg.inv(stm32[0:3,3:6])@v3minus
            
            dVdu[ii-1,:,0:3] = dVdu1
            dVdu[ii-1,:,3] = dVdu2
            dVdu[ii-1,:,4:7] = dVdu3
            dVdu[ii-1,:,7] = dVdu4
            dVdu[ii-1,:,8:11] = dVdu5
            dVdu[ii-1,:,11] = dVdu6
          
        bb = deltaVelocity
        M = np.zeros((len(bb), 4*(N)))
        for ii in np.arange(0,N-2):
            M[3*(ii):3*(ii+1),4*ii:4*(ii+3)] = dVdu[ii,:,:]
        deltas = M.T@np.linalg.inv(M@M.T)@bb
        deltas = np.reshape(deltas, (N,4))
        sigma = 1
        
        correctedInitialEpoches = correctedInitialEpoches + sigma*deltas[:,3]
        correctedInitialStates[:,0:3] = correctedInitialStates[:,0:3] + sigma*deltas[:,0:3]
        
        iterationNumberLevelTwo = iterationNumberLevelTwo + 1
        # stop after too many iterations
        if iterationNumberLevelTwo > iterationNumberLevelTwoMax:
            exitflag = -1
            print('Outer loop shooting exceeds maximum iteration number '+str(iterationNumberLevelTwoMax))
            break

    return correctedInitialEpoches, correctedInitialStates, exitflag, correctedFinalStates
    

def positionShootingForced(initialEpoch, initialState, targetEpoch, targetState, positionTolerance, GM, uT, timesInterp, omega_m):

    iterationNumberMax = 50
    iterationNumber = 1
    deltaR = 1
    phi0 = np.identity(6)
    phi0 = np.reshape(phi0, (36,1))
    while np.linalg.norm(deltaR) > positionTolerance:
        # calculate state transition matrix
        state0 = np.append(initialState, phi0)
        times, states = statePropFFIForced(np.array([initialEpoch, targetEpoch]), state0, GM, uT, timesInterp, omega_m)

        finalState = states[-1, 0:6]
        STM = np.reshape(states[-1, 6:], (6,6))
        
        # check if target is reached
        Rstar = targetState[0:3]
        deltaR = Rstar - finalState[0:3]

        # test early stop
        if np.linalg.norm(deltaR) < positionTolerance:
            exitflag = 1
            break

        B = STM[0:3, 3:6]

        correctionAtInitialState = np.linalg.inv(B)@deltaR

        # update state for next iteration
        sigma = 1
        initialState[3:6] = initialState[3:6] + sigma * correctionAtInitialState[0:3]
        iterationNumber = iterationNumber + 1
        
        # stop after too many iterations
        if iterationNumber > iterationNumberMax:
            exitflag = -1
            print('Position shooting maximum iteration reached')
            break

    return initialState, finalState, STM, exitflag
    
    
def ffInertialForced(tt, w, GM, uT=None, times=None, omega_m=None):

    x = w[0]
    y = w[1]
    z = w[2]
    
    r_sc = w[0:3]
    v_sc = w[3:6]
    phi = w[6:]

    r_Moon = spice.spkpos('Moon', tt, 'J2000', 'None', 'Moon')[0]
    r_Earth = spice.spkpos('Earth', tt, 'J2000', 'None', 'Moon')[0]
    r_Sun = spice.spkpos('Sun', tt, 'J2000', 'None', 'Moon')[0]

    r_bodies = np.vstack((r_sc, np.vstack((-r_Earth, -r_Sun))))
    
    f_Moon = -GM[0]*(r_sc - r_Moon)/np.linalg.norm(r_sc - r_Moon)**3
    f_Earth = -GM[1]*(r_sc - r_Earth)/np.linalg.norm(r_sc - r_Earth)**3 - GM[1]*r_Earth/np.linalg.norm(r_Earth)**3
    f_Sun = -GM[2]*(r_sc - r_Sun)/np.linalg.norm(r_sc - r_Sun)**3 - GM[2]*r_Sun/np.linalg.norm(r_Sun)**3

    Fg = f_Moon + f_Earth + f_Sun
    
    if np.any(uT):
        u_ii = linInterp(times, uT, tt)
        Crv_R2I = spice.sxform('MCR','MCI',tt)
        f_T = Crv_R2I[0:3,0:3]@u_ii
#        wR = Crv_R2I@w[0:6]
#        rotVec = np.array([0, 0, omega_m])
#        try:
#            tmp1 = 2*np.cross(rotVec, wR[3:6])
#            tmp2 = np.cross(rotVec, np.cross(rotVec, wR[0:3]))
#        except:
#            breakpoint()
#        f_T = u_ii + tmp1 + tmp2
        Fg = Fg + f_T

    if len(w) > 6:
        Z = np.zeros((3,3))
        I = np.identity(3)
        U = np.zeros((3,3))
        for ii in np.arange(len(GM)):
            P = r_bodies[ii]
            px = P[0]
            py = P[1]
            pz = P[2]
            p3 = ((px)**2 + (py)**2 + (pz)**2)**(3/2)
            p5 = ((px)**2 + (py)**2 + (pz)**2)**(5/2)
            GM3 = 3*GM[ii]
            
            U[0,0] = U[0,0] + GM3*(px)**2/(p5) - GM[ii]/p3
            U[0,1] = U[0,1] + GM3*(py)*(px)/p5
            U[0,2] = U[0,2] + GM3*(pz)*(px)/p5
            U[1,0] = U[0,1]
            U[1,1] = U[1,1] + GM3*(py)**2/(p5) - GM[ii]/p3
            U[1,2] = U[1,2] + GM3*(pz)*(py)/p5
            U[2,0] = U[0,2]
            U[2,1] = U[1,2]
            U[2,2] = U[2,2] + GM3*(pz)**2/(p5) - GM[ii]/p3

        J1 = np.hstack((Z, I))
        J2 = np.hstack((U, Z))
        J = np.vstack((J1, J2))
        
        dPhi = J@np.reshape(phi, (6,6))
        dPhi = np.reshape(dPhi, (1,36))[0]

        dw = np.hstack((np.hstack((v_sc, Fg)), dPhi))
    else:
        dw = np.hstack((v_sc, Fg))
        
    return dw

def linInterp(times, uT, currentTime):

    if currentTime >= times[-1]:
        u_current = uT[-1,:]
        return u_current
    elif currentTime <= times[0]:
        u_current = uT[0,:]
        return u_current

    min_ind = np.argwhere(currentTime >= times)[:,0][-1]
    u_current = uT[min_ind,:] + (currentTime - times[min_ind])*(uT[min_ind+1,:] - uT[min_ind,:])/(times[min_ind+1] - times[min_ind])
    
    return u_current

def statePropFFIForced(Ts,state0,GM,uT,times,omega_m):
    ti = Ts[0]
    tf = Ts[1]
    
    sol_int = solve_ivp(ffInertialForced, [ti, tf], state0, args=(GM,uT,times,omega_m), rtol=1E-12, atol=1E-12, method='LSODA')

    states = sol_int.y.T
    times = sol_int.t
    
    return times, states

def getPatches(N, times_dim, times_mjd, pos_dim, vel_dim):

    dt_int = (times_dim[-1]-times_dim[0])/(N-1)
    taus = Time(np.zeros(N), format='mjd', scale='utc')
    posvel = np.array([])
    for ii in np.arange(N):
        time_i = ii*dt_int

        # find the index
        difference_array_i = np.absolute(times_dim-time_i).value
        index_i = difference_array_i.argmin()
        
        # index the time, position, and velocity
        taus[ii] = times_mjd[index_i]
        pos_i = pos_dim[index_i,:]
        vel_i = vel_dim[index_i,:]
        
        # package the state
        state_i = np.append(pos_i, vel_i)
        posvel = np.append(posvel, state_i)
    posvel = np.reshape(posvel,(N,6))
    
    return posvel, taus

def hitMoon(tt, w, GM, radii, uT=None, times=None):
    r_scM = w[0:3]

    if np.linalg.norm(r_scM) < radii[0]:
        moonCrash = 0
    else:
        moonCrash = 1
        
    return moonCrash
    
def hitEarth(tt, w, GM, radii, uT=None, times=None):
    r_scM = w[0:3]

    r_Earth = spice.spkpos('Earth', tt, 'J2000', 'None', 'Moon')[0]
    
    r_scE = r_scM - r_Earth

    if np.linalg.norm(r_scE) < radii[1]:
        earthCrash = 0
    else:
        earthCrash = 1
        
    return earthCrash

def hitSun(tt, w, GM, radii, uT=None, times=None):
    r_scM = w[0:3]

    r_Sun = spice.spkpos('Sun', tt, 'J2000', 'None', 'Moon')[0]
    
    r_scS = r_scM - r_Sun

    if np.linalg.norm(r_scS) < radii[2]:
        sunCrash = 0
    else:
        sunCrash = 1
        
    return sunCrash
    
def lostShape(tt, w, GM, radii, uT=None, times=None):
    r_scM = w[0:3]
    
    if np.linalg.norm(r_scM) > 1.5*radii[-1]:
        chaoticBehavior = 0
    else:
        chaoticBehavior = 1
        
    return chaoticBehavior

