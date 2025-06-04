import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp

def multipleShooting(initialEpoches, initialStates, positionTolerance, velocityTolerance, GM):

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
        print('level-1 position shooting')
        for ii in np.arange(N-1):
            cInitial, cFinal, STM, exitFlag1 = positionShooting(correctedInitialEpoches[ii], correctedInitialStates[ii,:], correctedInitialEpoches[ii+1], correctedInitialStates[ii+1,:], positionTolerance, GM)
            correctedInitialStates[ii,:] = cInitial
            correctedFinalStates[ii,:] = cFinal
            STMs[ii,:,:] = STM
            exitFlagLevel1[ii] = exitFlag1
            if not exitFlagLevel1[ii]:
                print('# !!! fail: segment '+str(ii)+' fails at the level-1 shooting.')
                deltaV = -1
                break
            else:
                print('#      segment '+str(ii)+' done.')

        print('#    level-1 done.')
        
        # plot level-1 shooting results
#        figure(99); clf; PlotInitialState(dynamicFcn, correctedInitialEpoches, correctedInitialStates);
        
        # test failure
        if np.any(exitFlagLevel1 != 1):
            exitflag = -2
            break
        
        #---- level-2 shooting ----
        # collcect the target error
        deltaVelocity = correctedFinalStates[:-1,3:6] - correctedInitialStates[1:-1,3:6]
        deltaVelocity = np.reshape(deltaVelocity,(1,3*(N-2)))[0]
        deltaV = np.linalg.norm(deltaVelocity)
        print('#  level-2 iter '+str(iterationNumberLevelTwo)+' norm: '+str(deltaV))
        
        # test early stop
        if deltaV < velocityTolerance:
            exitflag = 1
            print('# multiple-shooting success. norm(dV) = '+str(deltaV))
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
        breakpoint()
        # stop after too many iterations
        if iterationNumberLevelTwo > iterationNumberLevelTwoMax:
            exitflag = -1
            print('#  !!! fail: level-2 shooting exceeds maximum iteration number '+str(iterationNumberLevelTwoMax)+'.')
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
        times, states = statePropFF(np.array([initialEpoch, targetEpoch]), state0, GM)

        finalState = states[-1, 0:6]
        STM = np.reshape(states[-1, 6:], (6,6))
        
        # check if target is reached
        Rstar = targetState[0:3]
        deltaR = Rstar - finalState[0:3]
        #disp(['debug: position shooting: iter ' num2str(iterationNumber) ': error is ' num2str(norm(errorFinalState(1:3)))]);

        # test early stop
        if np.linalg.norm(deltaR) < positionTolerance:
            exitflag = 1
            break
        # you left off here. Continue fixing syntax. Including changing function names
        # solve for the correction at initial state
        B = STM[0:3, 3:6]

        correctionAtInitialState = np.linalg.inv(B)@deltaR

        # update state for next iteration
        sigma = 0.618
        initialState[3:6] = initialState[3:6] + sigma * correctionAtInitialState[0:3]
        iterationNumber = iterationNumber + 1
        
        # stop after too many iterations
        if iterationNumber > iterationNumberMax:
            exitflag = -1
            print('position shooting: max iteration reached.')
            break

    return initialState, finalState, STM, exitflag


def ffInertial(tt, w, GM):

    x = w[0]
    y = w[1]
    z = w[2]
    
    r_sc = w[0:3]
    v_sc = w[3:6]
    phi = w[6:]

    r_Moon = spice.spkpos('Moon', tt, 'J2000', 'None', 'Moon')[0]
    r_Earth = spice.spkpos('Earth', tt, 'J2000', 'None', 'Moon')[0]
    r_Sun = spice.spkpos('Sun', tt, 'J2000', 'None', 'Moon')[0]

    r_bodies = np.vstack((r_Moon, np.vstack((r_Earth, r_Sun))))
    
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
            p3 = ((px - x)**2 + (py - y)**2 + (pz - z)**2)**(3/2)
            p5 = ((px - x)**2 + (py - y)**2 + (pz - z)**2)**(5/2)
            GM3 = 3*GM[ii]
            
            U[0,0] = U[0,0] + GM3*(px - x)**2/(p5) - GM[ii]/p3
            U[0,1] = U[0,1] + GM3*(py - y)*(px - x)/p5
            U[0,2] = U[0,2] + GM3*(pz - z)*(px - x)/p5
            U[1,0] = U[0,1]
            U[1,1] = U[1,1] + GM3*(py - y)**2/(p5) - GM[ii]/p3
            U[1,2] = U[1,2] + GM3*(pz - z)*(py - y)/p5
            U[2,0] = U[0,2]
            U[2,1] = U[1,2]
            U[2,2] = U[2,2] + GM3*(pz - z)**2/(p5) - GM[ii]/p3

        J1 = np.hstack((Z, I))
        J2 = np.hstack((U, Z))
        J = np.vstack((J1, J2))
        
        dPhi = J@np.reshape(phi, (6,6))
        dPhi = np.reshape(dPhi, (1,36))[0]

        dw = np.hstack((np.hstack((v_sc, Fg)), dPhi))
    else:
        dw = np.hstack((v_sc, Fg))
        
    return dw


def statePropFF(Ts,state0,GM):
    ti = Ts[0]
    tf = Ts[1]
    
    sol_int = solve_ivp(ffInertial, [ti, tf], state0, args=(GM,), rtol=1E-12, atol=1E-12, method='LSODA')

    states = sol_int.y.T
    times = sol_int.t
    
    return times, states
