import numpy as np
import sys
from astropy.time import Time
import astropy.units as u
from scipy.integrate import solve_ivp
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
from scipy.optimize import fsolve

spice.furnsh("fullForce.txt")

showPlots = False
# Parameters
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
mu_star = gmMoon/(gmEarth + gmMoon)
m1 = (1 - mu_star)
m2 = mu_star

radiiMoon = spice.bodvrd( 'Moon', 'RADII', 3 )[1][0]
rMoon = unitConversion.convertPos_to_canonical(radiiMoon*u.km)

# Initial condition in canonical units in rotating frame R [pos, vel]
IC = [1.0118, 0, 0.1739, 0, -0.0799, 0, 1.3743]     # L2 Northern
#IC = [1.0118, 0, -0.1739, 0, -0.0799, 0, 1.3743]     # L2 Southern

#IC = [((1 - mu_star) - 0.023413), 0, 0, 0, 0.720544, 0, 0.102081]

#IC = [1.01103506347211, 0, -0.17315001039682773, 0, -0.07801414771853428, 0, 1.363209636932144/2]  #L2, 5.92773293-day period
#IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]   # DRO, 0.9697497-day period
#IC = [0.429519110229904, 0, 0, 0, 1.440796689672539, 0, 3.051133070334277] # DRO
#IC = [0.517332653163958, 0, 0, 0, 1.12965881302616, 0, 8.50664047891897] # P3DRO, fails miserably
#IC = [1.165130674583613, 0, -0.110699848144854, 0, 0.201519926517907, 0, 1.652428300688599]
#IC = [1.114959432252717, 0, 0.027057507726036, 0, 0.191674660415012, 0, 3.403442494940593/2]   # matlab
#IC = [1.11495, 0, 0.02705, 0, 0.19167, 0, 3.40344/2]   # matlab
#IC = [0.856382122325864, 0, -0.181519309916197, 0, 0.257898218422393, 0, 1.22727308466325]  # L1
#IC = [1.06896234204296, 0, 0.159599443574046, 0, -0.00769167653854165, 0, 1.66142030228280] # butterfly
#IC = [0.766044481790803, 0, 0, 0, 0.488736680662207, 0, 2.20546980585774]   # L1 lyapunov
#IC = [0.265819894849149, 0, 0, 0, 2.27750677757506, 0, 6.25588866460133]    # 2:1 resonant, fails miserably
#IC = [0.139106790847531, 0, 0, 0, 3.35999055380076, 0, 9.40977341640670]    # 2:3 resonant, fails miserably

# Generate new ICs using the free variable and constraint method
X = [IC[0], IC[2], IC[4], IC[6]]
max_iter = 50
error = 10
eps = 1E-6
step = 0.01
Tp_max = unitConversion.convertTime_to_canonical(14*u.d)
goodSols = np.array([])
Nsols = 0
while X[-1] < Tp_max:
    ctr = 0
    error = 10
    z = np.array([0, 0, 0, 1])
    while error > eps and ctr < max_iter:
        Fx = ss.calcFx_R(X, mu_star)

        error = np.linalg.norm(Fx)
        if error < eps:
            print('Error is: '+str(error))
            break
            
        dFx = ss.calcdFx_CRTBP(X, mu_star, m1, m2)

        X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)

        ctr = ctr + 1
        print('Error is: '+str(error))

    print('Number of attempts: '+str(ctr))
    Nsols = Nsols + 1

    # Propagate the dynamics (states in AU or AU/day, times in days starting from 0)
    freeVar0CRTBP_R = X.copy()
#    freeVar0CRTBP_R[-1] = 2*freeVar0CRTBP_R[-1]
    statesCRTBP_R, timesCRTBP_R = orbitEOMProp.statePropCRTBP_R(freeVar0CRTBP_R, mu_star)  # State is in the R frame
    posCRTBP_R = statesCRTBP_R[:, 0:3]
    velCRTBP_R = statesCRTBP_R[:, 3:6]
    
    rmag = np.linalg.norm(posCRTBP_R,axis=1)
    if np.any(rmag < rMoon):
        print('Intersects moon. Not a solution')
    else:
        if showPlots:
            ax1 = plt.figure().add_subplot(projection='3d')
            ax1.plot(posCRTBP_R[:,0], posCRTBP_R[:,1], posCRTBP_R[:,2])
            plt.show()
        
        sol0 = np.append(statesCRTBP_R[0,:], timesCRTBP_R[-1])
        goodSols = np.append(goodSols, sol0)
    
    # Generate new z and X for another orbit
    solp = X + z * step
    fss = fsolve(orbitEOMProp.fsolve_eqns, X, args=(z, solp, mu_star), full_output=True, xtol=1E-12)
    X = fss[0]
    Q = fss[1]['fjac']
    Rs = fss[1]['r']
    R = np.zeros((4, 4))
    idx, col = np.triu_indices(4, k=0)
    R[idx, col] = Rs
    J = Q.T @ R

    z = np.linalg.inv(J) @ z
    z = z / np.linalg.norm(z)

goodSols = np.reshape(goodSols, (Nsols, 7))
goodSols = goodSols[1:,:]
states = goodSols[:,0:6]
periods = goodSols[:,7]

np.savez('/Users/gracegenszler/Documents/Research/starlift/orbits/graveyardOrbits/L2_Northern.npz', states = states, periods = periods, mu_star = mu_star)
print(Nsols)
breakpoint()
