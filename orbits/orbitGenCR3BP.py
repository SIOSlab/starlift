import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp

import pdb

mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

IC = [1.011035058791837, 0, -0.173149999816971, 0, -0.078014304750278, 0, 0.681604842920932]
X = [IC[0], IC[2], IC[4], IC[6]]

eps = 1E-6
N = 10
solutions = np.zeros([N,4])
z = np.array([0, 0, 0, 1])
step = 1E-2

max_iter = 1000
ax = plt.figure().add_subplot(projection='3d')
for ii in np.arange(N):
    error = 10
    ctr = 0
    while error > eps and ctr < max_iter:
        Fx = orbitEOMProp.calcFx_R(X, mu_star)

        error = np.linalg.norm(Fx)

        dFx = orbitEOMProp.calcdFx(X,mu_star,m1,m2)

        X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
        
        ctr = ctr + 1
        
    IV = np.array([X[0], X[1], X[2], 2*X[3]])
    solutions[ii] = IV
    states, times = orbitEOMProp.stateProp_R(IV,mu_star)
    
    ax.plot(states[:,0],states[:,1],states[:,2])

    solp = X + z*step
    ss = fsolve(orbitEOMProp.fsolve_eqns,X,args=(z,solp,mu_star),full_output=True,xtol=1E-12)
    X = ss[0]
    Q = ss[1]['fjac']
    Rs = ss[1]['r']
    R = np.zeros((4, 4))
    idx, col = np.triu_indices(4, k=0)
    R[idx, col] = Rs
    J = Q.T@R

    z = np.linalg.inv(J)@z
    z = z/np.linalg.norm(z)
    
plt.show()
#breakpoint()
