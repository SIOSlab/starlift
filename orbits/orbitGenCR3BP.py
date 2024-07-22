import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, 'tools')
import orbitEOMProp
import pdb

guess = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]
mu_star = 1.215059*10**(-2)
N = 10
ICs = orbitEOMProp.generateFamily_CRTBP(guess, mu_star, N)

ax = plt.figure().add_subplot(projection='3d')
for ii in np.arange(N):
    ax.plot(ICs[ii, 0], ICs[ii, 1], ICs[ii, 2])

plt.show()

breakpoint()


# # Parameters
# mu_star = 1.215059*10**(-2)
# m1 = (1 - mu_star)
# m2 = mu_star
#
# # Initial guess for the free variable vector
# IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period
# X = [IC[0], IC[2], IC[4], IC[6]]
#
# eps = 1E-6
# N = 10
# solutions = np.zeros([N, 4])
# z = np.array([0, 0, 0, 1])
# step = 1E-2
#
# max_iter = 1000
# ax = plt.figure().add_subplot(projection='3d')
# for ii in np.arange(N):
#     error = 10
#     ctr = 0
#     while error > eps and ctr < max_iter:
#         # Generate the free variable vector
#         Fx = orbitEOMProp.calcFx_R(X, mu_star)
#         error = np.linalg.norm(Fx)
#         dFx = orbitEOMProp.calcdFx_CRTBP(X, mu_star, m1, m2)
#         X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)
#         ctr = ctr + 1
#
#     # Generate an orbit from the found free variable vector
#     IV = np.array([X[0], X[1], X[2], 2 * X[3]])
#     solutions[ii] = IV
#     states, times = orbitEOMProp.statePropCRTBP_R(IV, mu_star)
#
#     # Plot the orbit
#     ax.plot(states[:, 0], states[:, 1], states[:, 2])
#
#     # Generate new z and X for another orbit
#     solp = X + z * step
#     ss = fsolve(orbitEOMProp.fsolve_eqns, X, args=(z, solp, mu_star), full_output=True, xtol=1E-12)
#     X = ss[0]
#     Q = ss[1]['fjac']
#     Rs = ss[1]['r']
#     R = np.zeros((4, 4))
#     idx, col = np.triu_indices(4, k=0)
#     R[idx, col] = Rs
#     J = Q.T @ R
#
#     z = np.linalg.inv(J) @ z
#     z = z / np.linalg.norm(z)
#
# plt.show()

