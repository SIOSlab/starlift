import numpy as np
import os.path
import pickle
from scipy.integrate import solve_ivp
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
import gmatTools
import plot_tools

#import tools.unitConversion as unitConversion
#import tools.frameConversion as frameConversion
#import tools.orbitEOMProp as orbitEOMProp
#import tools.plot_tools as plot_tools
import pdb

# ~~~~~PROPAGATE THE DYNAMICS  ~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de440')

# Parameters
t_veq = Time(64041, format='mjd', scale='utc')
t_start = Time(64041, format='mjd', scale='utc')
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

C_I2G = frameConversion.inert2geo(t_start, t_veq)
C_G2I = C_I2G
C_I2R = frameConversion.rot(np.pi/6,3)

S1_H = get_body_barycentric_posvel('Sun', t_veq)[0].get_xyz().to('AU')
S1_Gt = frameConversion.icrs2gmec(S1_H, t_veq).to('AU')
B1_H = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_veq)[0].get_xyz().to('AU')
B1_Gt = frameConversion.icrs2gmec(B1_H, t_veq).to('AU')
S1_G = S1_Gt - B1_Gt
g1 = S1_G/np.linalg.norm(S1_G)*.003

S2_H = get_body_barycentric_posvel('Sun', t_veq+.25*u.yr)[0].get_xyz().to('AU')
S2_Gt = frameConversion.icrs2gmec(S2_H, t_veq+.25*u.yr).to('AU')
B2_H = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_veq+.25*u.yr)[0].get_xyz().to('AU')
B2_Gt = frameConversion.icrs2gmec(B2_H, t_veq+.25*u.yr).to('AU')
S2_G = S2_Gt - B2_Gt
g2 = S2_G/np.linalg.norm(S2_G)*.003

G3 = np.cross(g1,g2)
g3 = G3/np.linalg.norm(G3)*.003

i1 = C_G2I @ g1
i2 = C_G2I @ g2

M_H = get_body_barycentric_posvel('Moon', t_veq)[0].get_xyz().to('AU')
M_Gt = frameConversion.icrs2gmec(M_H, t_veq).to('AU')
M_G = M_Gt - B1_Gt
M_I = C_G2I @ M_Gt
M_R = C_I2R @ M_I

r1 = C_I2R @ i1
r2 = C_I2R @ i2


plt.rcParams.update({'font.size': 10})
ax1 = plt.figure().add_subplot(projection='3d')
ax1.scatter(-B1_Gt[0], -B1_Gt[1],0, c='g', marker='o', s=40, label='Earth')
ax1.scatter(M_R[0], M_R[1], c='b', marker='o', s=10, label='Moon')

ax1.quiver([0],[0],0,i1[0],i1[1],0, arrow_length_ratio=0.1)
ax1.text(i1[0],i1[1],0, '$\hat{i}_1$')
ax1.quiver([0],[0],0,i2[0],i2[1],0, arrow_length_ratio=0.1)
ax1.text(i2[0],i2[1],0, '$\hat{i}_2$')

ax1.quiver([0],[0],0,r1[0],r1[1],0, arrow_length_ratio=0.1)
ax1.text(r1[0],r1[1],0, '$\hat{r}_1$')
ax1.quiver([0],[0],0,r2[0],r2[1],0, arrow_length_ratio=0.1)
ax1.text(r2[0],r2[1],0, '$\hat{r}_2$')


ax1.set_xlabel('X [AU]')
ax1.set_ylabel('Y [AU]')
ax1.set_zlabel('Z [AU]')
ax1.set_xlim([-.001, .004])
ax1.set_ylim([-.001, .004])
ax1.set_zlim([-.002, .003])
plt.legend()


#S1_H = get_body_barycentric_posvel('Sun', t_veq)[0].get_xyz().to('AU')
#S1_Gt = frameConversion.icrs2gmec(S1_H, t_veq).to('AU')
#B1_H = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_veq)[0].get_xyz().to('AU')
#B1_Gt = frameConversion.icrs2gmec(B1_H, t_veq).to('AU')
#S1_G = S1_Gt - B1_Gt
#g1 = S1_G/np.linalg.norm(S1_G)*.003
#
#S2_H = get_body_barycentric_posvel('Sun', t_veq+.25*u.yr)[0].get_xyz().to('AU')
#S2_Gt = frameConversion.icrs2gmec(S2_H, t_veq+.25*u.yr).to('AU')
#B2_H = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_veq+.25*u.yr)[0].get_xyz().to('AU')
#B2_Gt = frameConversion.icrs2gmec(B2_H, t_veq+.25*u.yr).to('AU')
#S2_G = S2_Gt - B2_Gt
#g2 = S2_G/np.linalg.norm(S2_G)*.003
#
#G3 = np.cross(g1,g2)
#g3 = G3/np.linalg.norm(G3)*.003
#
#i1 = C_G2I @ g1
#i2 = C_G2I @ g2
#I3 = np.cross(i1,i2)
#i3 = I3/np.linalg.norm(I3)*.003
#
#M_H = get_body_barycentric_posvel('Moon', t_veq)[0].get_xyz().to('AU')
#M_Gt = frameConversion.icrs2gmec(M_H, t_veq).to('AU')
#M_G = M_Gt - B1_Gt
#M_Gi = C_G2I @ M_G
#
#plt.rcParams.update({'font.size': 10})
#ax1 = plt.figure().add_subplot(projection='3d')
#ax1.scatter(-B1_Gt[0], -B1_Gt[1], -B1_Gt[2], c='g', marker='o', s=400, label='Earth')
#ax1.scatter(M_G[0], M_G[1], M_G[2], c='b', marker='o', s=100, label='Moon')
#
#ax1.quiver([0],[0],[0],g1[0],g1[1],g1[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(g1[0]+.0005,g1[1],g1[2], '$\hat{g}_1$')
#ax1.quiver([0],[0],[0],g2[0],g2[1],g2[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(g2[0],g2[1]+.0005,g2[2], '$\hat{g}_2$')
#ax1.quiver([0],[0],[0],g3[0],g3[1],g3[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(g3[0],g3[1],g3[2]+.0005, '$\hat{g}_3$')
#
#ax1.quiver([0],[0],[0],i1[0],i1[1],i1[2], colors='r', arrow_length_ratio=0.1)
#ax1.text(i1[0]+.0005,i1[1],i1[2], '$\hat{i}_1$')
#ax1.quiver([0],[0],[0],i2[0],i2[1],i2[2], colors='r', arrow_length_ratio=0.1)
#ax1.text(i2[0],i2[1]+.0005,i2[2], '$\hat{i}_2$')
#ax1.quiver([0],[0],[0],i3[0],i3[1],i3[2], colors='r', arrow_length_ratio=0.1)
#ax1.text(i3[0],i3[1],i3[2]+.0005, '$\hat{i}_3$')
#
#ax1.set_xlabel('X [AU]')
#ax1.set_ylabel('Y [AU]')
#ax1.set_zlabel('Z [AU]')
#ax1.set_xlim([-.001, .004])
#ax1.set_ylim([-.001, .004])
#ax1.set_zlim([-.002, .003])
#plt.legend()


#S1_H = get_body_barycentric_posvel('Sun', t_veq)[0].get_xyz().to('AU')
#S1_Gt = frameConversion.icrs2gmec(S1_H, t_veq).to('AU')
#B1_H = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_veq)[0].get_xyz().to('AU')
#B1_Gt = frameConversion.icrs2gmec(B1_H, t_veq).to('AU')
#S1_G = S1_Gt - B1_Gt
#g1 = S1_G/np.linalg.norm(S1_G)
#
#S2_H = get_body_barycentric_posvel('Sun', t_veq+.25*u.yr)[0].get_xyz().to('AU')
#S2_Gt = frameConversion.icrs2gmec(S2_H, t_veq+.25*u.yr).to('AU')
#B2_H = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_veq+.25*u.yr)[0].get_xyz().to('AU')
#B2_Gt = frameConversion.icrs2gmec(B2_H, t_veq+.25*u.yr).to('AU')
#S2_G = S2_Gt - B2_Gt
#g2 = S2_G/np.linalg.norm(S2_G)
#
#G3 = np.cross(g1,g2)
#g3 = G3/np.linalg.norm(G3)
#
#plt.rcParams.update({'font.size': 10})
#ax1 = plt.figure().add_subplot(projection='3d')
#ax1.scatter(-B1_Gt[0], -B1_Gt[1], -B1_Gt[2], c='g', marker='o', s=100, label='Earth')
#ax1.scatter(S1_G[0], S1_G[1], S1_G[2], c='y', marker='*', s=400, label='Sun')
#
#ax1.quiver([0],[0],[0],g1[0],g1[1],g1[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(g1[0],g1[1],g1[2]+.1, '$\hat{g}_1$')
#ax1.quiver([0],[0],[0],g2[0],g2[1],g2[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(g2[0],g2[1],g2[2]+.2, '$\hat{g}_2$')
#ax1.quiver([0],[0],[0],g3[0],g3[1],g3[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(g3[0]+.1,g3[1],g3[2], '$\hat{g}_3$')
#ax1.set_xlabel('X [AU]')
#ax1.set_ylabel('Y [AU]')
#ax1.set_zlabel('Z [AU]')
#ax1.set_xlim([-.5, 1.5])
#ax1.set_ylim([-.5, 1.5])
#ax1.set_zlim([-.5, 1.5])
#plt.legend()


#H1 = get_body_barycentric_posvel('Earth', t_veq)[0].get_xyz().to('AU').value
#h1 = H1/np.linalg.norm(H1)*1.5
#H2 = get_body_barycentric_posvel('Earth', t_veq + .25*u.yr)[0].get_xyz().to('AU').value
#h2 = H2/np.linalg.norm(H2)*1.5
#H3 = np.cross(h1,h2)
#h3 = H3/np.linalg.norm(H3)*1.5
#
#M1 = get_body_barycentric_posvel('Moon', t_veq)[0].get_xyz().to('AU').value
#S1 = get_body_barycentric_posvel('Sun', t_veq)[0].get_xyz().to('AU').value
#
#plt.rcParams.update({'font.size': 10})
#ax1 = plt.figure().add_subplot(projection='3d')
#ax1.scatter(H1[0], H1[1], H1[2], c='g', marker='o', s=100, label='Earth')
#ax1.scatter(S1[0], S1[1], S1[2], c='y', marker='*', s=400, label='Sun')
#
#ax1.quiver([0],[0],[0],h1[0],h1[1],h1[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(h1[0],h1[1],h1[2]+.1, '$\hat{h}_1$')
#ax1.quiver([0],[0],[0],h2[0],h2[1],h2[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(h2[0],h2[1],h2[2]+.2, '$\hat{h}_2$')
#ax1.quiver([0],[0],[0],h3[0],h3[1],h3[2], colors='k', arrow_length_ratio=0.1)
#ax1.text(h3[0]+.1,h3[1],h3[2], '$\hat{h}_3$')
#ax1.set_xlabel('X [AU]')
#ax1.set_ylabel('Y [AU]')
#ax1.set_zlabel('Z [AU]')
#ax1.set_xlim([-1.5, .5])
#ax1.set_ylim([-1.5, .5])
#ax1.set_zlim([-.5, 1.5])
#plt.legend()
plt.show()
breakpoint()
