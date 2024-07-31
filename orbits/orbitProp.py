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
#import plot_tools

#import tools.unitConversion as unitConversion
#import tools.frameConversion as frameConversion
#import tools.orbitEOMProp as orbitEOMProp
#import tools.plot_tools as plot_tools
import pdb

# ~~~~~PROPAGATE THE DYNAMICS  ~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de432s')

# Parameters
t_mjd = Time(57727, format='mjd', scale='utc')
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

#tarray = t_mjd + np.arange((1*u.yr).to('d').value)

#moon_G1 = np.zeros([len(tarray), 3])
#moon_G2 = np.zeros([len(tarray), 3])
#r_MB_I2s = np.zeros([len(tarray), 3])
#r_MB_LAAN = np.zeros([len(tarray), 3])
#r_MB_INC = np.zeros([len(tarray), 3])
ctr = 0
#C_G2B = frameConversion.body2geo(t_mjd,t_mjd,mu_star)
C_B2G, C_LAAN, C_INC, C_AOP, n_LAAN, n_INC, n_AOP = frameConversion.body2geo2(t_mjd,t_mjd)

C_G2B = C_B2G.T
#for ii in tarray:
##    C_B2G = frameConversion.body2geo(ii,t_mjd,mu_star)
#    moon_H = get_body_barycentric_posvel('Moon', ii)[0].get_xyz()
#    moon_1 = frameConversion.icrs2gcrs(moon_H,ii).to('AU').value
#
#    bary_H = get_body_barycentric_posvel('Earth-Moon-Barycenter', ii)[0].get_xyz()
#    bary_1 = frameConversion.icrs2gcrs(bary_H,ii).to('AU').value
#    moon_1 = moon_1 - bary_1
#
#    m_norm = np.linalg.norm(moon_1)
#    
#    r_moon_bary_R = np.array([m_norm, 0, 0])
#    
#    dt = ii.value - t_mjd.value
#    theta_BR = unitConversion.convertTime_to_canonical(dt*u.d)
#        
#    C_B2R = frameConversion.rot(theta_BR, 3)
#    C_R2B = C_B2R.T
#    
#    r_moon_bary_I = C_R2B @ r_moon_bary_R
#    
##    C_B2G = frameConversion.body2geo(ii,t_mjd,mu_star)
##    C_G2B = C_B2G.T
#    
##    r_MB_I2 = C_G2B @ moon_1
#    
#    r_LAAN = C_LAAN @ moon_1
#    r_INC = C_INC @ r_LAAN
#    r_MB_I2 = C_AOP @ r_INC
#
##    tmp = C_LAAN @ C_INC @ moon_1
##    breakpoint()
#    
##    r_MB_I3 = C_G2B @ C_I2B @ moon_1
##    
##    r_MB_I4 = C_G2B @ C_I2B.T @ moon_1
#    
#    m_1 = moon_1
#    r_I = r_moon_bary_I
#
##    ax1 = plt.figure().add_subplot(projection='3d')
##    ax1.plot([0, m_1[0]],[0, m_1[1]],[0, m_1[2]],'b', label='G frame')
##    ax1.plot([0, r_I[0]],[0, r_I[1]],[0, r_I[2]],'c', label='I frame')
##    ax1.plot([0, r_MB_I2[0]],[0, r_MB_I2[1]],[0, r_MB_I2[2]],'r-.', label='G2B')
##    ax1.plot([0, r_MB_I3[0]],[0, r_MB_I3[1]],[0, r_MB_I3[2]],'g.', label='G2B + I2B')
##    ax1.plot([0, r_MB_I4[0]],[0, r_MB_I4[1]],[0, r_MB_I4[2]],'k.', label='G2B + B2I')
##    ax1.plot([0, r_MB_I5[0]],[0, r_MB_I5[1]],[0, r_MB_I5[2]],'y-.', label='I2B + G2B')
##    ax1.plot([0, r_MB_I6[0]],[0, r_MB_I6[1]],[0, r_MB_I6[2]],'m-.', label='B2I + G2B')
##    plt.legend()
##    plt.show()
##    breakpoint()
#    moon_G1[ctr,:] = moon_1
#    r_MB_I2s[ctr,:] = r_MB_I2
#    r_MB_LAAN[ctr,:] = r_LAAN
#    r_MB_INC[ctr,:] = r_INC
###    moon_G2[ctr,:] = moon_2
#    ctr = ctr + 1
#
#n_LAAN = n_LAAN * max(np.linalg.norm(moon_G1,axis=1))
#n_INC = n_INC * max(np.linalg.norm(moon_G1,axis=1))
#n_AOP = n_AOP * max(np.linalg.norm(moon_G1,axis=1))
#
##ax1 = plt.figure().add_subplot(projection='3d')
##ax1.plot(moon_G1[:, 0], moon_G1[:, 1], moon_G1[:, 2], 'b', label='time ii')
##ax1.scatter(moon_G1[0, 0], moon_G1[0, 1], moon_G1[0, 2], 'b', label='time ii')
###ax1.plot(moon_G2[:,0], moon_G2[:,1], moon_G2[:,2], 'r-.', label='time mjd')
##ax1.set_xlabel('X [AU]')
##ax1.set_ylabel('Y [AU]')
##ax1.set_zlabel('Z [AU]')
##ax1.set_title('G frame')
##plt.legend()
##plt.show()
##breakpoint()
##
##
#ax1 = plt.figure().add_subplot(projection='3d')
#ax1.plot(moon_G1[:, 0], moon_G1[:, 1], moon_G1[:, 2], 'b', label='Start')
#ax1.plot(r_MB_LAAN[:, 0], r_MB_LAAN[:, 1], r_MB_LAAN[:, 2], 'r', label='LAAN')
#ax1.plot(r_MB_INC[:, 0], r_MB_INC[:, 1], r_MB_INC[:, 2], 'g', label='LAAN + INC')
#ax1.plot(r_MB_I2s[:, 0], r_MB_I2s[:, 1], r_MB_I2s[:, 2], 'k', label='LAAN + INC + AOP')
##ax1.plot(moon_G2[:,0], moon_G2[:,1], moon_G2[:,2], 'r-.', label='time mjd')
#ax1.set_xlabel('X [AU]')
#ax1.set_ylabel('Y [AU]')
#ax1.set_zlabel('Z [AU]')
#ax1.set_title('I frame')
#plt.legend()
#
#
#ax1 = plt.figure().add_subplot(projection='3d')
#ax1.plot(moon_G1[:, 0], moon_G1[:, 1], moon_G1[:, 2], 'b', label='Start')
#ax1.plot(r_MB_LAAN[:, 0], r_MB_LAAN[:, 1], r_MB_LAAN[:, 2], 'r', label='LAAN')
#ax1.plot(np.array([0, n_LAAN[0]]), np.array([0, n_LAAN[1]]), np.array([0, n_LAAN[2]]),'y')
#ax1.set_xlabel('X [AU]')
#ax1.set_ylabel('Y [AU]')
#ax1.set_zlabel('Z [AU]')
#ax1.set_title('I frame')
#plt.legend()
#
#ax1 = plt.figure().add_subplot(projection='3d')
#ax1.plot(r_MB_LAAN[:, 0], r_MB_LAAN[:, 1], r_MB_LAAN[:, 2], 'r', label='LAAN')
#ax1.plot(r_MB_INC[:, 0], r_MB_INC[:, 1], r_MB_INC[:, 2], 'g', label='LAAN + INC')
#ax1.plot(np.array([0, n_INC[0]]), np.array([0, n_INC[1]]), np.array([0, n_INC[2]]),'y')
#ax1.set_xlabel('X [AU]')
#ax1.set_ylabel('Y [AU]')
#ax1.set_zlabel('Z [AU]')
#ax1.set_title('I frame')
#plt.legend()
#
#ax1 = plt.figure().add_subplot(projection='3d')
#ax1.plot(r_MB_INC[:, 0], r_MB_INC[:, 1], r_MB_INC[:, 2], 'g', label='LAAN + INC')
#ax1.plot(r_MB_I2s[:, 0], r_MB_I2s[:, 1], r_MB_I2s[:, 2], 'k', label='LAAN + INC + AOP')
#ax1.plot(np.array([0, n_AOP[0]]), np.array([0, n_AOP[1]]), np.array([0, n_AOP[2]]),'y')
#ax1.set_xlabel('X [AU]')
#ax1.set_ylabel('Y [AU]')
#ax1.set_zlabel('Z [AU]')
#ax1.set_title('I frame')
#plt.legend()
#
#ax1 = plt.figure().add_subplot(projection='3d')
#ax1.plot(r_MB_I2s[:, 0], r_MB_I2s[:, 1], r_MB_I2s[:, 2], 'k', label='LAAN + INC + AOP')
##ax1.plot(np.array([0, n_AOP[0]]), np.array([0, n_AOP[1]]), np.array([0, n_AOP[2]]),'y')
#ax1.set_xlabel('X [AU]')
#ax1.set_ylabel('Y [AU]')
#ax1.set_zlabel('Z [AU]')
#ax1.set_title('I frame')
#
#plt.show()
#breakpoint()

# Initial condition in non dimensional units in rotating frame R [pos, vel]
IC = [1.0110350593505575, 0, -0.17315000084377485, 0, -0.0780142664611386, 0, 0.6816048399338378]
X = [IC[0], IC[2], IC[4], IC[6]]

max_iter = 1000
error = 10
ctr = 0
eps = 4E-6
while error > eps and ctr < max_iter:
    Fx = orbitEOMProp.calcFx_R(X, mu_star)

    error = np.linalg.norm(Fx)
    print(error)
    dFx = orbitEOMProp.calcdFx_CRTBP(X,mu_star,m1,m2)

    X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
    
    ctr = ctr + 1
    
IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])
    
# Convert the velocity to I frame from R frame
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar_CRTBP = np.array([IC[0], IC[2], vI[1], IC[-1]])   # IC[-1] for 1 period

# Propagate the dynamics in the CRTBP model
statesCRTBP, timesCRTBP = orbitEOMProp.statePropCRTBP(freeVar_CRTBP, mu_star)
posCRTBP = statesCRTBP[:, 0:3]
velCRTBP = statesCRTBP[:, 3:6]

# Preallocate space
r_PO_CRTBP = np.zeros([len(timesCRTBP), 3])
r_PEM_CRTBP = np.zeros([len(timesCRTBP), 3])
r_EarthEM_CRTBP = np.zeros([len(timesCRTBP), 3])
r_MoonEM_CRTBP = np.zeros([len(timesCRTBP), 3])
r_CRTBP_rot = np.zeros([len(timesCRTBP), 3])
r_CRTBP_I = np.zeros([len(timesCRTBP), 3])
r_CRTBP_G = np.zeros([len(timesCRTBP), 3])
r_CRTBP_I2 = np.zeros([len(timesCRTBP), 3])
r_diff = np.zeros([len(timesCRTBP), 3])

r_PEM_CRTBP_R = np.zeros([len(timesCRTBP), 3])
r_MoonEM_CRTBP_R = np.zeros([len(timesCRTBP), 3])

# sim time in mjd
times_dim = unitConversion.convertTime_to_dim(timesCRTBP)
timesCRTBP_mjd = times_dim + t_mjd      # works in Grace's code
#timesCRTBP_mjd = Time(timesCRTBP + t_mjd.value, format='mjd', scale='utc')     # works in Anna's code

# DCM for G frame and I frame
#C_B2G = frameConversion.body2geo(t_mjd, t_mjd, mu_star)
#C_G2B = C_B2G.T

# Obtain Moon and Earth positions for CRTBP
for kk in np.arange(len(timesCRTBP)):
    time = timesCRTBP_mjd[kk]

#    # Positions of the Moon and EM barycenter relative SS barycenter in H frame
#    r_MoonO = get_body_barycentric_posvel('Moon', tim)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)
    r_EMO = EMO[0].get_xyz().to('AU').value
#
#    # Convert from H frame to GCRS frame
    r_EMG = (frameConversion.icrs2gcrs(r_EMO * u.AU, time)).to('AU')
#    r_MoonG = frameConversion.icrs2gcrs(r_MoonO * u.AU, time)
#
#    # Change the origin to the EM barycenter, G frame
#    r_EarthEM = -r_EMG
#    r_MoonEM = r_MoonG - r_EMG
#
#    # Convert from G frame to I frame
#    r_EarthEM_CRTBP[ii, :] = C_G2B @ r_EarthEM.to('AU')
#    r_MoonEM_CRTBP[ii, :] = C_G2B @ r_MoonEM.to('AU')

#    C_B2G = frameConversion.body2geo(time, t_mjd, mu_star)
#    C_G2B = C_B2G.T

    # Convert to AU
#    r_PEM_CRTBP[ii, :] = (unitConversion.convertPos_to_dim(posCRTBP[ii, :])).to('AU')
    r_dim = (unitConversion.convertPos_to_dim(posCRTBP[kk, :])).to('AU').value
    r_EM = C_B2G @ r_dim
    r_GCRS = r_EM +  r_EMG.value
    
    r_PO_H, _ = frameConversion.convertIC_I2H(posCRTBP[kk,:], velCRTBP[kk,:], time, t_mjd, mu_star, C_B2G)
    r_PO_CRTBP[kk, :] = r_PO_H
    
    C_I2R = frameConversion.body2rot(time,t_mjd)
    r_CRTBP_rot[kk,:] = C_I2R @ r_dim
    r_CRTBP_I[kk,:] = r_dim
#    r_CRTBP_G[kk,:] = r_GCRS
    r_CRTBP_G[kk,:] = r_EM
    r_CRTBP_I2[kk,:] = C_G2B @ r_EM
    
    r_diff[kk,:] = C_G2B @ r_EM - r_dim

# Convert position from I frame to H frame [AU]
pos_H, vel_H, Tp_dim = frameConversion.convertIC_I2H(posCRTBP[0], velCRTBP[0], t_mjd, t_mjd, mu_star, C_B2G, timesCRTBP[-1])

# Define the initial state array
state0 = np.append(np.append(pos_H.value, vel_H.value), 1*Tp_dim.value)   # Change to Tp_dim.value for one orbit

# Propagate the dynamics in the full force model (H frame) [AU]
statesFF, timesFF = orbitEOMProp.statePropFF(state0, t_mjd)
posFF = statesFF[:, 0:3]
velFF = statesFF[:, 3:6]

# Preallocate space
r_PEM_i = np.zeros([len(timesFF), 3])
r_SunEM_i = np.zeros([len(timesFF), 3])
r_EarthEM_i = np.zeros([len(timesFF), 3])
r_MoonEM_i = np.zeros([len(timesFF), 3])
r_PEM_g = np.zeros([len(timesFF), 3])
r_SunEM_g = np.zeros([len(timesFF), 3])
r_EarthEM_g = np.zeros([len(timesFF), 3])
r_MoonEM_g = np.zeros([len(timesFF), 3])
r_EMO_h = np.zeros([len(timesFF), 3])
r_SunO_h = np.zeros([len(timesFF), 3])
r_EarthO_h = np.zeros([len(timesFF), 3])
r_MoonO_h = np.zeros([len(timesFF), 3])

# sim time in mjd
timesFF_mjd = timesFF + t_mjd

# Obtain Moon, Earth, and Sun positions for FF
for ii in np.arange(len(timesFF_mjd)):
    time = timesFF_mjd[ii]

    # Positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU').value
    r_EarthO = get_body_barycentric_posvel('Earth', time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)
    r_EMO = EMO[0].get_xyz().to('AU').value
    
    r_SunO_h[ii, :] = r_SunO
    r_EarthO_h[ii, :] = r_EarthO
    r_MoonO_h[ii, :] = r_MoonO
    r_EMO_h[ii, :] = r_EMO

    # Convert from H frame to GCRS frame
    r_PG = frameConversion.icrs2gcrs(posFF[ii]*u.AU, time)
    r_EMG = frameConversion.icrs2gcrs(r_EMO*u.AU, time)
    r_SunG = frameConversion.icrs2gcrs(r_SunO*u.AU, time)
    r_MoonG = frameConversion.icrs2gcrs(r_MoonO*u.AU, time)

    # Change the origin to the EM barycenter, G frame
    r_PEM = r_PG - r_EMG
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG
    
#    r_PEM_g[ii, :] = r_PEM.to('AU')
    r_SunEM_g[ii, :] = r_SunEM.to('AU')
    r_EarthEM_g[ii, :] = r_EarthEM.to('AU')
    r_MoonEM_g[ii, :] = r_MoonEM.to('AU')

#    C_B2G = frameConversion.body2geo(time, t_mjd, mu_star)
#    C_G2B = C_B2G.T

    # Convert from G frame to I frame
    r_PEM_i[ii, :] = C_G2B@r_PEM.to('AU')
    r_SunEM_i[ii, :] = C_G2B@r_SunEM.to('AU')
    r_EarthEM_i[ii, :] = C_G2B@r_EarthEM.to('AU')
    r_MoonEM_i[ii, :] = C_G2B@r_MoonEM.to('AU')
    

ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(posFF[:, 0], posFF[:, 1], posFF[:, 2], 'b', label='Full Force')
ax1.plot(r_PO_CRTBP[:,0], r_PO_CRTBP[:,1], r_PO_CRTBP[:,2], 'r-.', label='CRTBP')
ax1.set_title('FF vs CRTBP in H frame (ICRS)')
ax1.set_xlabel('X [AU]')
ax1.set_ylabel('Y [AU]')
ax1.set_zlabel('Z [AU]')
plt.legend()

ax2 = plt.figure().add_subplot(projection='3d')
ax2.plot(r_PEM_i[:, 0], r_PEM_i[:, 1], r_PEM_i[:, 2], 'b', label='Full Force')
ax2.plot(r_CRTBP_I[:,0], r_CRTBP_I[:,1], r_CRTBP_I[:,2], 'r-.', label='CRTBP')
#ax2.plot(r_CRTBP_I2[:,0], r_CRTBP_I2[:,1], r_CRTBP_I2[:,2], 'b', label='CRTBP')
ax2.plot(r_EarthEM_i[:,0], r_EarthEM_i[:,1], r_EarthEM_i[:,2], 'g', label='Earth')
ax2.plot(r_MoonEM_i[:,0], r_MoonEM_i[:,1], r_MoonEM_i[:,2], 'k', label='Moon')
ax2.plot(r_diff[:,0], r_diff[:,1], r_diff[:,2], label='difference')
ax2.set_title('FF vs CRTBP in I frame (Inertial EM)')
ax2.set_xlabel('X [AU]')
ax2.set_ylabel('Y [AU]')
ax2.set_zlabel('Z [AU]')
plt.legend()

ax3 = plt.figure().add_subplot(projection='3d')
ax3.plot(r_PEM_i[:, 0], r_PEM_i[:, 1], r_PEM_i[:, 2], 'b', label='Full Force')
ax3.plot(r_CRTBP_I[:,0], r_CRTBP_I[:,1], r_CRTBP_I[:,2], 'r-.', label='CRTBP')
ax3.plot(r_EarthEM_i[:,0], r_EarthEM_i[:,1], r_EarthEM_i[:,2], 'g', label='Earth')
ax3.plot(r_MoonEM_i[:,0], r_MoonEM_i[:,1], r_MoonEM_i[:,2], 'k', label='Moon')
ax3.plot(r_SunEM_i[:,0], r_SunEM_i[:,1], r_SunEM_i[:,2], 'y', label='Sun')
ax3.set_title('FF vs CRTBP in I frame (Inertial EM)')
ax3.set_xlabel('X [AU]')
ax3.set_ylabel('Y [AU]')
ax3.set_zlabel('Z [AU]')
plt.legend()

ax4 = plt.figure().add_subplot(projection='3d')
ax4.plot(r_CRTBP_G[:,0], r_CRTBP_G[:,1], r_CRTBP_G[:,2], 'r-.', label='CRTBP')
ax4.plot(r_EarthEM_g[:,0], r_EarthEM_g[:,1], r_EarthEM_g[:,2], 'g', label='Earth')
ax4.plot(r_MoonEM_g[:,0], r_MoonEM_g[:,1], r_MoonEM_g[:,2], 'k', label='Moon')
ax4.set_title('FF vs CRTBP in G frame (GCRS centered at EM)')
ax4.set_xlabel('X [AU]')
ax4.set_ylabel('Y [AU]')
ax4.set_zlabel('Z [AU]')
plt.legend()

ax5 = plt.figure().add_subplot(projection='3d')
ax5.plot(r_PEM_g[:, 0], r_PEM_g[:, 1], r_PEM_g[:, 2], 'b', label='Full Force')
ax5.plot(r_CRTBP_G[:,0], r_CRTBP_G[:,1], r_CRTBP_G[:,2], 'r-.', label='CRTBP')
ax5.plot(r_EarthEM_g[:,0], r_EarthEM_g[:,1], r_EarthEM_g[:,2], 'g', label='Earth')
ax5.plot(r_MoonEM_g[:,0], r_MoonEM_g[:,1], r_MoonEM_g[:,2], 'k', label='Moon')
ax5.plot(r_SunEM_g[:,0], r_SunEM_g[:,1], r_SunEM_g[:,2], 'y', label='Sun')
ax5.set_title('FF vs CRTBP in G frame (GCRS centered at EM)')
ax5.set_xlabel('X [AU]')
ax5.set_ylabel('Y [AU]')
ax5.set_zlabel('Z [AU]')
plt.legend()

ax6 = plt.figure().add_subplot(projection='3d')
ax6.plot(r_SunEM_g[:,0], r_SunEM_g[:,1], r_SunEM_g[:,2], 'b', label='Sun G')
ax6.plot(r_SunEM_i[:,0], r_SunEM_i[:,1], r_SunEM_i[:,2], 'g', label='Sun I')
ax6.set_xlabel('X [AU]')
ax6.set_ylabel('Y [AU]')
ax6.set_zlabel('Z [AU]')
plt.legend()

ax7 = plt.figure().add_subplot(projection='3d')
ax7.plot(r_EarthEM_g[:,0], r_EarthEM_g[:,1], r_EarthEM_g[:,2], 'b', label='Earth G')
ax7.plot(r_EarthEM_i[:,0], r_EarthEM_i[:,1], r_EarthEM_i[:,2], 'g', label='Earth I')
ax7.set_xlabel('X [AU]')
ax7.set_ylabel('Y [AU]')
ax7.set_zlabel('Z [AU]')
plt.legend()

#ax6 = plt.figure().add_subplot(projection='3d')
#ax6.plot(r_EarthO_h[:,0], r_EarthO_h[:,1], r_EarthO_h[:,2], 'g', label='Earth')
#ax6.plot(r_EMO_h[:,0], r_EMO_h[:,1], r_EMO_h[:,2], 'r', label='Earth-Moon Barycenter')
#ax6.plot(r_MoonO_h[:,0], r_MoonO_h[:,1], r_MoonO_h[:,2], 'k', label='Moon')
#ax6.set_title('H frame (ICRS)')
#ax6.set_xlabel('X [AU]')
#ax6.set_ylabel('Y [AU]')
#ax6.set_zlabel('Z [AU]')
#plt.legend()

#ax7 = plt.figure().add_subplot(projection='3d')
#ax7.plot(r_EarthO_h[:,0], r_EarthO_h[:,1], r_EarthO_h[:,2], 'g', label='Earth')
#ax7.plot(r_EMO_h[:,0], r_EMO_h[:,1], r_EMO_h[:,2], 'r', label='Earth-Moon Barycenter')
#ax7.plot(r_MoonO_h[:,0], r_MoonO_h[:,1], r_MoonO_h[:,2], 'k', label='Moon')
#ax7.plot(r_SunO_h[:,0], r_SunO_h[:,1], r_SunO_h[:,2], 'y', label='Sun')
#ax7.set_title('H frame (ICRS)')
#ax7.set_xlabel('X [AU]')
#ax7.set_ylabel('Y [AU]')
#ax7.set_zlabel('Z [AU]')
#plt.legend()

plt.show()
breakpoint()

