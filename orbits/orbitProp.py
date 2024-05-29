import numpy as np
import os.path
import pickle
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
#import rebound
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
#import orbitGenCR3BP as orgen
sys.path.insert(1, '/Users/gracegenszler/Documents/Research/starlift/tools')
import unitConversion
import frameConversion


import pdb

# TU = 27.321582 d
# DU = 384400 km
# m_moon = 7.349x10**22 kg

def CRTBP_EOM(t,w,mu_star):
    """Equations of motion for the CRTBP in the rotating frame

    Args:
        w (float):
            State in non dimensional units [position, velocity]
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            Time derivative of the state [velocity, acceleration]

    """
    [x,y,z,vx,vy,vz] = w
    
    m1 = 1 - mu_star
    m2 = mu_star

    r1 = mu_star
    r2 = 1 - mu_star
    r_1O_H = r1*np.array([-1,0,0])
    r_2O_H = r2*np.array([1,0,0])
    
    C_I2R = frameConversion.rot(t,3)
    C_R2I = C_I2R.T
    
    r_1O_I = C_R2I@r_1O_H
    r_2O_I = C_R2I@r_2O_H

    r_PO_I = np.array([x,y,z])
    v_PO_I = np.array([vx,vy,vz])

    r_P1_I = r_PO_I - r_1O_I
    r_P2_I = r_PO_I - r_2O_I
    
    r1_mag = np.linalg.norm(r_P1_I)
    r2_mag = np.linalg.norm(r_P2_I)

    F_g1 = -m1/r1_mag**3*r_P1_I
    F_g2 = -m2/r2_mag**3*r_P2_I
    F_g = F_g1 + F_g2

    e3_hat = np.array([0,0,1])

    a_PO_H = F_g
    
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx,vy,vz,ax,ay,az]
    return dw
    
def FF_EOM(tt,w,t_mjd,mu_star):
    """Equations of motion for the full force model in the inertial frame

    Args:
        w (float):
            State in non dimensional units [position, velocity]
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            Time derivative of the state [velocity, acceleration]

    """
    
    [x,y,z,vx,vy,vz] = w
    
    gmSun = const.GM_sun.to('AU3/d2').value        # in AU^3/d^2
    gmEarth = const.GM_earth.to('AU3/d2').value
    gmMoon = 0.109318945437743700E-10              # from de432s header
    
    r_PEM = np.array([x,y,z])
    v_PEM = np.array([vx,vy,vz])

    time = tt + t_mjd
#    print(tt)

    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().to('AU')
    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().to('AU')
    r_EarthO = get_body_barycentric_posvel('Earth',time)[0].get_xyz().to('AU')

    r_PSun = r_PEM - r_SunO.value
    r_PEarth = r_PEM - r_EarthO.value
    r_PMoon = r_PEM - r_MoonO.value
    rSun_mag = np.linalg.norm(r_PSun)
    rEarth_mag = np.linalg.norm(r_PEarth)
    rMoon_mag = np.linalg.norm(r_PMoon)

    F_gSun = -gmSun/rSun_mag**3*r_PSun
    F_gEarth = -gmEarth/rEarth_mag**3*r_PEarth
    F_gMoon = -gmMoon/rMoon_mag**3*r_PMoon

    F_g = F_gSun + F_gEarth + F_gMoon
    F_g2 = F_gEarth + F_gMoon
    
    a_PO_H = F_g
    
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx,vy,vz,ax,ay,az]
    
    if tt < .01:
#        tmp1 = dw[3:6]*u.AU/u.day
#        tmp1 = tmp1.to('km/s')
#        tmp2 = np.linalg.norm(tmp1)
#        
#        print(np.linalg.norm(F_gSun))
#        print(np.linalg.norm(F_gEarth))
#        print(np.linalg.norm(F_gMoon))
        print(F_g)
        print(F_g2)
        breakpoint()
    elif tt > 75 and tt < 80:
#        tmp1 = dw[3:6]*u.AU/u.day
#        tmp1 = tmp1.to('km/s')
#        tmp2 = np.linalg.norm(tmp1)
#        print(np.linalg.norm(F_gSun))
#        print(np.linalg.norm(F_gEarth))
#        print(np.linalg.norm(F_gMoon))
        print(F_g)
        print(F_g2)
        breakpoint()
    elif tt > 230:
        print(F_g)
        print(F_g2)
        breakpoint()
    return dw


def statePropCRTBP(freeVar,mu_star):
    """Propagates the dynamics using the free variables

    Args:
        state (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    x0 = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    T = freeVar[-1]

#    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,))
    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,),rtol=1E-12,atol=1E-12,)
    states = sol_int.y.T
    
    return states
    
def statePropFF(freeVar,t_mjd,mu_star):
    """Propagates the dynamics using the free variables

    Args:
        state (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    T = freeVar[-1]

    sol_int = solve_ivp(FF_EOM, [0, T], freeVar[0:6], args=(t_mjd,mu_star), method='LSODA',t_eval=np.arange(0,T,.1))

    states = sol_int.y.T
    times = sol_int.t
    
    return states, times
    

#Barycentric (ICRS)
t_mjd = Time(60380+0,format='mjd',scale='utc')
coord.solar_system.solar_system_ephemeris.set('de432s')

days = 232
# 0         232
# 90        any
# 180       224
# 270       207
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

C_B2G = frameConversion.body2geo(t_mjd,t_mjd,mu_star)
C_G2B = C_B2G.T

IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

IV_CRTBP = np.array([IC[0], IC[2], vI[1], days])     #2*IC[6]

#statesCRTBP = statePropCRTBP(IV_CRTBP,mu_star)
#posCRTBP = unitConversion.convertPos_to_dim(statesCRTBP[:,0:3]).to('AU').value
#print('CRTBP done')
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)
v_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',t_mjd)[1].get_xyz().to('AU/day')

x_dim = unitConversion.convertPos_to_dim(IC[0]).to('AU').value
z_dim = unitConversion.convertPos_to_dim(IC[2]).to('AU').value
v_dim = unitConversion.convertVel_to_dim(vI).to('AU/day')
Tp_dim = unitConversion.convertTime_to_dim(2*IC[6]).to('day').value

pos_dim = np.array([x_dim, 0, z_dim])*u.AU

C_B2G = frameConversion.body2geo(t_mjd,t_mjd,mu_star)

pos_GCRS = C_B2G@pos_dim
pos_ICRS = (frameConversion.gcrs2icrs(pos_GCRS,t_mjd)).to('AU').value

vel_ICRS = (v_EMO + v_dim).value

state0 = np.array([pos_ICRS[0], pos_ICRS[1], pos_ICRS[2], vel_ICRS[0], vel_ICRS[1], vel_ICRS[2], days])   # Tp_dim

statesFF, timesFF = statePropFF(state0,t_mjd,mu_star)
posFF = statesFF[:,0:3]
velFF = statesFF[:,3:6]
print('FF done')

r_PEM_r = np.zeros([len(timesFF),3])
r_SunEM_r = np.zeros([len(timesFF),3])
r_EarthEM_r = np.zeros([len(timesFF),3])
r_MoonEM_r = np.zeros([len(timesFF),3])
for ii in np.arange(len(timesFF)):
    time = timesFF[ii] + t_mjd
    
    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',time)
    r_EMO = EMO[0].get_xyz().to('AU').value
    
    r_PG = frameConversion.icrs2gcrs(posFF[ii]*u.AU,time)
    r_EMG = frameConversion.icrs2gcrs(r_EMO*u.AU,time)
    r_SunG = frameConversion.icrs2gcrs(r_SunO*u.AU,time)
    r_MoonG = frameConversion.icrs2gcrs(r_MoonO*u.AU,time)
    
    r_PEM = r_PG - r_EMG
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG
    
    r_PEM_r[ii,:] = C_G2B@r_PEM.to('AU')
    r_SunEM_r[ii,:] = C_G2B@r_SunEM.to('AU')
    r_EarthEM_r[ii,:] = C_G2B@r_EarthEM.to('AU')
    r_MoonEM_r[ii,:] = C_G2B@r_MoonEM.to('AU')

tmp1 = r_PEM_r - r_EarthEM_r
norms = np.linalg.norm(tmp1,axis=1)*u.AU
minNorm = min(norms).to('km')
if minNorm.value < 6378:
    print(minNorm)

velFF = velFF*u.AU/u.day
velFF = velFF.to('km/s')
velNorm = np.linalg.norm(velFF,axis=1)
velMax = max(velNorm)

indMax = np.argwhere(velNorm == velMax)[0][0]


plt.figure()
plt.plot(timesFF,velNorm)
plt.xlabel('Time [days]')
plt.ylabel('Velocity Magnitude [km/s]')

ax = plt.figure().add_subplot(projection='3d')
#ax.plot(posCRTBP[:,0],posCRTBP[:,1],posCRTBP[:,2],'r',label='CRTBP')
#ax.plot(posFF[:,0],posFF[:,1],posFF[:,2],'b',label='Full Force')
#ax.scatter(r_PEM_r[0,0],r_PEM_r[0,1],r_PEM_r[0,2],marker='*',label='FF Start')
#ax.scatter(r_PEM_r[-1,0],r_PEM_r[-1,1],r_PEM_r[-1,2],label='FF End')
#ax.scatter(r_PEM_r[-10:,0],r_PEM_r[-10:,1],r_PEM_r[-10:,2],marker='*',label='FF last 10')
#ax.scatter(r_PEM_r[0:10,0],r_PEM_r[0:10,1],r_PEM_r[0:10,2],label='FF first 10')
ax.scatter(r_PEM_r[indMax,0],r_PEM_r[indMax,1],r_PEM_r[indMax,2],label='max Velocity')
ax.plot(r_PEM_r[:,0],r_PEM_r[:,1],r_PEM_r[:,2],'b',label='Full Force')
ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2],'g',label='Earth')
ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2],'r',label='Moon')
#ax.plot(r_SunEM_r[:,0],r_SunEM_r[:,1],r_SunEM_r[:,2])
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
plt.legend()

#ax = plt.figure().add_subplot(projection='3d')
#ax.plot(r_SunEM_r[:,0],r_SunEM_r[:,1],r_SunEM_r[:,2])
#ax.scatter(r_SunEM_r[0,0],r_SunEM_r[0,1],r_SunEM_r[0,2])
#ax.set_xlabel('X [AU]')
#ax.set_ylabel('Y [AU]')
#ax.set_zlabel('Z [AU]')
#
#ax = plt.figure().add_subplot(projection='3d')
#ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2])
#ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2])


plt.show()
breakpoint()
