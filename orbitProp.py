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
    
    mSun = const.M_sun.value        # in kg
    mEarth = const.M_earth.value    # in kg
    mMoon = 7.349*10**22            # in kg, from JPL horizons
    
    G =  const.G.to('AU3/(kg d2)').value
    
    r_PEM = np.array([x,y,z])
    v_PEM = np.array([vx,vy,vz])

    time = tt + t_mjd

    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',time)
    r_EMO = EMO[0].get_xyz().to('AU').value
    
    r_SunG = frameConversion.icrs2gcrs(r_SunO*u.AU,time)
    r_MoonG = frameConversion.icrs2gcrs(r_MoonO*u.AU,time)
    r_EMG = frameConversion.icrs2gcrs(r_EMO*u.AU,time)
    
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG
    
    C_B2G = frameConversion.body2geo(time,t_mjd,mu_star)
    C_G2B = C_B2G.T
    
    r_SunEM_r = C_G2B@r_SunEM.to('AU')
    r_EarthEM_r = C_G2B@r_EarthEM.to('AU')
    r_MoonEM_r = C_G2B@r_MoonEM.to('AU')
    
    

    r_PSun = r_PEM - r_SunEM_r.value
    r_PEarth = r_PEM - r_EarthEM_r.value
    r_PMoon = r_PEM - r_MoonEM_r.value
    rSun_mag = np.linalg.norm(r_PSun)
    rEarth_mag = np.linalg.norm(r_PEarth)
    rMoon_mag = np.linalg.norm(r_PMoon)

    F_gSun = -mSun/rSun_mag**3*r_PSun
    F_gEarth = -mEarth/rEarth_mag**3*r_PEarth
    F_gMoon = -mMoon/rMoon_mag**3*r_PMoon

    F_g = F_gSun + F_gEarth + F_gMoon
    
    a_PO_H = G*F_g
    
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx,vy,vz,ax,ay,az]

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

    sol_int = solve_ivp(FF_EOM, [0, T], freeVar[0:6], args=(t_mjd,mu_star), method='LSODA')

    states = sol_int.y.T
    times = sol_int.t
    
    return states, times
    

#Barycentric (ICRS)
t_mjd = Time(60380+180,format='mjd',scale='utc')
coord.solar_system.solar_system_ephemeris.set('de432s')

mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

IV_CRTBP = np.array([IC[0], IC[2], vI[1], 2*IC[6]])

statesCRTBP = statePropCRTBP(IV_CRTBP,mu_star)
posCRTBP = unitConversion.convertPos_to_dim(statesCRTBP[:,0:3]).to('AU').value

get_body_barycentric_posvel('Earth-Moon-Barycenter',t_mjd)
x_dim = unitConversion.convertPos_to_dim(IC[0]).to('AU').value
z_dim = unitConversion.convertPos_to_dim(IC[2]).to('AU').value
v_dim = unitConversion.convertVel_to_dim(vI).to('AU/day').value
Tp_dim = unitConversion.convertTime_to_dim(2*IC[6]).to('day').value
state0 = [x_dim, 0, z_dim, v_dim[0], v_dim[1], v_dim[2], 1*Tp_dim]

statesFF, timesFF = statePropFF(state0,t_mjd,mu_star)
posFF = statesFF[:,0:3]

ax = plt.figure().add_subplot(projection='3d')
ax.plot(posCRTBP[:,0],posCRTBP[:,1],posCRTBP[:,2],'r',label='CRTBP')
ax.plot(posFF[:,0],posFF[:,1],posFF[:,2],'b',label='Full Force')
plt.legend()

plt.show()
breakpoint()
