import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
import jplephem
#import rebound
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
sys.path.insert(1, '/Users/gracegenszler/Documents/Research/starlift')
import unitConversion
import frameConversion

import pdb

# TU = 27.321582 d
# DU = 384400 km
# m_moon = 7.349x10**22 kg
# m_earth = 5.97219x10**24 kg

#"Earth": 399,
#"Sun": 10,
#"Moon": 301,
#"EM Barycenter": 3
#"Solar System Barycenter": 0

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

    r_PO_H = np.array([x,y,z])
    v_PO_H = np.array([vx,vy,vz])

    r_P1_H = r_PO_H - r_1O_H
    r_P2_H = r_PO_H - r_2O_H
    r1_mag = np.linalg.norm(r_P1_H)
    r2_mag = np.linalg.norm(r_P2_H)

    F_g1 = -m1/r1_mag**3*r_P1_H
    F_g2 = -m2/r2_mag**3*r_P2_H
    F_g = F_g1 + F_g2

    e3_hat = np.array([0,0,1])

    a_PO_H = F_g - 2*np.cross(e3_hat,v_PO_H) - np.cross(e3_hat,np.cross(e3_hat,r_PO_H))
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx,vy,vz,ax,ay,az]
    return dw
    
def FF_EOM(tt,w,t_mjd):
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
    
    # write code so everything in and out is ss barycenter referenced
    [x,y,z,vx,vy,vz] = w
    
    mSun = 1.9885*10**30    # in kg
    mMoon = 7.349*10**22    # in kg
    mEarth = 5.97219*10**24 # in kg

    time = tt + t_mjd

    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().value
    r_EarthO = get_body_barycentric_posvel('Earth',time)[0].get_xyz().value
    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',time)
    r_EMO = EMO[0].get_xyz().value
    v_EMO = EMO[1].get_xyz().value

    r_PO = np.array([x,y,z]) + r_EMO
    v_PO = np.array([vx,vy,vz]) + v_EMO
    
    e3_hat = np.array([0,0,1])

    r_PSun = r_PO - r_SunO
    r_PEarth = r_PO - r_EarthO
    r_PMoon = r_PO - r_MoonO
    rSun_mag = np.linalg.norm(r_PSun)
    rEarth_mag = np.linalg.norm(r_PEarth)
    rMoon_mag = np.linalg.norm(r_PMoon)

    F_gSun = -mSun/rSun_mag**3*r_PSun
    F_gEarth = -mEarth/rEarth_mag**3*r_PEarth
    F_gMoon = -mMoon/rMoon_mag**3*r_PMoon
    F_g = F_gSun + F_gEarth + F_gMoon
    
    a_PO_H = F_g
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
    T = freeVar[3]

    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,),rtol=1E-12,atol=1E-12,t_eval=np.linspace(0,T,300))
    states = sol_int.y.T
    
    return states
    
def statePropFF(freeVar,t_mjd):
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
    x0 = [freeVar[0].value, 0, freeVar[1].value, 0, freeVar[2].value, 0]
    T = freeVar[3].value

    sol_int = solve_ivp(FF_EOM, [0, T], x0, args=(t_mjd,),rtol=1E-12,atol=1E-12,t_eval=np.linspace(0,T,300))
    states = sol_int.y.T
    times = sol_int.x
    
    return states, times

#Barycentric (ICRS)
t_mjd = Time(60380,format='mjd',scale='utc')
coord.solar_system.solar_system_ephemeris.set('de432s')

mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]
X = [IC[0], IC[2], IC[4], IC[6]]

#eps = 3E-6
#N = 5
#solutions = np.zeros([N,4])
#zT = np.array([0, 0, 0, 1])
#z = np.array([0, 0, 0, 1])
#step = 1E-2
#
#error = 10
#ctr = 0
#while error > eps:
#    Fx = calcFx(X)
#
#    error_new = np.linalg.norm(Fx)
#
#    if error_new > error:
#        print('Solution Did Not Converge')
#        print(error_new)
#        break
#    else:
#        error = error_new
#        ctr = ctr + 1
#
#    dFx = calcdFx(X,mu_star,m1,m2)
#
#    X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
#
#IV = np.array([X[0], X[1], X[2], 2*X[3]])
#solutions[ii] = IV
#statesCRTBP = statePropCRTBP(IV,mu_star)

get_body_barycentric_posvel('Earth-Moon-Barycenter',time)
x_dim = unitConversion.convertPos_to_dim(X[0]).to('km')
z_dim = unitConversion.convertPos_to_dim(X[1]).to('km')
dy_dim = unitConversion.convertVel_to_dim(X[2]).to('km/day')
Tp_dim = unitConversion.convertTime_to_dim(X[3]).to('day')
state0 = [x_dim, z_dim, dy_dim, Tp_dim]

statesFF, timesFF = statePropFF(state0,t_mjd)
posFF = statesFF[:,0:3]




ax = plt.figure().add_subplot(projection='3d')
#ax.plot(statesCRTBP[:,0],statesCRTBP[:,1],statesCRTBP[:,2])
ax.plot(statesFF[:,0],statesFF[:,1],statesFF[:,2])

plt.show()
breakpoint()
