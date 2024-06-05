import numpy as np
import os.path
import pickle
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion

import pdb

def CRTBP_EOM(t,w,mu_star):
    """Equations of motion for the CRTBP in the inertial frame

    Args:
        w (~numpy.ndarray(float)):
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
    r_1O_R = r1*np.array([-1,0,0])
    r_2O_R = r2*np.array([1,0,0])
    
    C_I2R = frameConversion.rot(t,3)
    C_R2I = C_I2R.T
    
    r_1O_I = C_R2I@r_1O_R
    r_2O_I = C_R2I@r_2O_R

    r_PO_I = np.array([x,y,z])
    v_PO_I = np.array([vx,vy,vz])

    r_P1_I = r_PO_I - r_1O_I
    r_P2_I = r_PO_I - r_2O_I
    
    r1_mag = np.linalg.norm(r_P1_I)
    r2_mag = np.linalg.norm(r_P2_I)

    F_g1 = -m1/r1_mag**3*r_P1_I
    F_g2 = -m2/r2_mag**3*r_P2_I
    F_g = F_g1 + F_g2

    a_PO_I = F_g
    
    ax = a_PO_I[0]
    ay = a_PO_I[1]
    az = a_PO_I[2]

    dw = [vx,vy,vz,ax,ay,az]
    return dw

def FF_EOM(tt,w,t_mjd):
    """Equations of motion for the full force model in the ICRS frame

    Args:
        w (~numpy.ndarray(float)):
            State in non dimensional units [position, velocity]
        t_mjd (astropy Time array):
            Mission start time in MJD

    Returns:
        ~numpy.ndarray(float):
            Time derivative of the state [velocity, acceleration]

    """
    
    [x,y,z,vx,vy,vz] = w
    
    gmSun = const.GM_sun.to('AU3/d2').value        # in AU^3/d^2
    gmEarth = const.GM_earth.to('AU3/d2').value
    gmMoon = 0.109318945437743700E-10              # from de432s header
    
    r_PO = np.array([x,y,z])
    v_PO = np.array([vx,vy,vz])

    time = tt + t_mjd

    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().to('AU')
    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().to('AU')
    r_EarthO = get_body_barycentric_posvel('Earth',time)[0].get_xyz().to('AU')

    r_PSun = r_PO - r_SunO.value
    r_PEarth = r_PO - r_EarthO.value
    r_PMoon = r_PO - r_MoonO.value
    rSun_mag = np.linalg.norm(r_PSun)
    rEarth_mag = np.linalg.norm(r_PEarth)
    rMoon_mag = np.linalg.norm(r_PMoon)
    
    r_SunEarth = (r_SunO - r_EarthO).value
    r_MoonEarth = (r_MoonO - r_EarthO).value
    rSE_mag = np.linalg.norm(r_SunEarth)
    rME_mag = np.linalg.norm(r_MoonEarth)
        
    F_gSun_p = -gmSun*(r_SunEarth/rSE_mag**3 + r_PSun/rSun_mag**3)
    F_gEarth = -gmEarth/rEarth_mag**3*r_PEarth
    F_gMoon_p = -gmMoon*(r_MoonEarth/rME_mag**3 + r_PMoon/rMoon_mag**3)

    F_g = F_gSun_p + F_gEarth + F_gMoon_p
    
    a_PO_H = F_g
    
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx,vy,vz,ax,ay,az]
    
    return dw


def statePropCRTBP(freeVar,mu_star):
    """Propagates the dynamics using the free variables

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions (AU), y velocity (AU/d), and orbit period (d)
        mu_star (float):
            Non-dimensional mass parameter


    Returns:
        tuple:
        states ~numpy.ndarray(float):
            Positions and velocities in non dimensional units
        times ~numpy.ndarray(float):
            Times in non dimensional units

    """
    x0 = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    T = freeVar[-1]

    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,),rtol=1E-12,atol=1E-12,)
    states = sol_int.y.T
    times = sol_int.t
    
    return states, times
    
def statePropFF(state0,t_mjd):
    """Propagates the dynamics using the free variables

    Args:
        state(~numpy.ndarray(float)):
            Position in non dimensional units
        t_mjd (astropy Time array):
            Mission start time in MJD


    Returns:
        Returns:
            tuple:
            states ~numpy.ndarray(float):
                Positions and velocities in non dimensional units
            times ~numpy.ndarray(float):
                Times in non dimensional units

    """
    T = state0[-1]

    sol_int = solve_ivp(FF_EOM, [0, T], state0[0:6], args=(t_mjd,), method='LSODA')

    states = sol_int.y.T
    times = sol_int.t
    
    return states, times
    

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de432s')

# Parameters
t_mjd = Time(60380,format='mjd',scale='utc')
days = 30
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in non dimensional units in rotating frame R [pos, vel]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]

# Convert the velocity to inertial from I
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar_CRTBP = np.array([IC[0], IC[2], vI[1], days])

# propagate the dynamics in the CRTBP model
statesCRTBP, timesCRTBP = statePropCRTBP(freeVar_CRTBP,mu_star)
posCRTBP = statesCRTBP[:,0:3]
velCRTBP = statesCRTBP[:,3:6]

# convert the states to dimensional units AU/d/kg
posCRTBP = unitConversion.convertPos_to_dim(posCRTBP).to('AU')
pos_dim = posCRTBP[0]
v_dim = unitConversion.convertVel_to_dim(vI).to('AU/day')
Tp_dim = unitConversion.convertTime_to_dim(2*IC[6]).to('day').value

# convert position from I frame to H frame
C_B2G = frameConversion.body2geo(t_mjd,t_mjd,mu_star)
C_G2B = C_B2G.T
pos_GCRS = C_B2G@pos_dim

pos_ICRS = (frameConversion.gcrs2icrs(pos_GCRS,t_mjd)).to('AU').value

# convert velocity from I frame to H frame
v_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',t_mjd)[1].get_xyz().to('AU/day')
vel_ICRS = (v_EMO + v_dim).value

# Define the initial state array
state0 = np.append(np.append(pos_ICRS, vel_ICRS), days)   # Tp_dim

# propagate the dynamics
statesFF, timesFF = statePropFF(state0,t_mjd)
posFF = statesFF[:,0:3]
velFF = statesFF[:,3:6]

# preallocate space
r_PEM_r = np.zeros([len(timesFF),3])
r_SunEM_r = np.zeros([len(timesFF),3])
r_EarthEM_r = np.zeros([len(timesFF),3])
r_MoonEM_r = np.zeros([len(timesFF),3])

# sim time in mjd
timesFF_mjd = timesFF + t_mjd
for ii in np.arange(len(timesFF)):
    time = timesFF_mjd[ii]
    
    # positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',time)
    r_EMO = EMO[0].get_xyz().to('AU').value
    
    # convert from H frame to GCRS frame
    r_PG = frameConversion.icrs2gcrs(posFF[ii]*u.AU,time)
    r_EMG = frameConversion.icrs2gcrs(r_EMO*u.AU,time)
    r_SunG = frameConversion.icrs2gcrs(r_SunO*u.AU,time)
    r_MoonG = frameConversion.icrs2gcrs(r_MoonO*u.AU,time)
    
    # change the origin to the EM barycenter, G frame
    r_PEM = r_PG - r_EMG
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG
    
    # convert from G frame to I frame
    r_PEM_r[ii,:] = C_G2B@r_PEM.to('AU')
    r_SunEM_r[ii,:] = C_G2B@r_SunEM.to('AU')
    r_EarthEM_r[ii,:] = C_G2B@r_EarthEM.to('AU')
    r_MoonEM_r[ii,:] = C_G2B@r_MoonEM.to('AU')

# plots
#ax = plt.figure().add_subplot(projection='3d')

# plot CRTBP and FF solutions
#ax.plot(posCRTBP[:,0],posCRTBP[:,1],posCRTBP[:,2],'r',label='CRTBP')
#ax.plot(posFF[:,0],posFF[:,1],posFF[:,2],'b',label='Full Force')
#ax.scatter(r_PEM_r[0,0],r_PEM_r[0,1],r_PEM_r[0,2],marker='*',label='FF Start')
#ax.scatter(r_PEM_r[-1,0],r_PEM_r[-1,1],r_PEM_r[-1,2],label='FF End')
#plt.legend()

# plot the bodies and the FF solution
#ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2],'g',label='Earth')
#ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2],'r',label='Moon')
#ax.plot(r_SunEM_r[:,0],r_SunEM_r[:,1],r_SunEM_r[:,2],'y',label='Sun')
#ax.plot(r_PEM_r[:,0],r_PEM_r[:,1],r_PEM_r[:,2],'b',label='Full Force')
#ax.set_xlabel('X [AU]')
#ax.set_ylabel('Y [AU]')
#ax.set_zlabel('Z [AU]')
#plt.legend()

#plt.show()
breakpoint()
