import numpy as np
from scipy.integrate import solve_ivp
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import astropy.constants as const
import frameConversion

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
    
    a_PO = F_g
    
    ax = a_PO[0]
    ay = a_PO[1]
    az = a_PO[2]

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
