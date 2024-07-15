import numpy as np
from scipy.integrate import solve_ivp
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import astropy.constants as const
import astropy.units as u
import frameConversion
import unitConversion


def CRTBP_EOM(t, w, mu_star):
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
    
    [x, y, z, vx, vy, vz] = w
    
    m1 = 1 - mu_star
    m2 = mu_star

    r1 = mu_star
    r2 = 1 - mu_star
    r_1O_R = r1*np.array([-1, 0, 0])
    r_2O_R = r2*np.array([1, 0, 0])
    
    C_I2R = frameConversion.rot(t, 3)
    C_R2I = C_I2R.T
    
    r_1O_I = C_R2I@r_1O_R
    r_2O_I = C_R2I@r_2O_R

    r_PO_I = np.array([x, y, z])
    v_PO_I = np.array([vx, vy, vz])

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

    dw = [vx, vy, vz, ax, ay, az]
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

    dw = [vx, vy, vz, ax, ay, az]
    
    return dw


def statePropCRTBP(freeVar, mu_star):
    """Propagates the dynamics using the free variables

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions (DU), y velocity (DU/TU), and orbit period (DU)
        mu_star (float):
            Non-dimensional mass parameter


    Returns:
        tuple:
        states ~numpy.ndarray(float):
            Positions and velocities in DU and DU/TU
        times ~numpy.ndarray(float):
            Times in DU

    """
    
    x0 = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    T = freeVar[-1]

    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,), rtol=1E-12, atol=1E-12,)
    states = sol_int.y.T
    times = sol_int.t
    
    return states, times


def statePropFF(state0, t_mjd):
    """Propagates the dynamics using the free variables

    Args:
        state0 (~numpy.ndarray(float)):
            Position and velocity in the H frame
        t_mjd (astropy Time array):
            Mission start time in MJD


    Returns:
        tuple:
        states ~numpy.ndarray(float):
            Positions and velocities in AU and AU/d
        times ~numpy.ndarray(float):
            Times in d

    """
    
    T = state0[-1]

#    sol_int = solve_ivp(FF_EOM, [0, T], state0[0:6], args=(t_mjd,), method='LSODA',t_eval=np.arange(0,T,1E-4))
    sol_int = solve_ivp(FF_EOM, [0, T], state0[0:6], args=(t_mjd,), rtol=1E-12, atol=1E-12, method='LSODA')

    states = sol_int.y.T
    times = sol_int.t
    
    return states, times

def CRTBP_EOM_R(t,w,mu_star):
    """Equations of motion for the CRTBP in the rotating frame

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

def calcMonodromyMatrix(freeVar,mu_star,m1,m2):
    """Calculates the monodromy matrix

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period
        mu_star (float):
            Non-dimensional mass parameter
        m1 (float):
            Mass of the larger primary in non dimensional units
        m2 (float):
            Mass of the smaller primary in non dimensional units

    Returns:
        ~numpy.ndarray(float):
            monodromy matrix

    """
    
    x = freeVar[0]
    y = freeVar[1]
    z = freeVar[2]

    r_P0 = np.array([x, y, z])
    r_m10 = -np.array([mu_star, 0, 0])
    r_m20 = np.array([(1 - mu_star), 0, 0])
    
    r_Pm1 = r_P0 - r_m10
    r_Pm2 = r_P0 - r_m20

    rp1 = np.linalg.norm(r_Pm1)
    rp2 = np.linalg.norm(r_Pm2)

    dxdx = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*(x + m2)**2/rp1**5 + 3*m2*(x - 1 + m2)**2/rp2**5
    dxdy = 3*m1*(x + m2)*y/rp1**5 + 3*m2*(x - 1 + m2)*y/rp2**5
    dxdz = 3*m1*(x + m2)*z/rp1**5 + 3*m2*(x - 1 + m2)*z/rp2**5
    dydy = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*y**2/rp1**5 + 3*m2*y**2/rp2**5
    dydz = 3*m1*y*z/rp1**5 + 3*m2*y*z/rp2**5
    dzdz = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*z**2/rp1**5 + 3*m2*z**2/rp2**5

    Z = np.zeros([3,3])
    I = np.identity(3)
    A = np.array([[dxdx, dxdy, dxdz],
            [dxdy, dydy, dydz],
            [dxdz, dydz, dzdz]])
    Phi = np.block([[Z, I],
           [A, Z]])
           
    return Phi

def calcdFx(freeVar,mu_star,m1,m2):
    """Calculates the jacobian of the free variables wrt the constraints

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period
        mu_star (float):
            Non-dimensional mass parameter
        m1 (float):
            Mass of the larger primary in non dimensional units
        m2 (float):
            Mass of the smaller primary in non dimensional units
            
            
    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    
    s_T = stateProp_R(freeVar,mu_star)
    state = s_T[-1]
    
    Phi = calcMonodromyMatrix(state, mu_star,m1,m2)

    phis = np.array([[Phi[1,0], Phi[1,2], Phi[1,4]],
                    [Phi[3,0], Phi[3,2], Phi[3,4]],
                    [Phi[5,0], Phi[5,2], Phi[5,4]]])

    X = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    dw = CRTBP_EOM(freeVar[-1], X, mu_star)
    ddT = np.array([dw[1], dw[3], dw[5]])
    dFx = np.zeros([3,4])
    dFx[:,0:3] = phis
    dFx[:,3] = ddT
    
    return dFx

def calcFx(freeVar,mu_star):
    """Applies constraints to the free variables

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period


    Returns:
        ~numpy.ndarray(float):
            constraint array

    """
    
    s_T = stateProp(freeVar)
    state = s_T[-1]

    Fx = np.array([state[1], state[3], state[5]])
    return Fx
    
def calcFx_R(freeVar, mu_star):
    """Applies constraints to the free variables

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period


    Returns:
        ~numpy.ndarray(float):
            constraint array

    """
    
    s_T, times = stateProp_R(freeVar, mu_star)
    state = s_T[-1]

    Fx = np.array([state[1], state[3], state[5]])
    return Fx

def stateProp_R(freeVar,mu_star):
    """Propagates the dynamics using the free variables

    Args:
        state (float):
            Position in non dimensional units
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

    sol_int = solve_ivp(CRTBP_EOM_R, [0, T], x0, args=(mu_star,),rtol=1E-12,atol=1E-12,)
    states = sol_int.y.T
    times = sol_int.t
    return states, times
    
def fsolve_eqns(w,z,solp, mu_star):
    """Finds the initial guess for a new orbit and the Jacobian for continuation

    Args:
        w (~numpy.ndarray(float)):
            symbolic representation free variables solution
        z (~numpy.ndarray(float)):
            tangent vector to move along to find corrected free variables
        solp (~numpy.ndarray(float)):
            next free variables prediction


    Returns:
        ~numpy.ndarray(float):
            system of equations as a function of w

    """
    
    Fx = calcFx_R(w, mu_star)
    zeq = z.T@(w-solp)
    sys_w = np.append(Fx,zeq)

    return sys_w


def convertIC_R2H(pos_R, vel_R, t_mjd, mu_star, Tp_can = None):
    """Converts initial conditions from the R frame to the H frame

    Args:
        pos_R (float n array):
            Array of distance in canonical units
        vel_R (float n array):
            Array of velocities in canonical units
        t_mjd (astropy Time array):
            Mission start time in MJD
        mu_star (float):
            Non-dimensional mass parameter
        Tp_can (float n array, optional):
            Optional array of times in canonical units


    Returns:
        tuple:
        pos_H (float n array):
            Array of distance in AU
        vel_H (float n array):
            Array of velocities in AU/day
        Tp_dim (float n array):
            Array of times in units of days

    """
    
    pos_I = unitConversion.convertPos_to_dim(pos_R).to('AU')
    
    C_B2G = frameConversion.body2geo(t_mjd, t_mjd, mu_star)
    pos_G = C_B2G@pos_I
    
    state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_mjd)
    posEMB = state_EMB[0].get_xyz().to('AU')
    velEMB = state_EMB[1].get_xyz().to('AU/day')
    
    posEMB_E = (frameConversion.icrs2gcrs(posEMB, t_mjd)).to('AU')

    pos_GCRS = pos_G + posEMB_E  # G frame
    
    pos_H = (frameConversion.gcrs2icrs(pos_GCRS, t_mjd)).to('AU')
    
    vel_I = frameConversion.rot2inertV(np.array(pos_R), np.array(vel_R), 0)
    v_dim = unitConversion.convertVel_to_dim(vel_I).to('AU/day')
    vel_H = velEMB + v_dim
    
    if Tp_can is not None:
        Tp_dim = unitConversion.convertTime_to_dim(Tp_can).to('day')
        return pos_H, vel_H, Tp_dim
    else:
        return pos_H, vel_H
        
def convertIC_I2H(pos_I, vel_I, tau, t_mjd, mu_star, C_B2G, Tp_can = None):
    """Converts initial conditions from the I frame to the H frame

    Args:
        pos_I (float n array):
            Array of distance in canonical units
        vel_I (float n array):
            Array of velocities in canonical units
        t_mjd (astropy Time array):
            Mission start time in MJD
        mu_star (float):
            Non-dimensional mass parameter
        Tp_can (float n array, optional):
            Optional array of times in canonical units


    Returns:
        tuple:
        pos_H (float n array):
            Array of distance in AU
        vel_H (float n array):
            Array of velocities in AU/day
        Tp_dim (float n array):
            Array of times in units of days

    """
    
    pos_I = unitConversion.convertPos_to_dim(pos_I).to('AU')
    
    pos_G = C_B2G@pos_I
    
    state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', tau)
    posEMB = state_EMB[0].get_xyz().to('AU')
    velEMB = state_EMB[1].get_xyz().to('AU/day')
    
    posEMB_E = (frameConversion.icrs2gcrs(posEMB, tau)).to('AU')

    pos_GCRS = pos_G + posEMB_E  # G frame
    
    v_dim = unitConversion.convertVel_to_dim(vel_I).to('AU/day')
    vel_G = C_B2G @ v_dim
    pos_H, vel_H = frameConversion.gcrs2icrsPV(pos_GCRS, vel_G, tau)
    pos_H = pos_H.to('AU')
    vel_H = vel_H.to('AU/d')
    
    if Tp_can is not None:
        Tp_dim = unitConversion.convertTime_to_dim(Tp_can).to('day')
        return pos_H, vel_H, Tp_dim
    else:
        return pos_H, vel_H
    
def calcFx_FF(X,taus,N,t_mjd,X0,dt):
    
    ctr = 0
    Fx = np.array([])

    for ii in np.arange(N):
        IC = np.append(X[ctr*6:((ctr+1)*6)],dt)
        tau = taus[ctr]
        const = X0[ctr*6:((ctr+1)*6)]
        states, times = statePropFF(IC,tau)

        Fx = np.append(Fx,states[-1,:] - const)
        
        ctr = ctr + 1
    return Fx
    
def calcdFx_FF(X,taus,N,t_mjd,X0,dt):
    hstep = 1E-4
    
    Fx_0 = calcFx_FF(X,taus,N,t_mjd,X0,dt)
    
    dFx = np.zeros((6*N,6*N))
    indsXh = np.arange(0,N*6,6)
    indsD = np.arange(0,N*6,6)
    for ii in np.arange(6):
        dh = np.zeros(N*6)
        dh[indsXh] = hstep

        Xh = X + dh
        
        Fx_ii = calcFx_FF(Xh,taus,N,t_mjd,X0,dt)
        
        dFx_ii = (Fx_ii - Fx_0)/hstep
        
        for jj in np.arange(len(indsD)):
            ind1 = indsD[jj]
            ind2 = ind1 + 6
            dFx[jj*6:(jj+1)*6,ind1] = dFx_ii[jj*6:(jj+1)*6]
        
        indsD = indsD + 1
        indsXh = indsXh + 1

    return dFx
