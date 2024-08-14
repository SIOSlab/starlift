import numpy as np
from scipy.integrate import solve_ivp
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import astropy.constants as const
import astropy.units as u
import frameConversion
import unitConversion
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


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


def CRTBP_EOM_R(t, w, mu_star):
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

    [x, y, z, vx, vy, vz] = w

    m1 = 1 - mu_star
    m2 = mu_star

    r1 = mu_star
    r2 = 1 - mu_star
    r_1O_H = r1 * np.array([-1, 0, 0])
    r_2O_H = r2 * np.array([1, 0, 0])

    r_PO_H = np.array([x, y, z])
    v_PO_H = np.array([vx, vy, vz])

    r_P1_H = r_PO_H - r_1O_H
    r_P2_H = r_PO_H - r_2O_H
    r1_mag = np.linalg.norm(r_P1_H)
    r2_mag = np.linalg.norm(r_P2_H)

    F_g1 = -m1 / r1_mag ** 3 * r_P1_H
    F_g2 = -m2 / r2_mag ** 3 * r_P2_H
    F_g = F_g1 + F_g2

    e3_hat = np.array([0, 0, 1])

    a_PO_H = F_g - 2 * np.cross(e3_hat, v_PO_H) - np.cross(e3_hat, np.cross(e3_hat, r_PO_H))
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx, vy, vz, ax, ay, az]

    return dw


def FF_EOM(tt, w, t_mjd):
    """Equations of motion for the full force model in the ICRS frame

    Args:
        w (~numpy.ndarray(float)):
            State [position in AU, velocity in AU/d]
        t_mjd (astropy Time array):
            Mission start time in MJD

    Returns:
        ~numpy.ndarray(float):
            Time derivative of the state [velocity in AU/d, acceleration in AU/d^2]

    """

    # H frame
    [x, y, z, vx, vy, vz] = w
    
    gmSun = const.GM_sun.to('AU3/d2').value        # in AU^3/d^2
    gmEarth = const.GM_earth.to('AU3/d2').value
    gmMoon = 0.109318945437743700E-10              # from de432s header
    gmJupiter = const.GM_jup.to('AU3/d2').value
    
    r_PO = np.array([x, y, z])  # AU
    v_PO = np.array([vx, vy, vz])  # AU/d

    time = unitConversion.convertTime_to_dim(tt) + t_mjd  # Current mission time in mjd

    # Get Sun, Moon, and Earth positions at the current time in the H frame [AU]
    r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU')
    r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU')
    r_EarthO = get_body_barycentric_posvel('Earth', time)[0].get_xyz().to('AU')
    r_JupiterO = get_body_barycentric_posvel('Jupiter', time)[0].get_xyz().to('AU')
    
    # Distance vectors
    r_PSun = r_PO - r_SunO.value
    r_PEarth = r_PO - r_EarthO.value
    r_PMoon = r_PO - r_MoonO.value
    r_PJupiter = r_PO - r_JupiterO.value

    # Magnitudes
    rSun_mag = np.linalg.norm(r_PSun)
    rEarth_mag = np.linalg.norm(r_PEarth)
    rMoon_mag = np.linalg.norm(r_PMoon)
    rJupiter_mag = np.linalg.norm(r_PJupiter)

    # Equations of motion
    F_gSun_p = -gmSun*(r_PSun/rSun_mag**3)
    F_gEarth_p = -gmEarth*(r_PEarth/rEarth_mag**3)
    F_gMoon_p = -gmMoon*(r_PMoon/rMoon_mag**3)
    F_gJupiter_p = -gmJupiter*(r_PJupiter/rJupiter_mag**3)

    F_g = F_gSun_p + F_gEarth_p + F_gMoon_p + F_gJupiter_p
    
    a_PO = F_g
    
    ax = a_PO[0]
    ay = a_PO[1]
    az = a_PO[2]

    dw = [vx, vy, vz, ax, ay, az]
    
    return dw


def statePropCRTBP(freeVar, mu_star):
    """Propagates the dynamics using the free variables in the CRTBP

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions (DU), y velocity (DU/TU), and orbit period (DU) in the I frame
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

    # sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,), method='LSODA', first_step=0.0001, min_step=1E-10, max_step=2700, rtol=1E-12, atol=1E-12)
    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,), rtol=1E-12, atol=1E-12)
    states = sol_int.y.T
    times = sol_int.t
    
    return states, times


def statePropCRTBP_R(freeVar, mu_star):
    """Propagates the dynamics using the free variables in the rotating frame

    Args:
        freeVar (float np.array):
            Free variable in non-dimensional units of the form [x y z dx dy dz T/2]
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        tuple:
        states ~numpy.ndarray(float):
            Positions and velocities in non-dimensional units
        times ~numpy.ndarray(float):
            Times in non-dimensional units

    """

    x0 = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    T = freeVar[-1]

    sol_int = solve_ivp(CRTBP_EOM_R, [0, T], x0, args=(mu_star,), rtol=1E-12, atol=1E-12, )
    states = sol_int.y.T
    times = sol_int.t
    return states, times


def statePropFF(state0, t_mjd, timesTMP=None):
    """Propagates the dynamics using the free variables in the full force model

    Args:
        state0 (~numpy.ndarray(float)):
            Position [AU], velocity [AU/d], and propagation time [DU] in the H frame
        t_mjd (astropy Time array):
            Mission start time in MJD

    Returns:
        tuple:
        states ~numpy.ndarray(float):
            Positions and velocities in AU and AU/d
        times ~numpy.ndarray(float):
            Canonical times

    """
    
    T = state0[-1]

    sol_int = solve_ivp(FF_EOM, [0, T], state0[0:6], args=(t_mjd,), method='LSODA')
#    sol_int = solve_ivp(FF_EOM, [0, T], state0[0:6], args=(t_mjd,), rtol=1E-12, atol=1E-12, method='LSODA',t_eval=timesTMP)

    states = sol_int.y.T
    times = sol_int.t
    
    return states, times


def calcFx_CRTBP(freeVar, mu_star):
    """Applies constraints to the free variables in the CRTBP inertial frame

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            constraint array

    """
    
    s_T = statePropCRTBP(freeVar, mu_star)
    state = s_T[-1]

    Fx = np.array([state[1], state[3], state[5]])
    return Fx


def calcFx_R(freeVar, mu_star):
    """Applies constraints to the free variables in the CRTBP rotating frame

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            constraint array

    """
    
    s_T, times = statePropCRTBP_R(freeVar, mu_star)
    state = s_T[-1]

    Fx = np.array([state[1], state[3], state[5]])
    return Fx


def calcFx_FF(X, taus, N, X0, dt):
    """Applies constraints to the free variables for a full force model

    *Add documentation

    """
    ctr = 0
    Fx = np.array([])

    for ii in np.arange(N):
        IC = np.append(X[ctr * 6:((ctr + 1) * 6)], dt)
        tau = taus[ctr]
        const = X0[ctr * 6:((ctr + 1) * 6)]
        states, times = statePropFF(IC, tau)

        Fx = np.append(Fx, states[-1, :] - const)

        ctr = ctr + 1
    return Fx


def calcdFx_CRTBP(freeVar, mu_star, m1, m2):
    """Calculates the Jacobian of the free variables wrt the constraints for the CRTBP

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period
        mu_star (float):
            Non-dimensional mass parameter
        m1 (float):
            Mass of the larger primary in non-dimensional units
        m2 (float):
            Mass of the smaller primary in non-dimensional units

    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """

    s_T = statePropCRTBP_R(freeVar, mu_star)
    state = s_T[-1]

    Phi = calcMonodromyMatrix(state, mu_star, m1, m2)

    phis = np.array([[Phi[1, 0], Phi[1, 2], Phi[1, 4]],
                     [Phi[3, 0], Phi[3, 2], Phi[3, 4]],
                     [Phi[5, 0], Phi[5, 2], Phi[5, 4]]])

    X = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    dw = CRTBP_EOM(freeVar[-1], X, mu_star)
    ddT = np.array([dw[1], dw[3], dw[5]])
    dFx = np.zeros([3, 4])
    dFx[:, 0:3] = phis
    dFx[:, 3] = ddT

    return dFx


def calcdFx_FF(X, taus, N, X0, dt):
    """Calculates the Jacobian of the free variables wrt the constraints for the full force model

    *Add documentation

    """
    hstep = 1E-4

    Fx_0 = calcFx_FF(X, taus, N, X0, dt)

    dFx = np.zeros((6 * N, 6 * N))
    indsXh = np.arange(0, N * 6, 6)
    indsD = np.arange(0, N * 6, 6)
    for ii in np.arange(6):
        dh = np.zeros(N * 6)
        dh[indsXh] = hstep

        Xh = X + dh

        Fx_ii = calcFx_FF(Xh, taus, N, X0, dt)

        dFx_ii = (Fx_ii - Fx_0) / hstep

        for jj in np.arange(len(indsD)):
            ind1 = indsD[jj]
            ind2 = ind1 + 6
            dFx[jj * 6:(jj + 1) * 6, ind1] = dFx_ii[jj * 6:(jj + 1) * 6]

        indsD = indsD + 1
        indsXh = indsXh + 1

    return dFx


def fsolve_eqns(w, z, solp, mu_star):
    """Finds the initial guess for a new orbit and the Jacobian for continuation

    Args:
        w (~numpy.ndarray(float)):
            symbolic representation free variables solution
        z (~numpy.ndarray(float)):
            tangent vector to move along to find corrected free variables
        solp (~numpy.ndarray(float)):
            next free variables prediction
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            system of equations as a function of w

    """
    
    Fx = calcFx_R(w, mu_star)
    zeq = z.T@(w-solp)
    sys_w = np.append(Fx, zeq)

    return sys_w


def generateFamily_CRTBP(guess, mu_star, N):
    """Generates and plots a family of orbits in the CRTBP model given a guess for the initial state.
    Each orbit is generated over 1 orbital period.

    Args:
        guess (float array):
            Initial guess for the orbit family in the form [x y z dx dy dz T/2], where T is the orbit period
        mu_star (float):
            Non-dimensional mass parameter
        N (float):
            Number of orbits in the family to be generated

    Returns:
        ICs (float n array):
            A matrix consisting of the initial states of for all the orbits in the family

    """

    # Parameters
    m1 = (1 - mu_star)
    m2 = mu_star

    # Initial guess for the free variable vector
    X = [guess[0], guess[2], guess[4], guess[6]]

    eps = 1E-6
    solutions = np.zeros([N, 4])
    z = np.array([0, 0, 0, 1])
    step = 1E-2

    max_iter = 1000
    ax = plt.figure().add_subplot(projection='3d')
    ICs = np.zeros([N, 7])
    for ii in np.arange(N):
        error = 10
        ctr = 0
        while error > eps and ctr < max_iter:
            # Generate the free variable vector
            Fx = calcFx_R(X, mu_star)
            error = np.linalg.norm(Fx)
            dFx = calcdFx_CRTBP(X, mu_star, m1, m2)
            X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)
            ctr = ctr + 1

        # Generate an orbit from the found free variable vector
        freeVar = np.array([X[0], X[1], X[2], 2*X[3]])
        # solutions[ii] = freeVar
        # states, times = statePropCRTBP_R(freeVar, mu_star)

        ICs[ii] = [X[0], 0, X[1], 0, X[2], 0, X[3]]

        # # Plot the orbit
        # ax.plot(states[:, 0], states[:, 1], states[:, 2])

        # Generate new z and X for another orbit
        solp = X + z * step
        ss = fsolve(fsolve_eqns, X, args=(z, solp, mu_star), full_output=True, xtol=1E-12)
        X = ss[0]
        Q = ss[1]['fjac']
        Rs = ss[1]['r']
        R = np.zeros((4, 4))
        idx, col = np.triu_indices(4, k=0)
        R[idx, col] = Rs
        J = Q.T @ R

        z = np.linalg.inv(J) @ z
        z = z / np.linalg.norm(z)

    # plt.show()
    return ICs


def calcMonodromyMatrix(freeVar, mu_star, m1, m2):
    """Calculates the monodromy matrix

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period
        mu_star (float):
            Non-dimensional mass parameter
        m1 (float):
            Mass of the larger primary in non-dimensional units
        m2 (float):
            Mass of the smaller primary in non-dimensional units

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

    dxdx = 1 - m1 / rp1 ** 3 - m2 / rp2 ** 3 + 3 * m1 * (x + m2) ** 2 / rp1 ** 5 + 3 * m2 * (x - 1 + m2) ** 2 / rp2 ** 5
    dxdy = 3 * m1 * (x + m2) * y / rp1 ** 5 + 3 * m2 * (x - 1 + m2) * y / rp2 ** 5
    dxdz = 3 * m1 * (x + m2) * z / rp1 ** 5 + 3 * m2 * (x - 1 + m2) * z / rp2 ** 5
    dydy = 1 - m1 / rp1 ** 3 - m2 / rp2 ** 3 + 3 * m1 * y ** 2 / rp1 ** 5 + 3 * m2 * y ** 2 / rp2 ** 5
    dydz = 3 * m1 * y * z / rp1 ** 5 + 3 * m2 * y * z / rp2 ** 5
    dzdz = 1 - m1 / rp1 ** 3 - m2 / rp2 ** 3 + 3 * m1 * z ** 2 / rp1 ** 5 + 3 * m2 * z ** 2 / rp2 ** 5

    Z = np.zeros([3, 3])
    I = np.identity(3)
    A = np.array([[dxdx, dxdy, dxdz],
                  [dxdy, dydy, dydz],
                  [dxdz, dydz, dzdz]])
    Phi = np.block([[Z, I], [A, Z]])

    return Phi
