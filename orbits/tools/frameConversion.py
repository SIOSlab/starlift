import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import GCRS, ICRS
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import sys
sys.path.insert(1, 'tools')
import unitConversion

# From JPL Horizons
# TU = 27.321582 d
# DU = 384400 km
# m_moon = 7.349x10^22 kg
# m_earth = 5.97219x10^24 kg


# =============================================================================
# Frame conversions
# =============================================================================

# rotation matrices
def rot(th, axis):
    """Finds the rotation matrix of angle th about the axis value

    Args:
        th (float):
            Rotation angle in radians
        axis (int):
            Integer value denoting rotation axis (1,2, or 3)

    Returns:
        ~numpy.ndarray(float):
            Rotation matrix

    """

    if axis == 1:
        rot_th = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(th), np.sin(th)],
                [0.0, -np.sin(th), np.cos(th)],
            ]
        )
    elif axis == 2:
        rot_th = np.array(
            [
                [np.cos(th), 0.0, -np.sin(th)],
                [0.0, 1.0, 0.0],
                [np.sin(th), 0.0, np.cos(th)],
            ]
        )
    elif axis == 3:
        rot_th = np.array(
            [
                [np.cos(th), np.sin(th), 0.0],
                [-np.sin(th), np.cos(th), 0.0],
                [0.0, 0.0, 1.0]
            ]
        )

    return rot_th


def body2geo(currentTime, equinox, mu_star):
    """Compute the directional cosine matrix to go from the Earth-Moon CR3BP
    perifocal frame (I) to the geocentric frame (G)
    
    Args:
        currentTime (astropy Time array):
            Current mission time in MJD
        equinox (astropy Time array):
            Mission start time in MJD
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        C_B2G (float n array):
            3x3 Array for the directional cosine matrix
    """
    
    # Define vector in G
    tmp = get_body_barycentric_posvel('Earth-Moon-Barycenter', equinox)[0].get_xyz()  # km
    tmp_rG = -icrs2gcrs(tmp, equinox)  # km
    tmp_x = unitConversion.convertPos_to_canonical(tmp_rG[0])
    tmp_y = unitConversion.convertPos_to_canonical(tmp_rG[1])
    tmp_z = unitConversion.convertPos_to_canonical(tmp_rG[2])
    r_earth_bary_G = np.array([tmp_x, tmp_y, tmp_z])
    mu_star = np.linalg.norm(r_earth_bary_G)
    
    # Define vector in R
    r_earth_bary_R = mu_star*np.array([-1, 0, 0])  # constant

    # Get DCM to go from R to I
    dt = currentTime.value - equinox.value  # days
    theta = unitConversion.convertTime_to_canonical(dt*u.d)
    C_B2R = rot(theta, 3)
    C_R2B = C_B2R.T

    # Vector from EM to Earth in I frame
    r_earth_bary_B = C_R2B @ r_earth_bary_R
    
    # Find the DCM to rotate vec 1 to vec 2
    n_vec = np.cross(r_earth_bary_B, r_earth_bary_G.T)
    n_hat = n_vec/np.linalg.norm(n_vec)

    r_skew = np.array([[0, -n_hat[2], n_hat[1]],
                       [n_hat[2], 0, -n_hat[0]],
                       [-n_hat[1], n_hat[0], 0]])

    r_sin = (np.linalg.norm(n_vec)/mu_star**2)
    r_cos = (np.dot(r_earth_bary_B/mu_star, r_earth_bary_G.T/mu_star))
    theta = np.arctan2(r_sin, r_cos)

    C_B2G = np.identity(3) + r_skew*np.sin(theta) + r_skew@r_skew*(1 - np.cos(theta))

    # # Anna trying something new for kicks
    # theta = np.dot(r_earth_bary_B, r_earth_bary_G)
    # n_vec = np.cross(r_earth_bary_B, r_earth_bary_G)
    # n_hat = n_vec/np.linalg.norm(n_vec)
    #
    # n_skew = np.array([[0, -n_hat[2], n_hat[1]],
    #                    [n_hat[2], 0, -n_hat[0]],
    #                    [-n_hat[1], n_hat[0], 0]])
    #
    # C_B2G = np.identity(3)*np.cos(theta) + np.sin(theta)*n_skew + (1-np.cos(theta))*n_hat*n_hat.T

    return C_B2G


def body2rot(currentTime, equinox):
    """Compute the directional cosine matrix to go from the Earth-Moon CR3BP
    perifocal frame (I) to the Earth-Moon CR3BP rotating frame (R)
    
    Args:
        currentTime (astropy Time array):
            Current mission time in MJD
        equinox (astropy Time array):
            Mission start time in MJD

    Returns:
        C_I2R (float n array):
            3x3 Array for the directional cosine matrix
    """

    dt = currentTime.value - equinox.value
    theta = unitConversion.convertTime_to_canonical(dt*u.d)
    
    C_I2R = rot(theta, 3)
    
    return C_I2R


def icrs2rot(pos, currentTime, equinox, mu_star, C_G2B):
    """Convert position vector in ICRS coordinate frame to rotating coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        equinox (astropy Time array):
            Mission start time in MJD
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        r_rot (astropy Quantity array):
            Position vector in rotating frame in km
    """
    
    state_EM = get_body_barycentric_posvel('Earth-Moon-Barycenter', equinox)
    r_EMG_icrs = state_EM[0].get_xyz().to('AU')

    # pos = pos * u.AU
    r_PE_gcrs = icrs2gcrs(pos, equinox)
    r_rot = r_PE_gcrs
    r_EME_gcrs = icrs2gcrs(r_EMG_icrs, equinox)
    r_PEM = r_PE_gcrs - r_EME_gcrs

    C_I2R = body2rot(currentTime, equinox)
    
    r_rot = C_G2B@C_I2R@r_PEM

    return r_rot


def gcrs2inert(pos,currentTime,equinox,mu_star):
    """Convert position vector in GCRS coordinate frame to inertial Earth-Moon CR3BP coordinate frame
    
    Args:
        pos (float n array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        equinox (astropy Time array):
            Mission start time in MJD
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        r_inert (float n array):
            Position vector in the inertial Earth-Moon CR3BP frame in km
    """
    
    C_B2G = body2geo(currentTime,equinox,mu_star)
    C_G2B = C_B2G.T
    
    r_inert = C_G2B @ pos
    return r_inert
    
def inert2gcrs(pos,currentTime,equinox,mu_star):
    """Convert position vector in inertial Earth-Moon CR3BP coordinate frame to GCRS coordinate frame
    
    Args:
        pos (float n array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        equinox (astropy Time array):
            Mission start time in MJD
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        r_gcrs (float n array):
            Position vector in the GCRS frame in km
    """
    
    C_B2G = body2geo(currentTime,equinox,mu_star)
    C_G2B = C_B2G.T
    
    r_gcrs = C_G2B @ pos
    return r_gcrs
    
def inert2rotP(pos, currentTime, equinox):
    """Convert position vector in inertial Earth-Moon CR3BP coordinate frame to rotating Earth-Moon CR3BP coordinate frame
    
    Args:
        pos (float n array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        equinox (astropy Time array):
            Mission start time in MJD

    Returns:
        r_rot (float n array):
            Position vector in the rotating Earth-Moon CR3BP frame in km
    """

    C_I2R = body2rot(currentTime, equinox)
    r_rot = C_I2R @ pos
    
    return r_rot


def rot2inertP(pos,currentTime,equinox):
    """Convert position vector in rotating Earth-Moon CR3BP coordinate frame to inertial Earth-Moon CR3BP coordinate frame
    
    Args:
        pos (float n array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        equinox (astropy Time array):
            Mission start time in MJD

    Returns:
        r_inert (float n array):
            Position vector in the rotating Earth-Moon CR3BP frame in km
    """

    C_I2R = body2rot(currentTime, equinox)
    C_R2I = C_I2R.T
    
    r_inert = C_R2I @ pos
    
    return r_inert


def icrs2gcrs(pos, currentTime):
    """Convert position vector in ICRS coordinate frame to GCRS coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD


    Returns:
        r_gcrs (astropy Quantity array):
            Position vector in GCRS (geocentric) frame in km
    """
    
    pos = pos.to('km')
    r_icrs = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km', representation_type='cartesian', frame='icrs')
    r_gcrs = r_icrs.transform_to(GCRS(obstime=currentTime))    # this throws an EFRA warning re: leap seconds, but it's fine
    r_gcrs = r_gcrs.cartesian.get_xyz()
    
    return r_gcrs

    
def gcrs2icrs(pos, currentTime):
    """Convert position vector in GCRS coordinate frame to ICRS coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in GCRS (geocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD


    Returns:
        r_icrs (float n array):
            Position vector in ICRS (heliocentric) frame in km
    """
    pos = pos.to('km')
    r_gcrs = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km', representation_type='cartesian', frame='gcrs', obstime=currentTime)
    r_icrs = r_gcrs.transform_to(ICRS())    # this throws an EFRA warning re: leap seconds, but it's fine
    r_icrs = r_icrs.cartesian.get_xyz()

    return r_icrs


def gcrs2icrsPV(pos, vel, currentTime):
    """Convert position vector in GCRS coordinate frame to ICRS coordinate frame
    
    Args:
        pos (float n array):
            Position vector in GCRS (geocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD


    Returns:
        r_icrs (float n array):
            Position vector in ICRS (heliocentric) frame in km
    """
    pos = pos.to('km')
    vel = vel.to('km/s')
    r_gcrs = coord.SkyCoord(x = pos[0], y = pos[1], z = pos[2], v_x = vel[0], v_y = vel[1], v_z = vel[2], representation_type='cartesian', frame='gcrs', obstime=currentTime)
    s_icrs = r_gcrs.transform_to(ICRS())    # this throws an EFRA warning re: leap seconds, but it's fine
    r_icrs = s_icrs.cartesian.get_xyz()
    v_icrs = s_icrs.velocity.get_d_xyz()

    return r_icrs, v_icrs


# velocity conversion
def rot2inertV(rR, vR, t_norm):
    """Convert velocity from rotating frame to inertial frame

    Args:
        rR (float nx3 array):
            Rotating frame position vectors
        vR (float nx3 array):
            Rotating frame velocity vectors
        t_norm (float):
            Normalized time units for current epoch
    Returns:
        float nx3 array:
            Inertial frame velocity vectors
    """

    if rR.shape[0] == 3 and len(rR.shape) == 1:
        At = rot(t_norm, 3).T
        drR = np.array([-rR[1], rR[0], 0])
        vI = np.dot(At, vR.T) + np.dot(At, drR.T)
    else:
        vI = np.zeros([len(rR), 3])
        for t in range(len(rR)):
            At = rot(t_norm, 3).T
            drR = np.array([-rR[t, 1], rR[t, 0], 0])
            vI[t, :] = np.dot(At, vR[t, :].T) + np.dot(At, drR.T)
    return vI
    

def inert2rotV(rR, vI, t_norm):
    """Convert velocity from inertial frame to rotating frame

    Args:
        rR (float nx3 array):
            Rotating frame position vectors
        vI (float nx3 array):
            Inertial frame velocity vectors
        t_norm (float):
            Normalized time units for current epoch
    Returns:
        float nx3 array:
            Rotating frame velocity vectors
    """
    
    if t_norm.size == 1:
        t_norm = np.array([t_norm])
    vR = np.zeros([len(t_norm), 3])
    for t in range(len(t_norm)):
        At = rot(t_norm[t], 3)
        vR[t, :] = np.dot(At, vI[t, :].T) + np.array([rR[t, 1], -rR[t, 0], 0]).T
    return vR


def convertIC_R2H(pos_R, vel_R, t_mjd, mu_star, Tp_can=None):
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

    C_B2G = body2geo(t_mjd, t_mjd, mu_star)
    pos_G = C_B2G @ pos_I

    state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_mjd)
    posEMB = state_EMB[0].get_xyz().to('AU')
    velEMB = state_EMB[1].get_xyz().to('AU/day')

    posEMB_E = (icrs2gcrs(posEMB, t_mjd)).to('AU')

    pos_GCRS = pos_G + posEMB_E  # G frame

    pos_H = (gcrs2icrs(pos_GCRS, t_mjd)).to('AU')

    vel_I = rot2inertV(np.array(pos_R), np.array(vel_R), 0)
    v_dim = unitConversion.convertVel_to_dim(vel_I).to('AU/day')
    vel_H = velEMB + v_dim

    if Tp_can is not None:
        Tp_dim = unitConversion.convertTime_to_dim(Tp_can).to('day')
        return pos_H, vel_H, Tp_dim
    else:
        return pos_H, vel_H


def convertIC_I2H(pos_I, vel_I, tau, t_mjd, mu_star, C_B2G, Tp_can=None):
    """Converts initial conditions from the I frame to the H frame

    Args:
        pos_I (float n array):
            Array of distance in canonical units
        vel_I (float n array):
            Array of velocities in canonical units
        tau (float)
            Current mission time
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

    pos_G = C_B2G @ pos_I

    state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', tau)
    posEMB = state_EMB[0].get_xyz().to('AU')
    velEMB = state_EMB[1].get_xyz().to('AU/day')

    posEMB_E = (icrs2gcrs(posEMB, tau)).to('AU')

    pos_GCRS = pos_G + posEMB_E  # G frame

    v_dim = unitConversion.convertVel_to_dim(vel_I).to('AU/day')
    vel_G = C_B2G @ v_dim

    pos_H, vel_H = gcrs2icrsPV(pos_GCRS, vel_G, tau)
    pos_H = pos_H.to('AU')
    vel_H = vel_H.to('AU/d')

    if Tp_can is not None:
        Tp_dim = unitConversion.convertTime_to_dim(Tp_can).to('day')
        return pos_H, vel_H, Tp_dim
    else:
        return pos_H, vel_H
