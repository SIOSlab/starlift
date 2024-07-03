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
    tmp = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[0].get_xyz()
    tmp_rG = -icrs2gcrs(tmp, currentTime)
    tmp_x = unitConversion.convertPos_to_canonical(tmp_rG[0])
    tmp_y = unitConversion.convertPos_to_canonical(tmp_rG[1])
    tmp_z = unitConversion.convertPos_to_canonical(tmp_rG[2])
    r_earth_bary_G = np.array([tmp_x, tmp_y, tmp_z])
    mu_star = np.linalg.norm(r_earth_bary_G)
    
    # Define vector in B
    r_earth_bary_R = mu_star*np.array([-1, 0, 0])
    
    dt = currentTime.value - equinox.value
    theta = unitConversion.convertTime_to_canonical(dt*u.d)
    C_B2R = rot(theta, 3)
    C_R2B = C_B2R.T
    
    r_earth_bary_B = C_R2B @ r_earth_bary_R
    
    # Find the DCM to rotate vec 1 to vec 2
    n_vec = np.cross(r_earth_bary_B, r_earth_bary_G.T)
    n_hat = n_vec/np.linalg.norm(n_vec)
    
    r_sin = (np.linalg.norm(n_vec)/mu_star**2)
    r_cos = (np.dot(r_earth_bary_B/mu_star,r_earth_bary_G.T/mu_star))
    theta = np.arctan2(r_sin,r_cos)
    
    r_skew = np.array([[0, -n_hat[2], n_hat[1]],
                       [n_hat[2], 0, -n_hat[0]],
                       [-n_hat[1], n_hat[0], 0]])
                        
    C_B2G = np.identity(3) + r_skew*np.sin(theta) + r_skew@r_skew*(1 - np.cos(theta))

    return C_B2G


def body2rot(currentTime,equinox):
    """Compute the directional cosine matrix to go from the Earth-Moon CR3BP
    perifocal frame to the Earth-Moon CR3BP rotating frame
    
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


# position conversions
def icrs2rot(pos,currentTime,equinox,mu_star,C_G2B):
    """Convert position vector in ICRS coordinate frame to rotating coordinate frame
    
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
        r_rot (float n array):
            Position vector in rotating frame in km
    """
    
    state_EM = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)
    r_EMG_icrs = state_EM[0].get_xyz().to('AU')
    
    r_PE_gcrs = icrs2gcrs(pos,currentTime)
    r_rot = r_PE_gcrs
    r_EME_gcrs = icrs2gcrs(r_EMG_icrs,currentTime)
    r_PEM = r_PE_gcrs - r_EME_gcrs

    C_I2R = body2rot(currentTime,equinox)
    
    r_rot = C_G2B@C_I2R@r_PEM
#    breakpoint()
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
    
def inert2rotP(pos,currentTime,equinox):
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

def icrs2gcrs(pos,currentTime):
    """Convert position vector in ICRS coordinate frame to GCRS coordinate frame
    
    Args:
        pos (float n array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD


    Returns:
        r_gcrs (float n array):
            Position vector in GCRS (geocentric) frame in km
    """
    
    pos = pos.to('km')
    r_icrs = coord.SkyCoord(x = pos[0].value, y = pos[1].value, z = pos[2].value, unit='km', representation_type='cartesian', frame='icrs')
    r_gcrs = r_icrs.transform_to(GCRS(obstime=currentTime))    # this throws an EFRA warning re: leap seconds, but it's fine
    r_gcrs = r_gcrs.cartesian.get_xyz()
    
    return r_gcrs

    
def gcrs2icrs(pos, currentTime):
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
    r_gcrs = coord.SkyCoord(x = pos[0].value, y = pos[1].value, z = pos[2].value, unit='km', representation_type='cartesian', frame='gcrs', obstime=currentTime)
    r_icrs = r_gcrs.transform_to(ICRS())    # this throws an EFRA warning re: leap seconds, but it's fine
    r_icrs = r_icrs.cartesian.get_xyz()

    return r_icrs


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
