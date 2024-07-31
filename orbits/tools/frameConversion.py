import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import GCRS, ICRS, GeocentricMeanEcliptic
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


#def body2geo(currentTime, equinox, mu_star):
#    """Compute the directional cosine matrix to go from the Earth-Moon CR3BP
#    perifocal frame (I) to the geocentric frame (G)
#    
#    Args:
#        currentTime (astropy Time array):
#            Current mission time in MJD
#        equinox (astropy Time array):
#            Mission start time in MJD
#        mu_star (float):
#            Non-dimensional mass parameter
#
#    Returns:
#        C_B2G (float n array):
#            3x3 Array for the directional cosine matrix
#    """
#    
#    # Define vector in G
#    posM = get_body_barycentric_posvel('Moon', currentTime)[0].get_xyz()
#    posEM = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[0].get_xyz()
#    tmp_M = icrs2gcrs(posM, currentTime)
#    tmp_EM = icrs2gcrs(posEM, currentTime)
#    r_moon_bary_G = (tmp_M - tmp_EM).to('AU')
#    m_norm = np.linalg.norm(r_moon_bary_G).value
#    
#    # Define vector in B
#    r_moon_bary_R = np.array([m_norm, 0, 0])
#    
#    dt = currentTime.value - equinox.value
##    theta_BR = unitConversion.convertTime_to_canonical(dt*u.d)
#    theta_BR = rotAngle(currentTime,equinox).value
#        
#    C_B2R = rot(theta_BR, 3)
#    C_R2B = C_B2R.T
#    
#    r_moon_bary_B = C_R2B @ r_moon_bary_R
#    
#    # Find the DCM to rotate vec 1 to vec 2
#    n_vec = np.cross(r_moon_bary_B, r_moon_bary_G.T)
#    n_hat = n_vec/np.linalg.norm(n_vec)
#    
#    r_sin = (np.linalg.norm(n_vec)/(mu_star**2))
#    r_cos = (np.dot(r_moon_bary_B/mu_star, r_moon_bary_G.T/mu_star))
#    theta_GB = np.arctan2(r_sin, r_cos)
#    
#    r_skew = np.array([[0, -n_hat[2], n_hat[1]],
#                       [n_hat[2], 0, -n_hat[0]],
#                       [-n_hat[1], n_hat[0], 0]])
#                        
#    C_B2G = np.identity(3) + r_skew*np.sin(theta_GB) + r_skew@r_skew*(1 - np.cos(theta_GB))
#    print(theta_GB)
##    if theta_GB.value > 2:
##        breakpoint()
##    r_moon_bary_G = r_moon_bary_G.value
##    print(r_moon_bary_B)
#    
##    posS = get_body_barycentric_posvel('Sun', currentTime)[0].get_xyz()
##    tmp_S = icrs2gcrs(posS, currentTime)
##    r_sun_bary_G = (tmp_S - tmp_EM).to('AU').value
##    r_sun_bary_G = r_sun_bary_G*m_norm
##    r_sun_bary_B = C_B2G.T @ r_sun_bary_G
##    import matplotlib.pyplot as plt
##    ax1 = plt.figure().add_subplot(projection='3d')
##    ax1.plot([0, r_moon_bary_G[0]],[0, r_moon_bary_G[1]],[0, r_moon_bary_G[2]],'b',label='G frame')
##    ax1.plot([0, r_moon_bary_B[0]],[0, r_moon_bary_B[1]],[0, r_moon_bary_B[2]],'r',label='B frame')
##    ax1.plot([0, n_hat[0]*m_norm],[0, n_hat[1]*m_norm],[0, n_hat[2]*m_norm],'k',label='N hat')
###    ax1.plot([0, r_sun_bary_G[0]],[0, r_sun_bary_G[1]],[0, r_sun_bary_G[2]],'y',label='G frame')
###    ax1.plot([0, r_sun_bary_B[0]],[0, r_sun_bary_B[1]],[0, r_sun_bary_B[2]],'k',label='B frame')
##    ax1.set_xlim([-.003, .003])
##    ax1.set_ylim([-.003, .003])
##    ax1.set_zlim([-.003, .003])
##    plt.legend()
##    
##    
##    plt.show()
##    print(n_hat)
#    # try this function with the spacecraft?
#    # DCM is changing by a lot
#
#
#    return C_B2G
    
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
    tmp = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[0].get_xyz()  # km
    tmp_rG = -icrs2gcrs(tmp, currentTime)  # km
    tmp_x = unitConversion.convertPos_to_canonical(tmp_rG[0])
    tmp_y = unitConversion.convertPos_to_canonical(tmp_rG[1])
    tmp_z = unitConversion.convertPos_to_canonical(tmp_rG[2])
    r_earth_bary_G = np.array([tmp_x, tmp_y, tmp_z])
    mu_star = np.linalg.norm(r_earth_bary_G)
    
    # Define vector in B
    r_moon_bary_R = np.array([m_norm, 0, 0])
    
    dt = currentTime.value - equinox.value  # days
    theta = unitConversion.convertTime_to_canonical(dt*u.d)
    C_B2R = rot(theta, 3)
    C_R2B = C_B2R.T

    # Vector from EM to Earth in I frame
    r_earth_bary_B = C_R2B @ r_earth_bary_R
    
    # Find the DCM to rotate vec 1 to vec 2
    n_vec = np.cross(r_moon_bary_B, r_moon_bary_G.T)
    n_hat = n_vec/np.linalg.norm(n_vec)
    
    r_sin = (np.linalg.norm(n_vec)/(mu_star**2))
    r_cos = (np.dot(r_moon_bary_B/mu_star, r_moon_bary_G.T/mu_star))
    theta_GB = np.arctan2(r_sin, r_cos)
    
    r_skew = np.array([[0, -n_hat[2], n_hat[1]],
                       [n_hat[2], 0, -n_hat[0]],
                       [-n_hat[1], n_hat[0], 0]])
                        
    C_B2G = np.identity(3) + r_skew*np.sin(theta_GB) + r_skew@r_skew*(1 - np.cos(theta_GB))
    
#    theta_I = rotAngle(currentTime,equinox).value
#    C_I2B = rot(theta_I, 3)
#    C_B2G = C_I2B @ C_B2G
    
    return C_B2G
    
def rotAngle(currentTime,equinox):
    r_EM_ct = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[0].get_xyz()
    r_EM_et = get_body_barycentric_posvel('Earth-Moon-Barycenter', equinox)[0].get_xyz()
    
    r_EarthEM_ct = -icrs2gcrs(r_EM_ct, currentTime)
    r_EarthEM_et = -icrs2gcrs(r_EM_et, equinox)

    norm_ct = np.linalg.norm(r_EarthEM_ct)
    norm_et = np.linalg.norm(r_EarthEM_et)

    n_vec = np.cross(r_EarthEM_ct, r_EarthEM_et.T)
    
    dt = (currentTime - equinox).value
    t_mod = np.mod(dt, 27.321582)
    if t_mod < 27.321582/2:
        sign2 = 1
    elif t_mod > 27.321582/2 and t_mod < 27.321582:
        sign2 = -1
    r_sin = (np.linalg.norm(n_vec)/(norm_ct*norm_et))
    r_cos = (np.dot(r_EarthEM_ct/norm_ct, r_EarthEM_et.T/norm_et))
    theta = np.arctan2(sign2*r_sin, r_cos)

    return theta
    

def body2geo2(currentTime,equinox):
    tarray = equinox + np.arange(28)*u.d
    r_moon = get_body_barycentric_posvel('Moon', tarray)[0].get_xyz()
    r_bary = get_body_barycentric_posvel('Earth-Moon-Barycenter', tarray)[0].get_xyz()

    r_moons = r_moon - r_bary
    ctr = 0
    r_m = np.zeros([len(tarray), 3])
    for ii in tarray:
        tmp1 = (icrs2gcrs(r_moon[:,ctr],ii).to('AU').value)
        tmp2 = (icrs2gcrs(r_bary[:,ctr],ii).to('AU').value)
        r_m[ctr,:] = tmp1 - tmp2
        ctr = ctr + 1
    
    ZZ = r_m[:,2]
    signZ = np.sign(ZZ)
    diffZ = np.diff(signZ)
    indZ = np.argwhere(2==diffZ)[0][0]
    
    t1 = tarray[indZ]
    t2 = tarray[indZ + 1]
    dt = (t2 - t1)/2
    t3 = t1 + dt

    r_moon1 = (get_body_barycentric_posvel('Moon', t1)[0].get_xyz()).to('AU')
    r_moon2 = (get_body_barycentric_posvel('Moon', t2)[0].get_xyz()).to('AU')
    r_moon3 = (get_body_barycentric_posvel('Moon', t3)[0].get_xyz()).to('AU')
    r_bary1 = (get_body_barycentric_posvel('Earth-Moon-Barycenter', t1)[0].get_xyz()).to('AU')
    r_bary2 = (get_body_barycentric_posvel('Earth-Moon-Barycenter', t2)[0].get_xyz()).to('AU')
    r_bary3 = (get_body_barycentric_posvel('Earth-Moon-Barycenter', t3)[0].get_xyz()).to('AU')
    r_moon1 = icrs2gcrs(r_moon1,t1)
    r_moon2 = icrs2gcrs(r_moon2,t2)
    r_moon3 = icrs2gcrs(r_moon3,t3)
    r_bary1 = icrs2gcrs(r_bary1,t1)
    r_bary2 = icrs2gcrs(r_bary2,t2)
    r_bary3 = icrs2gcrs(r_bary3,t3)

    r_m1 = r_moon1 - r_bary1
    r_m2 = r_moon2 - r_bary2
    r_m3 = r_moon3 - r_bary3

    error = r_m3[2]

    while np.abs(error.value) > 6E-12:
        sign1 = np.sign(r_m1[2])
        sign2 = np.sign(r_m2[2])
        sign3 = np.sign(r_m3[2])

        if sign3 == sign1:
            t1 = t3
            r_m1 = r_m3
            
            dt = (t2 - t1)/2
            t3 = t3 + dt
            
        elif sign3 == sign2:
            t2 = t3
            r_m2 = r_m3
            
            dt = (t2 - t1)/2
            t3 = t1 + dt
            
        else:
            breakpoint()
            
        r_m = (get_body_barycentric_posvel('Moon', t3)[0].get_xyz()).to('AU')
        r_b = (get_body_barycentric_posvel('Earth-Moon-Barycenter', t3)[0].get_xyz()).to('AU')
        r_m = icrs2gcrs(r_m,t3)
        r_b = icrs2gcrs(r_b,t3)
        r_m3 = r_m - r_b
            
        error = r_m3[2]
        
    t_LAAN = t3
    t_LAAN2 = t_LAAN + 27.321582/4*u.d
    
    theta_LAAN = rotAngle(t_LAAN,equinox).value
    
    moon_LAAN1 = get_body_barycentric_posvel('Moon', t_LAAN)[0].get_xyz()
    moon_LAAN2 = get_body_barycentric_posvel('Moon', t_LAAN2)[0].get_xyz()
    bary_LAAN1 = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_LAAN)[0].get_xyz()
    bary_LAAN2 = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_LAAN2)[0].get_xyz()
    
    m_LAAN1 = icrs2gcrs(moon_LAAN1,t_LAAN)
    m_LAAN2 = icrs2gcrs(moon_LAAN2,t_LAAN2)
    b_LAAN1 = icrs2gcrs(bary_LAAN1,t_LAAN)
    b_LAAN2 = icrs2gcrs(bary_LAAN2,t_LAAN2)
    
    r_LAAN1 = m_LAAN1 - b_LAAN1
    r_LAAN2 = m_LAAN2 - b_LAAN2
    
    r_LAAN3 = np.cross(r_LAAN1, r_LAAN2)
    n_LAAN = r_LAAN3/np.linalg.norm(r_LAAN3)
    
    C_LAAN = rotMatAxisAng(n_LAAN, theta_LAAN)
    
    
    tarray = equinox + np.arange(2800)/100*u.d
    r_moons = get_body_barycentric_posvel('Moon', tarray)[0].get_xyz()
    r_barys = get_body_barycentric_posvel('Earth-Moon-Barycenter', tarray)[0].get_xyz()

    ctr = 0
    r_m = np.zeros([len(tarray), 3])

    for ii in tarray:
        tmp1 = C_LAAN @ (icrs2gcrs(r_moons[:,ctr],ii).to('AU').value)
        tmp2 = C_LAAN @ (icrs2gcrs(r_barys[:,ctr],ii).to('AU').value)
        r_m[ctr,:] = tmp1 - tmp2
        ctr = ctr + 1

    XX = max(r_m[:,0]) - min(r_m[:,0])
    YY = max(r_m[:,1]) - min(r_m[:,1])
    ZZ = max(r_m[:,2]) - min(r_m[:,2])
    
    theta_INC = -np.deg2rad(5.145) #-np.arctan2(ZZ,YY)

    n_INC = r_LAAN1/np.linalg.norm(r_LAAN1)

    C_INC = rotMatAxisAng(n_INC, theta_INC)

    r_norm = np.linalg.norm(r_m,axis=1)
    r_min = min(r_norm)
    
    r_ind = np.argwhere(r_min == r_norm)[0][0]

    t_AOP = tarray[r_ind]
    
    theta_AOP = rotAngle(t_AOP,t_LAAN).value
    
    n_AOP = C_INC @ C_LAAN @ n_LAAN
    
    C_AOP = rotMatAxisAng(n_AOP, theta_AOP)

    C_G2B = C_AOP @ C_INC @ C_LAAN
    C_B2G = C_G2B.T

    return C_B2G, C_LAAN, C_INC, C_AOP, n_LAAN, n_INC, n_AOP
    

        
def rotMatAxisAng(n_hat, theta):
    r_skew = np.array([[0, -n_hat[2], n_hat[1]],
                       [n_hat[2], 0, -n_hat[0]],
                       [-n_hat[1], n_hat[0], 0]])
                        
    R = np.identity(3) + r_skew*np.sin(theta) + r_skew@r_skew*(1 - np.cos(theta))
    
    return R


def body2rot(currentTime, equinox):
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
#    r_gcrs = r_icrs.transform_to(GCRS(obstime=currentTime))    # this throws an EFRA warning re: leap seconds, but it's fine
    r_gcrs = r_icrs.transform_to(GeocentricMeanEcliptic(obstime=currentTime))
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
    r_gcrs = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km', representation_type='cartesian', frame='GeocentricMeanEcliptic', obstime=currentTime)
    r_icrs = r_gcrs.transform_to(ICRS())    # this throws an EFRA warning re: leap seconds, but it's fine
    r_icrs = r_icrs.cartesian.get_xyz()

    return r_icrs
    
def icrs2gcrsPV(pos, vel, currentTime):
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
    vel = vel.to('km/s')
    r_icrs = coord.SkyCoord(x = pos[0], y = pos[1], z = pos[2], v_x = vel[0], v_y = vel[1], v_z = vel[2], representation_type='cartesian', frame='icrs', obstime=currentTime)
    s_gcrs = r_icrs.transform_to(GeocentricMeanEcliptic(obstime=currentTime))    # this throws an EFRA warning re: leap seconds, but it's fine
    r_gcrs = s_gcrs.cartesian.get_xyz()
    v_gcrs = s_gcrs.velocity.get_d_xyz()

    return r_gcrs, v_gcrs
    
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
