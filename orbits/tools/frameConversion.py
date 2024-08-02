import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import GCRS, ICRS, GeocentricMeanEcliptic, GeocentricTrueEcliptic
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

    
def equinoxAngle(r_LAAN, t_LAAN, equinox):
    """Finds the angle between the GMECL equinox and the moon's ascending node

    Args:
        startTime (astropy Time array):
            Mission start time in MJD
        equinox (astropy Time array):
            Equinox time in MJD

    Returns:
        theta (float):
            Angle between the two vectors in rad

    """
    
    r_S_et = get_body_barycentric_posvel('Sun', equinox)[0].get_xyz()
    r_EM_et = get_body_barycentric_posvel('Earth-Moon-Barycenter', equinox)[0].get_xyz()
    
    r_Sun_et = icrs2gmec(r_S_et, equinox)
    r_Bary_et = icrs2gmec(r_EM_et, equinox)
    
    r_SB_et = r_Sun_et - r_Bary_et
    n_SB_et = r_SB_et/np.linalg.norm(r_SB_et)
    n_LAAN = r_LAAN/np.linalg.norm(r_LAAN)
    
    dt = (t_LAAN - equinox).value
    t_mod = np.mod(dt, 27.321582)
    if t_mod < 27.321582/2:
        sign2 = 1
    elif t_mod > 27.321582/2 and t_mod < 27.321582:
        sign2 = -1
        
#    import matplotlib.pyplot as plt
#    ax2 = plt.figure().add_subplot(projection='3d')
#    ax2.plot(np.array([0, n_LAAN[0]]), np.array([0, n_LAAN[1]]), np.array([0, n_LAAN[2]]), 'b', label='LAAN')
#    ax2.plot(np.array([0, n_SB_et[0]]), np.array([0,n_SB_et[1]]), np.array([0,n_SB_et[2]]), 'r-.', label='Equinox')
#    ax2.set_title('G frame (Inertial EM)')
#    ax2.set_xlabel('X [AU]')
#    ax2.set_ylabel('Y [AU]')
#    ax2.set_zlabel('Z [AU]')
#    plt.legend()
#    plt.show()
#    breakpoint()

    r_sin = np.linalg.norm(np.cross(n_LAAN, n_SB_et))
    r_cos = np.dot(n_LAAN, n_SB_et)
    theta = np.arctan2(sign2*r_sin, r_cos)
    breakpoint()
    return theta
    
def rotAngle(currentTime, startTime):
    """Finds the angle of rotation between two vectors in any Earth-Moon-Barycenter centered frame

    Args:
        currentTime (astropy Time array):
            Current mission time in MJD
        startTime (astropy Time array):
            Mission start time in MJD

    Returns:
        theta (float):
            Angle between the two vectors in rad

    """
    
    
    r_EM_ct = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[0].get_xyz()
    r_EM_st = get_body_barycentric_posvel('Earth-Moon-Barycenter', startTime)[0].get_xyz()

    r_EarthEM_ct = -icrs2gmec(r_EM_ct, currentTime)
    r_EarthEM_st = -icrs2gmec(r_EM_st, startTime)

    norm_ct = np.linalg.norm(r_EarthEM_ct)
    norm_st = np.linalg.norm(r_EarthEM_st)

    n_vec = np.cross(r_EarthEM_ct, r_EarthEM_st.T)
    
    dt = (currentTime - startTime).value
    t_mod = np.mod(dt, 27.321582)
    if t_mod < 27.321582/2:
        sign2 = 1
    elif t_mod > 27.321582/2 and t_mod < 27.321582:
        sign2 = -1
    r_sin = (np.linalg.norm(n_vec)/(norm_ct*norm_st))
    r_cos = (np.dot(r_EarthEM_ct/norm_ct, r_EarthEM_st.T/norm_st))
    theta = np.arctan2(sign2*r_sin, r_cos)

    return theta


def rotMatAxisAng(n_hat, theta):
    """Computes a rotation matrix given an axis of rotation and an angle of rotation

    Args:
        n_hat (float n array)
            A unit vector specifying the axis of rotation (3D)
        theta (float)
            Angle of rotation in radians

    Returns:
        R (float n array)
            A 3x3 rotation matrix

    """

    r_skew = np.array([[0, -n_hat[2], n_hat[1]],
                       [n_hat[2], 0, -n_hat[0]],
                       [-n_hat[1], n_hat[0], 0]])

    R = np.identity(3) + r_skew * np.sin(theta) + r_skew @ r_skew * (1 - np.cos(theta))

    return R
    

def inert2geo(startTime, equinox):
    """Computes the DCM to go from the inertial Earth-Moon CRTBP perifocal frame (I frame) to the GeocentricMeanEcliptic
     frame centered at the Earth-Moon barycenter (G frame)

    Args:
        startTime (astropy Time array):
            Mission start time in MJD
        equinox (astropy Time array):
            Reference frame equinox time in MJD

    Returns:
        C_I2G (float n array):
            3x3 Array for the directional cosine matrix

    """

    tarray = startTime + np.arange(28)*u.d
    r_moon = get_body_barycentric_posvel('Moon', tarray)[0].get_xyz()
    r_bary = get_body_barycentric_posvel('Earth-Moon-Barycenter', tarray)[0].get_xyz()

    ctr = 0
    r_m = np.zeros([len(tarray), 3])
    for ii in tarray:
        tmp1 = icrs2gmec(r_moon[:, ctr], ii).to('AU').value
        tmp2 = icrs2gmec(r_bary[:, ctr], ii).to('AU').value
        r_m[ctr, :] = tmp1 - tmp2
        ctr = ctr + 1
    
    ZZ = r_m[:, 2]
    signZ = np.sign(ZZ)
    diffZ = np.diff(signZ)
    indZ = np.argwhere(2 == diffZ)[0][0]
    
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
    r_moon1 = icrs2gmec(r_moon1, t1)
    r_moon2 = icrs2gmec(r_moon2, t2)
    r_moon3 = icrs2gmec(r_moon3, t3)
    r_bary1 = icrs2gmec(r_bary1, t1)
    r_bary2 = icrs2gmec(r_bary2, t2)
    r_bary3 = icrs2gmec(r_bary3, t3)

    r_m1 = r_moon1 - r_bary1
    r_m2 = r_moon2 - r_bary2
    r_m3 = r_moon3 - r_bary3

    error = r_m3[2]

    while np.abs(error.value) > 1E-8:
        print(np.abs(error.value))
        sign1 = np.sign(r_m1[2])
        sign2 = np.sign(r_m2[2])
        sign3 = np.sign(r_m3[2])

        if sign3 == sign1:
            t1 = t3
            r_m1 = r_m3
            
            dt = (t2 - t1)/2
            t3 = t3 + dt

            if sign1 == sign2:
                # if here something went wrong
                print('if')
                breakpoint()
            
        elif sign3 == sign2:
            t2 = t3
            r_m2 = r_m3
            
            dt = (t2 - t1)/2
            t3 = t1 + dt

            if sign1 == sign2:
                # if here something went wrong
                print('elif')
                breakpoint()
            
        else:
            # if here something went wrong
            breakpoint()
            
        r_m = (get_body_barycentric_posvel('Moon', t3)[0].get_xyz()).to('AU')
        r_b = (get_body_barycentric_posvel('Earth-Moon-Barycenter', t3)[0].get_xyz()).to('AU')
        r_m = icrs2gmec(r_m, t3)
        r_b = icrs2gmec(r_b, t3)
        r_m3 = r_m - r_b
            
        error = r_m3[2]
        
    t_LAAN = t3
    t_LAAN2 = t_LAAN + 27.321582/4*u.d
    
    moon_LAAN1 = get_body_barycentric_posvel('Moon', t_LAAN)[0].get_xyz()
    moon_LAAN2 = get_body_barycentric_posvel('Moon', t_LAAN2)[0].get_xyz()
    bary_LAAN1 = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_LAAN)[0].get_xyz()
    bary_LAAN2 = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_LAAN2)[0].get_xyz()
    
    m_LAAN1 = icrs2gmec(moon_LAAN1, t_LAAN)
    m_LAAN2 = icrs2gmec(moon_LAAN2, t_LAAN2)
    b_LAAN1 = icrs2gmec(bary_LAAN1, t_LAAN)
    b_LAAN2 = icrs2gmec(bary_LAAN2, t_LAAN2)
    
    r_LAAN1 = m_LAAN1 - b_LAAN1
    r_LAAN2 = m_LAAN2 - b_LAAN2
    
    r_LAAN3 = np.cross(r_LAAN1, r_LAAN2)
    n_LAAN = r_LAAN3/np.linalg.norm(r_LAAN3)
    
    theta_LAAN = equinoxAngle(r_LAAN1, t_LAAN, equinox)
    
    C_LAAN = rotMatAxisAng(n_LAAN, theta_LAAN)
    
    # find INC DCM
    tarray_r = startTime + np.arange(28)/1*u.d
    r_moons_r = get_body_barycentric_posvel('Moon', tarray)[0].get_xyz()
    r_barys_r = get_body_barycentric_posvel('Earth-Moon-Barycenter', tarray)[0].get_xyz()

    ctr = 0
    r_m_r = np.zeros([len(tarray_r), 3])

    for ii in tarray_r:
        tmp1 = C_LAAN @ icrs2gmec(r_moons_r[:, ctr], ii).to('AU').value
        tmp2 = C_LAAN @ icrs2gmec(r_barys_r[:, ctr], ii).to('AU').value
        r_m_r[ctr,:] = tmp1 - tmp2
        ctr = ctr + 1

    XX = max(r_m_r[:,0]) - min(r_m_r[:, 0])
    YY = max(r_m_r[:,1]) - min(r_m_r[:, 1])
    ZZ = max(r_m_r[:,2]) - min(r_m_r[:, 2])

    theta_INC = -np.deg2rad(5.145) #np.arctan2(ZZ,YY)

    n_INC = r_LAAN1/np.linalg.norm(r_LAAN1)

    C_INC = rotMatAxisAng(n_INC, theta_INC)

    # find AOP DCM
    # rough search
    r_norm_r = np.linalg.norm(r_m_r, axis=1)
    r_min_r = min(r_norm_r)
    
    r_ind_r = np.argwhere(r_min_r == r_norm_r)[0][0]

    # fine search
    t_AOP_r = tarray_r[r_ind_r-1]
    tarray_f = t_AOP_r + 0.5*u.d + np.arange(1600)/800*u.d
    
    r_moons_f = get_body_barycentric_posvel('Moon', tarray_f)[0].get_xyz()
    r_barys_f = get_body_barycentric_posvel('Earth-Moon-Barycenter', tarray_f)[0].get_xyz()

    ctr = 0
    r_m_f = np.zeros([len(tarray_f), 3])

    for ii in tarray_f:
        tmp1 = C_LAAN @ icrs2gmec(r_moons_f[:, ctr], ii).to('AU').value
        tmp2 = C_LAAN @ icrs2gmec(r_barys_f[:, ctr], ii).to('AU').value
        r_m_f[ctr,:] = tmp1 - tmp2
        ctr = ctr + 1
        
    r_norm_f = np.linalg.norm(r_m_f, axis=1)
    r_min_f = min(r_norm_f)
    
    r_ind_f = np.argwhere(r_min_f == r_norm_f)[0][0]
    t_AOP = tarray_f[r_ind_f]
    
    theta_AOP = rotAngle(t_AOP, t_LAAN).value
    
    n_AOP = C_INC @ C_LAAN @ n_LAAN
    
    C_AOP = rotMatAxisAng(n_AOP, theta_AOP)

    C_G2I = C_AOP @ C_INC @ C_LAAN
    C_I2G = C_G2I.T

    return C_I2G
    
    
def inert2rot(currentTime, startTime):
    """Compute the directional cosine matrix to go from the Earth-Moon CR3BP
    perifocal frame (I) to the Earth-Moon CR3BP rotating frame (R)
    
    Args:
        currentTime (astropy Time array):
            Current mission time in MJD
                startTime (astropy Time array):
            Mission start time in MJD

    Returns:
        C_I2R (float n array):
            3x3 Array for the directional cosine matrix
    """

    dt = currentTime.value - startTime.value
    theta = unitConversion.convertTime_to_canonical(dt*u.d)
    
    C_I2R = rot(theta, 3)
    
    return C_I2R


def icrs2rot(pos, currentTime, startTime, mu_star, C_G2B):
    """Convert position vector in ICRS coordinate frame to rotating coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        startTime (astropy Time array):
            Mission start time in MJD
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        r_rot (astropy Quantity array):
            Position vector in rotating frame in km
    """
    
    state_EM = get_body_barycentric_posvel('Earth-Moon-Barycenter', startTime)
    r_EMG_icrs = state_EM[0].get_xyz().to('AU')

    # pos = pos * u.AU
    r_PE_gmec = icrs2gmec(pos, startTime)
    r_rot = r_PE_gmec
    r_EME_gmec = icrs2gmec(r_EMG_icrs, startTime)
    r_PEM = r_PE_gmec - r_EME_gmec

    C_I2R = inert2rot(currentTime, startTime)
    
    r_rot = C_G2B@C_I2R@r_PEM

    return r_rot


def icrs2gmec(pos, currentTime, vel=None):
    """Convert position and velocity vectors in ICRS coordinate frame to Geocentric Mean Ecliptic coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        vel (astropy Quantity array, optional):
            Velocity vector in ICRS (heliocentric) frame in arbitrary distance units

    Returns:
        pos_gmec (astropy Quantity array):
            Position vector in Geocentric Mean Ecliptic frame in km
        vel_gmec (astropy Quantity array, optional):
            Velocity vector in Geocentric Mean Ecliptic frame in km/s
    """

    if vel is not None:
        pos = pos.to('km')
        vel = vel.to('km/s')
        state_icrs = coord.SkyCoord(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2],
                                    representation_type='cartesian', frame='icrs', obstime=currentTime)
        state_gmec = state_icrs.transform_to(GeocentricMeanEcliptic(obstime=currentTime))
        pos_gmec = state_gmec.cartesian.get_xyz()
        vel_gmec = state_gmec.velocity.get_d_xyz()
        return pos_gmec, vel_gmec
    else:
        pos = pos.to('km')
        pos_icrs = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km',
                                  representation_type='cartesian', frame='icrs', obstime=currentTime)
        pos_gmec = pos_icrs.transform_to(GeocentricMeanEcliptic(obstime=currentTime)).cartesian.get_xyz()
        return pos_gmec


def gmec2icrs(pos, currentTime, vel=None):
    """Convert position and velocity vectors in Geocentric Mean Ecliptic coordinate frame to ICRS coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in Geocentric Mean Ecliptic frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        vel (astropy Quantity array, optional):
            Velocity vector in Geocentric Mean Ecliptic frame in arbitrary distance units

    Returns:
        pos_icrs (astropy Quantity array):
            Position vector in ICRS frame in km
        vel_icrs (astropy Quantity array, optional):
            Velocity vector in ICRS frame in km/s
    """

    if vel is not None:
        pos = pos.to('km')
        vel = vel.to('km/s')
        state_gmec = coord.SkyCoord(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2],
                                    representation_type='cartesian', frame='geocentricmeanecliptic',
                                    obstime=currentTime)
        state_icrs = state_gmec.transform_to(ICRS())
        pos_icrs = state_icrs.cartesian.get_xyz()
        vel_icrs = state_icrs.velocity.get_d_xyz()
        return pos_icrs, vel_icrs
    else:
        pos = pos.to('km')
        pos_gmec = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km',
                                  representation_type='cartesian', frame='geocentricmeanecliptic', obstime=currentTime)
        pos_icrs = pos_gmec.transform_to(ICRS()).cartesian.get_xyz()
        return pos_icrs


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


def convertIC_R2H(pos_R, vel_R, t_mjd, Tp_can=None):
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
    C_I2G = inert2geo(t_mjd, t_mjd)
    pos_G = C_I2G @ pos_I

    state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_mjd)
    posEMB = state_EMB[0].get_xyz().to('AU')
    velEMB = state_EMB[1].get_xyz().to('AU/day')

    posEMB_E = (icrs2gmec(posEMB, t_mjd)).to('AU')

    pos_GCRS = pos_G + posEMB_E  # G frame

    pos_H = (gmec2icrs(pos_GCRS, t_mjd)).to('AU')

    vel_I = rot2inertV(np.array(pos_R), np.array(vel_R), 0)
    v_dim = unitConversion.convertVel_to_dim(vel_I).to('AU/day')
    vel_H = velEMB + v_dim

    if Tp_can is not None:
        Tp_dim = unitConversion.convertTime_to_dim(Tp_can).to('day')
        return pos_H, vel_H, Tp_dim
    else:
        return pos_H, vel_H


def convertIC_I2H(pos_I, vel_I, currentTime, mu_star, C_I2G, Tp_can=None):
    """Converts initial conditions from the I frame to the H frame

    Args:
        pos_I (float n array):
            Array of distance in canonical units
        vel_I (float n array):
            Array of velocities in canonical units
        currentTime (float)
            Current mission time
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

    pos_G = C_I2G @ pos_I

    state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)
    posEMB = state_EMB[0].get_xyz().to('AU')
    velEMB = state_EMB[1].get_xyz().to('AU/day')

    stateEMB_E = icrs2gmec(posEMB, currentTime, velEMB)
    posEMB_E = stateEMB_E[0].to('AU')
    velEMB_E = stateEMB_E[1].to('AU/d')

    pos_GMECL = pos_G + posEMB_E  # G frame

    v_dim = unitConversion.convertVel_to_dim(vel_I).to('AU/day')
    vel_G = C_I2G @ v_dim
    vel_GMECL = vel_G + velEMB_E

    pos_H, vel_H = gmec2icrs(pos_GCRS, currentTime, vel_GMECL)
    pos_H = pos_H.to('AU')
    vel_H = vel_H.to('AU/d')
    
    tmp_G = icrs2gcrs(pos_H, currentTime)
    tmp_GMECL = tmp_G - posEMB_E
    tmp_I = C_B2G.T @ tmp_GMECL
    
    tmpM_H = get_body_barycentric_posvel('Moon', currentTime)[0].get_xyz()
    tmpM_G = icrs2gcrs(tmpM_H, currentTime).to('AU')
    tmpM_GMECL = tmpM_G - posEMB_E
    tmpM_I = C_B2G.T @ tmpM_GMECL
    
#    breakpoint()

    if Tp_can is not None:
        Tp_dim = unitConversion.convertTime_to_dim(Tp_can).to('day')
        return pos_H, vel_H, Tp_dim
    else:
        return pos_H, vel_H
