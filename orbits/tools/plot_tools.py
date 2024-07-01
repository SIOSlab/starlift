import numpy as np
from matplotlib import pyplot as plt
import constants as c
import frameConversion
import unitConversion
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import astropy.units as u


def Orbit3D(solvec, time, args={}):
    """Plot the orbit in three dimensions. Default origin is the EM barycenter in the EM synodic reference frame. The dimensioned argument is also supplied to properly organize the display of the Earth, Moon, and axis scaling. 
    args={'Frame': 'Synodic', 'dimensioned':False}"""

    _args = {'Frame': 'Synodic', 'dimensioned':True}
    for key in args.keys():
        _args[ key ] = args[ key ]

    x_vals = np.array(solvec[:,0])
    y_vals = np.array(solvec[:,1])
    z_vals = np.array(solvec[:,2])

    ax = plt.axes(projection='3d')
    traj = ax.scatter(x_vals,y_vals,z_vals, c=time, cmap = 'plasma', s=.5)
    ax.scatter(0,0,0, c='m', marker='*')
    
    n = np.linspace(0,2*np.pi,100)
    v = np.linspace(0, np.pi, 100)

    if _args['dimensioned'] == False:
        re = c.earthR / c.lstar
        rm = c.moonR / c.lstar
        eoffset = -c.mustar
        moffset = 1-c.mustar
    else:
        re = c.earthR
        rm = c.moonR
        eoffset = -c.mustar*c.moonSMA
        moffset = (1-c.mustar)*c.moonSMA

    xe = re * np.outer(np.cos(n), np.sin(v)) + eoffset
    ye = re * np.outer(np.sin(n), np.sin(v))
    ze = re * np.outer(np.ones(np.size(n)), np.cos(v))

    xm = rm * np.outer(np.cos(n), np.sin(v)) + moffset
    ym = rm * np.outer(np.sin(n), np.sin(v))
    zm = rm * np.outer(np.ones(np.size(n)), np.cos(v))

    ax.plot_surface(xe,ye,ze)
    ax.plot_surface(xm,ym,zm)
    plt.title('Orbit in the Earth-Moon Rotating Frame')

    plt.axis('equal')
    ax.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(traj)
    plt.show()


def PlotManifold(solvec, time, mu, ax, title,eigval, eigvec):
    # _args = {'Frame': 'Synodic'}
    x_vals = np.array(solvec[0,:])
    y_vals = np.array(solvec[1,:])
    z_vals = np.array(solvec[2,:])
    
    # ax = plt.axes(projection='3d')
    traj = ax.scatter(x_vals,y_vals,z_vals, c=time, cmap = 'plasma',s=.5)
    ax.scatter(0,0,0, c='m', marker='*')

    n = np.linspace(0,2*np.pi,100)
    v = np.linspace(0, np.pi, 100)

    re = c.earthD / c.lstar
    rm = c.moonD / c.lstar

    xe = re * np.outer(np.cos(n), np.sin(v)) - mu
    ye = re * np.outer(np.sin(n), np.sin(v)) + 0
    ze = re * np.outer(np.ones(np.size(n)), np.cos(v)) + 0

    xm = rm * np.outer(np.cos(n), np.sin(v)) + (1-mu)
    ym = rm * np.outer(np.sin(n), np.sin(v))
    zm = rm * np.outer(np.ones(np.size(n)), np.cos(v))

    ax.plot_surface(xe,ye,ze)
    ax.plot_surface(xm,ym,zm)
    plt.suptitle(title)

    ax.set_title(("Eigenvalue: ", eigval ))
    plt.axis('equal')
    # ax.text2D(0.05, 0.95, (r'Eigenvalue: ', eigval, r'\nEigvec: ', eigvec), transform=ax.transAxes)
    ax.legend()
    plt.xlabel('X\n')
    plt.ylabel('Y\n')
    # plt.colorbar(traj)


def plotConvertBodies(timesFF, posFF, t_mjd, frame, C_G2B):
    # ** Add documentation



    # preallocate space
    r_PEM_r = np.zeros([len(timesFF), 3])
    r_SunEM_r = np.zeros([len(timesFF), 3])
    r_EarthEM_r = np.zeros([len(timesFF), 3])
    r_MoonEM_r = np.zeros([len(timesFF), 3])

    # sim time in mjd
    timesFF_mjd = timesFF + t_mjd
    for ii in np.arange(len(timesFF)):
        time = timesFF_mjd[ii]

        if frame == 0:
            # positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
            r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU')
            r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU')
            r_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time).get_xyz().to('AU')
        
            # convert from H frame to GCRS frame
            r_PG = frameConversion.icrs2gcrs(posFF[ii]*u.AU, time)
            r_EMG = frameConversion.icrs2gcrs(r_EMO, time)
            r_SunG = frameConversion.icrs2gcrs(r_SunO, time)
            r_MoonG = frameConversion.icrs2gcrs(r_MoonO, time)
            
            # change the origin to the EM barycenter, G frame
            r_PEM = r_PG - r_EMG
            r_SunEM = r_SunG - r_EMG
            r_EarthEM = -r_EMG
            r_MoonEM = r_MoonG - r_EMG
            
            r_PEM_r[ii, :] = r_PEM
            r_SunEM_r[ii, :] = r_SunEM
            r_EarthEM_r[ii, :] = r_EarthEM
            r_MoonEM_r[ii, :] = r_MoonEM
            
        elif frame >= 1:
            # positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
            r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU')
            r_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)[0].get_xyz().to('AU')
            
            # convert from H frame to GCRS frame
            r_PG = frameConversion.icrs2gcrs(posFF[ii]*u.AU, time)
            r_EMG = frameConversion.icrs2gcrs(r_EMO, time)
            r_MoonG = frameConversion.icrs2gcrs(r_MoonO, time)
            
            # change the origin to the EM barycenter, G frame
            r_PEM = r_PG - r_EMG
            r_EarthEM = -r_EMG
            r_MoonEM = r_MoonG - r_EMG
            
            # convert from G frame to I frame
            r_PEM = C_G2B@r_PEM.to('AU')
            r_EarthEM = C_G2B@r_EarthEM.to('AU')
            r_MoonEM = C_G2B@r_MoonEM.to('AU')
            
            if frame == 2:
                C_I2R = frameConversion.body2rot(time, t_mjd)
                r_PEM = C_I2R@r_PEM.to('AU')
                r_EarthEM = C_I2R@r_EarthEM.to('AU')
                r_MoonEM = C_I2R@r_MoonEM.to('AU')

            r_PEM_r[ii, :] = r_PEM.value
            r_EarthEM_r[ii, :] = r_EarthEM.value
            r_MoonEM_r[ii, :] = r_MoonEM.value

    return r_PEM_r, r_SunEM_r, r_EarthEM_r, r_MoonEM_r

        
def plotBodiesFF(timesFF,posFF,t_mjd,frame):
    # ** Add documentation

    r_PEM, r_SunEM, r_EarthEM, r_MoonEM = plotConvertBodies(timesFF,posFF,t_mjd,frame)
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(r_EarthEM[:, 0], r_EarthEM[:, 1], r_EarthEM[:, 2], 'g', label='Earth')
    ax.plot(r_MoonEM[:, 0], r_MoonEM[:, 1], r_MoonEM[:, 2], 'r', label='Moon')
    ax.plot(r_PEM[:, 0], r_PEM[:, 1], r_PEM[:, 2], 'b', label='Full Force')
    if frame == 0:
        ax.plot(r_SunEM[:, 0], r_SunEM[:, 1], r_SunEM[:, 2], 'y', label='Sun')
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    plt.legend()
    
    # ** Add lines to resize figure and automatically save png and svg
    return
    
def plotCompare_rot(timesFF, posFF, t_mjd, frame, timesCRTBP, posCRTBP, C_G2B):
    # ** Add documentation
    
    r_PEM, _, r_EarthEM, r_MoonEM = plotConvertBodies(timesFF, posFF, t_mjd, frame, C_G2B)

    posCRTBP = (unitConversion.convertPos_to_dim(posCRTBP).to('AU')).value
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(posCRTBP[:, 0], posCRTBP[:, 1], posCRTBP[:, 2], 'k', label='CRTBP')
    ax.plot(r_PEM[:, 0], r_PEM[:, 1],r_PEM[:, 2], 'b', label='Full Force')
    ax.plot(r_EarthEM[:, 0], r_EarthEM[:, 1], r_EarthEM[:, 2], 'g', label='Earth')
    ax.plot(r_MoonEM[:, 0], r_MoonEM[:, 1], r_MoonEM[:, 2], 'r', label='Moon')
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    plt.legend()
    breakpoint()
    # ** Add lines to resize figure and automatically save png and svg
    return


def plotCompare_inert(timesCRTBP, posCRTBP, t_mjd, timesFF, posFF,mu_star):
    # Fix this. Determine which inertial frame this should be
    # ** Add documentation
    
    times = timesCRTBP + t_mjd
    pos_dim = unitConversion.convertPos_to_dim(posCRTBP).to('AU')
    
    C_B2G = frameConversion.body2geo(t_mjd, t_mjd, mu_star)
    
    pos_H = np.zeros([len(times),3])
    for ii in np.arange(len(timesCRTBP)):
        currentTime = times[ii]
        pos_I = frameConversion.inert2rotP(pos_dim[ii],currentTime,t_mjd)
        
        pos_G = C_B2G@pos_I
        
        state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_mjd)
        posEMB = state_EMB[0].get_xyz().to('AU')
        velEMB = state_EMB[1].get_xyz().to('AU/day')
        posE = get_body_barycentric_posvel('Earth', t_mjd)[0].get_xyz().to('AU')
        posE_EMB = posE - posEMB

        pos_GCRS = pos_G - posE_EMB
        
        pos_H[ii,:] = (frameConversion.gcrs2icrs(pos_GCRS, t_mjd)).to('AU')

    r_EarthO = get_body_barycentric_posvel('Earth', times)[0].get_xyz().to('AU')
    r_MoonO = get_body_barycentric_posvel('Moon', times)[0].get_xyz().to('AU')
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pos_H[:, 0], pos_H[:, 1], pos_H[:, 2], 'k', label='CRTBP')
    ax.plot(posFF[:, 0], posFF[:, 1],posFF[:, 2], 'b', label='Full Force')
#    ax.plot(r_EarthO[:, 0], r_EarthO[:, 1], r_EarthO[:, 2], 'g', label='Earth')
#    ax.plot(r_MoonO[:, 0], r_MoonO[:, 1], r_MoonO[:, 2], 'r', label='Moon')
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    plt.legend()
    breakpoint()
    # ** Add lines to resize figure and automatically save png and svg
    return


def set_axes_equal(ax):
    # Make axes of 3D plot have equal scale so that spheres appear as spheres, cubes as cubes, etc.
    # Input ax: a matplotlib axis, e.g., as output from plt.gca().

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
