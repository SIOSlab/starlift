import numpy as np
from matplotlib import pyplot as plt
import tools.constants as c

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