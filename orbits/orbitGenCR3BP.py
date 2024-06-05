import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
#import rebound
import matplotlib.pyplot as plt

sys.path.insert(1, 'tools')
import unitConversion
import frameConversion

import pdb


def CRTBP_EOM(t,w,mu_star):
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
    
    s_T = stateProp(freeVar,mu_star)
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

def calcFx(freeVar):
    """Applies constraints to the free variables

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period


    Returns:
        ~numpy.ndarray(float):
            constraint array

    """
    s_T = stateProp(freeVar,mu_star)
    state = s_T[-1]

    Fx = np.array([state[1], state[3], state[5]])
    return Fx

def stateProp(freeVar,mu_star):
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

    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,),rtol=1E-12,atol=1E-12,)
    states = sol_int.y.T
    times = sol_int.t
    
    return states, times
    
def fsolve_eqns(w,z,solp):
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
    Fx = calcFx(w)
    zeq = z.T@(w-solp)
    sys_w = np.append(Fx,zeq)

    return sys_w

mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]
X = [IC[0], IC[2], IC[4], IC[6]]

eps = 1E-3
N = 5
solutions = np.zeros([N,4])
z = np.array([0, 0, 0, 1])
step = 1E-2

for ii in np.arange(N):
    error = 10
    ctr = 0
    while error > eps and ctr < 1000:
        Fx = calcFx(X)

        error_new = np.linalg.norm(Fx)

        dFx = calcdFx(X,mu_star,m1,m2)

        X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
        
        ctr = ctr + 1
    
    IV = np.array([X[0], X[1], X[2], 2*X[3]])
    solutions[ii] = IV
    states = stateProp(IV,mu_star)
    
    ax.plot(states[:,0],states[:,1],states[:,2])
    
    solp = X + z*step
    ss = fsolve(fsolve_eqns,X,args=(z,solp),full_output=True,xtol=1E-12)
    X = ss[0]
    J = ss[1]['fjac']
    
    z = np.linalg.inv(J.T)@zT
    z = z/np.linalg.norm(z)
    
#breakpoint()
