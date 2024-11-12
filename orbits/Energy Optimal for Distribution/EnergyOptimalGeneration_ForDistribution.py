import scipy.io
import numpy as np
import numpy.linalg as la
from sympy import *
import os
import os.path
from scipy.linalg import lu_factor, lu_solve, eigh
from STMint import STMint
import matplotlib
import matplotlib.pyplot as plt
import pdb
import dill #use dill to save lambda-fied class
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from shapely.geometry import Point, Polygon
import itertools
        
#symbolically define the dynamics for energy optimal control in the cr3bp
#this will be used by the STMInt package for numerically integrating the STM and STT
def optControlDynamics():
    x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En=symbols("x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En")
    #mu = 3.00348e-6
    mu = 0.012150590000000
    mu1 = 1. - mu
    mu2 = mu
    r1 = sqrt((x + mu2)**2 + (y**2) + (z**2))
    r2 = sqrt((x - mu1)**2 + (y**2) + (z**2))
    U = (-1/2)*((x**2) + (y**2)) - (mu1/r1) - (mu2/r2)
    dUdx = diff(U,x)
    dUdy = diff(U,y)
    dUdz = diff(U,z)

    RHS = Matrix([vx,vy,vz,((-1*dUdx) + 2*vy),((-1*dUdy)- 2*vx),(-1*dUdz)])

    variables = Matrix([x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En])

    dynamics = Matrix(BlockMatrix([[RHS - Matrix([0,0,0,lvx,lvy,lvz])], 
        [-1.*RHS.jacobian(Matrix([x,y,z,vx,vy,vz]).transpose())*Matrix([lx,ly,lz,lvx,lvy,lvz])],
        [Matrix([lvx**2+lvy**2+lvz**2])/2]]))
    #return Matrix([x,y,z,vx,vy,vz]), RHS
    return variables, dynamics 




# This function iterates on the initial conditions of the state to reduce the reference orbit error
def state_iterate(target_state, tolerance, stepsize, T_final):
    # load in threeBodyInt class if it doesn't exist
    FileName_threeBodyInt = "./EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle"
    if not os.path.isfile(FileName_threeBodyInt):
        variables, dynamics = optControlDynamics()
        threeBodyInt = STMint(variables, dynamics, variational_order=1) #this is the expensive line

        with open('EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle', 'wb') as threeBodyInt_file:
            dill.dump(threeBodyInt, threeBodyInt_file)

    with open('EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle', 'rb') as threeBodyInt_file:
        threeBodyInt = dill.load(threeBodyInt_file)

    state = target_state
    costates = np.array([[0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000]])
    i = 0
    residual = target_state
    STM = np.random.rand(13,13)

    # begin Newton-Raphson iteration
    while la.norm(la.inv(STM[:6, :6] - np.identity(6)) @ residual) > tolerance and i < 20:
        [state_output, STM] = threeBodyInt.dynVar_int([0,T_final], np.reshape(np.vstack((state,costates,np.array([0]))), (1,13)), output='final', max_step=stepsize, rtol=1e-12, atol=1e-12)
        residual = np.array(state_output[:6],ndmin=2).T - state
        state -= la.inv(STM[:6, :6] - np.identity(6)) @ residual 
        i += 1
        print(str(i))
        print(str(la.norm(la.inv(STM[:6, :6] - np.identity(6)) @ residual)) )

    return state_output




# This function computes the cost of the orbit and returns the costates for continuation purposes
def true_cost(target_state, u_inherent, tolerance, stepsize, T_final, costate_guess):
    # load in threeBodyInt class if it doesn't exist
    FileName_threeBodyInt = "./EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle"
    if not os.path.isfile(FileName_threeBodyInt):
        variables, dynamics = optControlDynamics()
        threeBodyInt = STMint(variables, dynamics, variational_order=1) #this is the expensive line

        with open('EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle', 'wb') as threeBodyInt_file:
            dill.dump(threeBodyInt, threeBodyInt_file)

    with open('EnergyOptimal_threeBodyInt_EarthMoon_L2nrho.pickle', 'rb') as threeBodyInt_file:
        threeBodyInt = dill.load(threeBodyInt_file)

    # set initial guesses for iteration
    costates_iterate = costate_guess
    i = 0
    residual = target_state
    STM = np.random.rand(13,13)

    # begin Newton-Raphson iteration
    while la.norm(la.inv(STM[:6, 6:12]) @ residual) > tolerance and i < 20:
        [state, STM] = threeBodyInt.dynVar_int([0,T_final], np.reshape(np.vstack((target_state,costates_iterate,np.array([0]))), (1,13)), output='final', max_step=stepsize)
        residual = np.array(state[:6],ndmin=2).T - target_state
        costates_iterate -= la.inv(STM[:6, 6:12]) @ residual 
        i += 1
        print(str(i))
        print(str(la.norm(la.inv(STM[:6, 6:12]) @ residual)) )

    # find the true cost
    J_computed = state[12]

    return J_computed, costates_iterate










##############################
###### LINEAR ANALYSIS #######
##############################

#For Earth-Moon, these are the conversions:
# 1 TU = 2.360584684800000E+06/(2*np.pi) seconds
# 1 DU = 384400000.0000000 meters

# Set initial conditions 
mu = 0.012150590000000
ics = [1.06315768e+00,  3.26952322e-04, -2.00259761e-01, 3.61619362e-04, -1.76727245e-01, -7.39327422e-04, 0, 0, 0, 0, 0, 0, 0] #(use this for _L2nrho)
T_final = 2.085034838884136
exponent = 8
t_step = T_final/2.**exponent

# Set file name to save data on first run
FileName_state = "./EnergyOptimal_state_EarthMoon_L2nrho_prime.mat"
FileName_STM = "./EnergyOptimal_STM_EarthMoon_L2nrho_prime.mat"
FileName_STT = "./EnergyOptimal_STT_EarthMoon_L2nrho_prime.mat"
FileName_time = "./EnergyOptimal_time_EarthMoon_L2nrho_prime.mat"

# Run if the file does not exist
if not os.path.isfile(FileName_time):
    variables, dynamics = optControlDynamics()
    threeBodyInt = STMint(variables, dynamics, variational_order=2)

    [state, STM, STT, time] = threeBodyInt.dynVar_int2([0,T_final], ics, output='all', max_step=t_step)

    scipy.io.savemat(FileName_state, {"state": state})
    scipy.io.savemat(FileName_STM, {"STM": STM})
    scipy.io.savemat(FileName_STT, {"STT": STT})
    scipy.io.savemat(FileName_time, {"time": time})
    
# load data
state_full = list(scipy.io.loadmat(FileName_state).values())[-1]
STM_full = list(scipy.io.loadmat(FileName_STM).values())[-1] # state transition matrix at every time
STT_full = list(scipy.io.loadmat(FileName_STT).values())[-1]
time_full = list(scipy.io.loadmat(FileName_time).values())[-1]



##### ACQUIRE STMS AND STTS FOR PERIOD STARTING AT SECOND HALF OF ORBIT #####
# Set file name to save data on first run
ics_halfway = [0.988309034135378,  -0.011056863643638, 0.029955671878931, -0.020221296056888, 0.816370643713805, 0.160733352222188, 0, 0, 0, 0, 0, 0, 0]

FileName_state = "./EnergyOptimal_state_EarthMoon_L2nrho_prime_HALFWAY.mat"
FileName_STM = "./EnergyOptimal_STM_EarthMoon_L2nrho_prime_HALFWAY.mat"
FileName_STT = "./EnergyOptimal_STT_EarthMoon_L2nrho_prime_HALFWAY.mat"
FileName_time = "./EnergyOptimal_time_EarthMoon_L2nrho_prime_HALFWAY.mat"

# Run if the file does not exist
if not os.path.isfile(FileName_state):
    variables, dynamics = optControlDynamics()
    threeBodyInt = STMint(variables, dynamics, variational_order=2)
 
    [state, STM, STT, time] = threeBodyInt.dynVar_int2([0,T_final], ics_halfway, output='all', max_step=t_step)

    scipy.io.savemat(FileName_state, {"state": state})
    scipy.io.savemat(FileName_STM, {"STM": STM})
    scipy.io.savemat(FileName_STT, {"STT": STT})
    scipy.io.savemat(FileName_time, {"time": time})
    
# load data
state_full_halfway = list(scipy.io.loadmat(FileName_state).values())[-1]
STM_full_halfway = list(scipy.io.loadmat(FileName_STM).values())[-1] # state transition matrix at every time
STT_full_halfway = list(scipy.io.loadmat(FileName_STT).values())[-1]
time_full_halfway = list(scipy.io.loadmat(FileName_time).values())[-1]

##### BACK TO NORMAL STUFF #####
# state_full = state_full_halfway
# STM_full = STM_full_halfway
# STT_full = STT_full_halfway
# time_full = time_full_halfway

J_max = 3.514110698664422E-04 # set any reasonable value for J_max but make sure it can be validated
[rows,columns,depth] = STM_full.shape
C = np.array([[1, 0],[0, 1],[0, 0],[0, 0],[0, 0],[0, 0]]) 
A = np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0]])
Ayz = np.array([[0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0]])

fig, ax = plt.subplots()
ax = plt.gca()
xmin = 0.95
xmax = 1.15
ymin = -0.1
ymax = 0.1
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax.set_aspect('equal', adjustable='box')

# ax = plt.gca()
# xmin = -0.1
# xmax = 0.1
# ymin = -0.23
# ymax = 0.07
# ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

start = 10
tipsright = np.zeros((2,rows))
tipsleft = np.zeros((2,rows))
pos = np.zeros((2,rows))
xypos = np.transpose(state_full[:,0:2])
yzpos = np.transpose(state_full[:,1:3])

tipsright_halfway = np.zeros((2,rows))
tipsleft_halfway = np.zeros((2,rows))
pos_halfway = np.zeros((2,rows))
xypos_halfway = np.transpose(state_full_halfway[:,0:2])
yzpos_halfway = np.transpose(state_full_halfway[:,1:3])


for i in range(0,rows):

    STM = STM_full[i,:,:] @ STM_full[-1,:,:] @ la.inv(STM_full[i,:,:])
    STT = STT_full[i,:,:,:]
    STT1 = STT_full[i,:,:,:]
    STT2 = STT_full[-1,:,:,:]

    STM_halfway = STM_full_halfway[i,:,:] @ STM_full_halfway[-1,:,:] @ la.inv(STM_full_halfway[i,:,:])
    STT_halfway = STT_full_halfway[i,:,:,:]
    STT1_halfway = STT_full_halfway[i,:,:,:]
    STT2_halfway = STT_full_halfway[-1,:,:,:]

    Matrix1 = np.block([[np.identity(6), np.zeros((6,6))],
    [-la.solve(STM[:6, 6:12], STM[:6, :6]), la.inv(STM[:6, 6:12])]])

    Matrix1_halfway = np.block([[np.identity(6), np.zeros((6,6))],
    [-la.solve(STM_halfway[:6, 6:12], STM_halfway[:6, :6]), la.inv(STM_halfway[:6, 6:12])]])

    # work with 13 x  13 matrices:
    # TempMatrix1 = la.inv(np.transpose(STM_full[i,:,:])) @ (STT_full[-1,12,:,:] - STT_full[i,12,:,:]) @ la.inv(STM_full[i,:,:])
    # TempMatrix2 = np.transpose(STM_full[-1,:,:]@la.inv(STM_full[i,:,:]))@STT_full[i,12,:,:]@(STM_full[-1,:,:]@la.inv(STM_full[i,:,:]))+TempMatrix1

    # work with 12 x 12 matrices:
    TempMatrix1 = la.inv(np.transpose(STM_full[i,:12,:12])) @ (STT_full[-1,12,:12,:12] - STT_full[i,12,:12,:12]) @ la.inv(STM_full[i,:12,:12])
    TempMatrix2 = np.transpose(STM_full[-1,:12,:12]@la.inv(STM_full[i,:12,:12]))@STT_full[i,12,:12,:12]@(STM_full[-1,:12,:12]@la.inv(STM_full[i,:12,:12]))+TempMatrix1
    Matrix2 = TempMatrix2[:12,:12]

    TempMatrix1_halfway = la.inv(np.transpose(STM_full_halfway[i,:12,:12])) @ (STT_full_halfway[-1,12,:12,:12] - STT_full_halfway[i,12,:12,:12]) @ la.inv(STM_full_halfway[i,:12,:12])
    TempMatrix2_halfway = np.transpose(STM_full_halfway[-1,:12,:12]@la.inv(STM_full_halfway[i,:12,:12]))@STT_full_halfway[i,12,:12,:12]@(STM_full_halfway[-1,:12,:12]@la.inv(STM_full_halfway[i,:12,:12]))+TempMatrix1_halfway
    Matrix2_halfway = TempMatrix2_halfway[:12,:12]

    # eigh returns eigenvalues in ascending order with eigenvectors in their corresponding positions
    E = np.transpose(Matrix1) @ Matrix2 @ Matrix1  # establish E-matrix
    E_star = np.block([[np.identity(6), np.identity(6)]]) @ E @ np.transpose(np.block([[np.identity(6), np.identity(6)]])) # establish Estar matrix
    gamma, w = eigh(E_star) # get eigenvalues and eigenvectors of E_star

    E_halfway = np.transpose(Matrix1_halfway) @ Matrix2_halfway @ Matrix1_halfway  # establish E-matrix
    E_star_halfway = np.block([[np.identity(6), np.identity(6)]]) @ E_halfway @ np.transpose(np.block([[np.identity(6), np.identity(6)]])) # establish Estar matrix
    gamma_halfway, w_halfway = eigh(E_star_halfway) # get eigenvalues and eigenvectors of E_star
    
    # Adjustments to compensate for degeneracy:
    w_adj = w[:,1:6] # remove first eigenvector (the one with infinite extent)
    gamma_adj = gamma[1:6] # remove first eigenvalue (the one with infinite extent)
    Qprime = np.transpose(w_adj) # establish Q matrix with row-vectors corresponding to eigenvectors 
    Eprime = np.diag(gamma_adj) # form adjusted eigenvalue matrix

    w_adj_halfway = w_halfway[:,1:6] # remove first eigenvector (the one with infinite extent)
    gamma_adj_halfway = gamma_halfway[1:6] # remove first eigenvalue (the one with infinite extent)
    Qprime_halfway = np.transpose(w_adj_halfway) # establish Q matrix with row-vectors corresponding to eigenvectors 
    Eprime_halfway = np.diag(gamma_adj_halfway) # form adjusted eigenvalue matrix

    lam, z = eigh(Qprime @ np.transpose(A) @ A @ np.transpose(Qprime), Eprime) # solve adjusted problem for new eigenvalues and eigenvectors
    lamyz, zyz = eigh(Qprime @ np.transpose(Ayz) @ Ayz @ np.transpose(Qprime), Eprime) # solve adjusted problem for new eigenvalues and eigenvectors

    lam_halfway, z_halfway = eigh(Qprime_halfway @ np.transpose(A) @ A @ np.transpose(Qprime_halfway), Eprime_halfway) # solve adjusted problem for new eigenvalues and eigenvectors
    lamyz_halfway, zyz_halfway = eigh(Qprime_halfway @ np.transpose(Ayz) @ Ayz @ np.transpose(Qprime_halfway), Eprime_halfway) # solve adjusted problem for new eigenvalues and eigenvectors
    
    # normalize eigenvectors
    z1 = z[:,3] / la.norm(z[:,3]) 
    z2 = z[:,4] / la.norm(z[:,4])

    z1_halfway = z_halfway[:,3] / la.norm(z_halfway[:,3]) 
    z2_halfway = z_halfway[:,4] / la.norm(z_halfway[:,4])

    zyz1 = zyz[:,3] / la.norm(zyz[:,3]) 
    zyz2 = zyz[:,4] / la.norm(zyz[:,4])

    zyz1_halfway = zyz_halfway[:,3] / la.norm(zyz_halfway[:,3]) 
    zyz2_halfway = zyz_halfway[:,4] / la.norm(zyz_halfway[:,4])

    # get directions of projected energy ellipsoid
    dir1 = A @ (np.transpose(Qprime)) @ z1 
    dir2 = A @ (np.transpose(Qprime)) @ z2
    angle = np.arctan2(dir1[1],dir1[0]) # radians, angle of ellipse rotation

    dir1_halfway = A @ (np.transpose(Qprime_halfway)) @ z1_halfway 
    dir2_halfway = A @ (np.transpose(Qprime_halfway)) @ z2_halfway
    angle_halfway = np.arctan2(dir1_halfway[1],dir1_halfway[0]) # radians, angle of ellipse rotation

    # get directions of projected energy ellipsoid
    diryz1 = Ayz @ (np.transpose(Qprime)) @ zyz1 
    diryz2 = Ayz @ (np.transpose(Qprime)) @ zyz2
    angleyz = np.arctan2(diryz1[1],diryz1[0]) # radians, angle of ellipse rotation

    # get directions of projected energy ellipsoid
    diryz1_halfway = Ayz @ (np.transpose(Qprime_halfway)) @ zyz1_halfway
    diryz2_halfway = Ayz @ (np.transpose(Qprime_halfway)) @ zyz2_halfway
    angleyz_halfway = np.arctan2(diryz1_halfway[1],diryz1_halfway[0]) # radians, angle of ellipse rotation

    ext1 = 2*np.sqrt(2*J_max*lam[3]) #  width of ellipse projection
    ext2 = 2*np.sqrt(2*J_max*lam[4]) # height of ellipse projection
    extyz1 = 2*np.sqrt(2*J_max*lamyz[3]) #  width of ellipse projection
    extyz2 = 2*np.sqrt(2*J_max*lamyz[4]) # height of ellipse projection

    ext1_halfway = 2*np.sqrt(2*J_max*lam_halfway[3]) #  width of ellipse projection
    ext2_halfway = 2*np.sqrt(2*J_max*lam_halfway[4]) # height of ellipse projection
    extyz1_halfway = 2*np.sqrt(2*J_max*lamyz_halfway[3]) #  width of ellipse projection
    extyz2_halfway = 2*np.sqrt(2*J_max*lamyz_halfway[4]) # height of ellipse projection

    rotation = angle*180/np.pi
    rotationyz = angleyz*180/np.pi

    rotation_halfway = angle_halfway*180/np.pi
    rotationyz_halfway = angleyz_halfway*180/np.pi

    position = state_full[i,0:3] # center of ellipse for current timestep
    pos[:,i] = np.transpose(state_full[i,0:2])

    position_halfway = state_full_halfway[i,0:3] # center of ellipse for current timestep
    pos_halfway[:,i] = np.transpose(state_full_halfway[i,0:2])

    plotangle = angle
    tipsright[:,i] = xypos[:,i] + np.array([ext2/2*np.cos(plotangle-np.pi/2),ext2/2*np.sin(plotangle-np.pi/2)])
    tipsleft[:,i] = xypos[:,i] - np.array([ext2/2*np.cos(plotangle-np.pi/2),ext2/2*np.sin(plotangle-np.pi/2)])

    # tipsright[:,i] = xypos[:,i] + (dir2 / la.norm(dir2))*ext2/2
    # tipsleft[:,i] = xypos[:,i] - (dir2 / la.norm(dir2))*ext2/2

    plotangle_halfway = angle_halfway
    tipsright_halfway[:,i] = xypos_halfway[:,i] + np.array([ext2_halfway/2*np.cos(plotangle_halfway-np.pi/2),ext2_halfway/2*np.sin(plotangle_halfway-np.pi/2)])
    tipsleft_halfway[:,i] = xypos_halfway[:,i] - np.array([ext2_halfway/2*np.cos(plotangle_halfway-np.pi/2),ext2_halfway/2*np.sin(plotangle_halfway-np.pi/2)])

    # if i == 20:
    #     print("CHECK OF THING: ")
    #     print("x-coordinate of center: " + str(position[0]))
    #     print("y-coordinate of center: " + str(position[1]))
    #     print("Extent 1 of ellipse: " + str(ext1))
    #     print("Extent 2 of ellipse: " + str(ext2))
    #     print("Ellipse angle (deg): " + str(rotation))
    #     print(" ")
    #     print("x-center of tip-line: " + str(xypos[0,i]))
    #     print("y-center of tip-line: " + str(xypos[1,i]))
    #     print("x-coordinate of tip: " + str(tipsright[0,i]))
    #     print("y-coordinate of tip: " + str(tipsright[1,i]))
    #     print(" ")



    # ellipse plotting:
    if i % 10 == 0:
        if i == 0:
            ellipse = matplotlib.patches.Ellipse(xy=(position[0],[position[1]]), width=ext1, height=ext2, edgecolor=[252/255, 227/255, 3/255], fc='r', angle=rotation, linewidth= 0)

        else:
            ellipse = matplotlib.patches.Ellipse(xy=(position[0],[position[1]]), width=ext1, height=ext2, edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=rotation, linewidth= 0)
    # ellipse = matplotlib.patches.Ellipse(xy=(position[1],position[2]), width=extyz1, height=extyz2, edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=rotationyz)

        ax.add_patch(ellipse)


tipsoutside = np.zeros((2,rows))
tipsinside = np.zeros((2,rows))

center = np.array([np.average(state_full[:,0]),np.average(state_full[:,1])])

for i in range(0,rows):
    if la.norm(tipsright[:,i] - center) > la.norm(pos[:,i] - center):
        tipsoutside[:,i] = tipsright[:,i]
        tipsinside[:,i] = tipsleft[:,i]
    else:
        tipsoutside[:,i] = tipsleft[:,i]
        tipsinside[:,i] = tipsright[:,i]


tipsoutside_halfway = np.zeros((2,rows))
tipsinside_halfway = np.zeros((2,rows))

center_halfway = np.array([np.average(state_full_halfway[:,0]),np.average(state_full_halfway[:,1])])

for i in range(0,rows):
    if la.norm(tipsright_halfway[:,i] - center_halfway) > la.norm(pos_halfway[:,i] - center_halfway):
        tipsoutside_halfway[:,i] = tipsright_halfway[:,i]
        tipsinside_halfway[:,i] = tipsleft_halfway[:,i]
    else:
        tipsoutside_halfway[:,i] = tipsleft_halfway[:,i]
        tipsinside_halfway[:,i] = tipsright_halfway[:,i]



ref_trajectory, = ax.plot(state_full[:,0], state_full[:,1], color=[0,0,0], label='Reference Trajectory')


bounds, = ax.plot(tipsoutside[0,:],tipsoutside[1,:], label='Bounds from Start Point 1', color = 'b', linewidth=3)
bounds_halfway, = ax.plot(tipsoutside_halfway[0,:],tipsoutside_halfway[1,:], label='Bounds from Start Point 2',color = 'r', linestyle='dashed')
ax.plot(tipsinside[0,:],tipsinside[1,:], color = 'b', linewidth=3)
ax.plot(tipsinside_halfway[0,:],tipsinside_halfway[1,:], color = 'r', linestyle='dashed')
SP1, = ax.plot(state_full[0,0], state_full[0,1],'bo', markersize=8, label='Start Point 1') 
SP2, = ax.plot(state_full_halfway[0,0], state_full_halfway[0,1],'ro', markersize=8, label='Start Point 2') 
ax.legend(handles=[ref_trajectory,bounds, bounds_halfway,SP1,SP2],loc='upper right')
for i in range(0,rows):
    if i % 10 == 0:
        ax.scatter(tipsoutside[0,i],tipsoutside[1,i], color = 'g')
        ax.scatter(tipsinside[0,i],tipsinside[1,i], color = 'g')
    
plt.title('Hyperellipsoid Projections in XY-Plane')
plt.xlabel("X [DU]")
plt.ylabel("Y [DU]")

# plt.plot(state_full[:,1], state_full[:,2], color=[0,0,0])
# plt.xlabel("Y [DU]")
# plt.ylabel("Z [DU]")
plt.show()


##### CONVERSION TO FUNCTION #####
def projection(STM_full,STT_full,state_full,J_max,dim1,dim2):
    # STM_full is the set of state transition matrices starting from t = 0 to t = end
    # STT_full is the set of state transition tensors starting from t = 0 to t = end
    # state_full is the set of states starting from t = 0 to t = end
    # J_max is the max energy cost associated with the orbit
    # dim1 and dim2 are strings describing the plot dimensions on the x-axis and y-axis, respectively
    # options for dim1 and dim2 are 'x', 'y', 'z', 'xdot', 'ydot', and 'zdot'

    fig, ax = plt.subplots()
    ax = plt.gca()
    plt.title('Hyperellipsoid Projections in 2D Plane')
    ellipses = []

    # establish projection matrix and relevant indices for plotting
    A = np.zeros((2,6)) 
    if dim1 == 'x':
        s1 = 0
        plt.xlabel("X [DU]")
    elif dim1 == 'y':
        s1 = 1
        plt.xlabel("Y [DU]")
    elif dim1 == 'z':
        s1 = 2
        plt.xlabel("Z [DU]")
    elif dim1 == 'xdot':
        s1 = 3
        plt.xlabel("Xdot [DU/TU]")
    elif dim1 == 'ydot':
        s1 = 4
        plt.xlabel("Ydot [DU/TU]")
    elif dim1 == 'zdot':
        s1 = 5
        plt.xlabel('Zdot [DU/TU]')

    if dim2 == 'x':
        s2 = 0
        plt.ylabel("X [DU]")
    elif dim2 == 'y':
        s2 = 1
        plt.ylabel("Y [DU]")
    elif dim2 == 'z':
        s2 = 2
        plt.ylabel("Z [DU]")
    elif dim2 == 'xdot':
        s2 = 3
        plt.ylabel("Xdot [DU/TU]")
    elif dim2 == 'ydot':
        s2 = 4
        plt.ylabel("Ydot [DU/TU]")
    elif dim2 == 'zdot':
        s2 = 5
        plt.ylabel("Zdot [DU/TU]")

    A[0,s1] = 1
    A[1,s2] = 1

    ref_trajectory, = plt.plot(state_full[:,s1], state_full[:,s2], color=[0,0,0], label='Reference Trajectory')

    [rows,columns,depth] = STM_full.shape # dimensions of STM_full
    ellipse_info = np.zeros((rows,5)) # contains x-center, y-center, ext1, ext2, and angle

    for i in range(0,rows):

        STM = STM_full[i,:,:] @ STM_full[-1,:,:] @ la.inv(STM_full[i,:,:]) # STM for current timestep
        STT = STT_full[i,:,:,:] # STT for current timestep

        Matrix1 = np.block([[np.identity(6), np.zeros((6,6))],
        [-la.solve(STM[:6, 6:12], STM[:6, :6]), la.inv(STM[:6, 6:12])]]) # construct Matrix1 from STMS

        # work with 13 x  13 matrices:
        # TempMatrix1 = la.inv(np.transpose(STM_full[i,:,:])) @ (STT_full[-1,12,:,:] - STT_full[i,12,:,:]) @ la.inv(STM_full[i,:,:])
        # TempMatrix2 = np.transpose(STM_full[-1,:,:]@la.inv(STM_full[i,:,:]))@STT_full[i,12,:,:]@(STM_full[-1,:,:]@la.inv(STM_full[i,:,:]))+TempMatrix1

        # work with 12 x 12 matrices:
        TempMatrix1 = la.inv(np.transpose(STM_full[i,:12,:12])) @ (STT_full[-1,12,:12,:12] - STT_full[i,12,:12,:12]) @ la.inv(STM_full[i,:12,:12])
        TempMatrix2 = np.transpose(STM_full[-1,:12,:12]@la.inv(STM_full[i,:12,:12]))@STT_full[i,12,:12,:12]@(STM_full[-1,:12,:12]@la.inv(STM_full[i,:12,:12]))+TempMatrix1
        Matrix2 = TempMatrix2[:12,:12]

        # eigh returns eigenvalues in ascending order with eigenvectors in their corresponding positions
        E = np.transpose(Matrix1) @ Matrix2 @ Matrix1  # establish E-matrix
        E_star = np.block([[np.identity(6), np.identity(6)]]) @ E @ np.transpose(np.block([[np.identity(6), np.identity(6)]])) # establish Estar matrix
        gamma, w = eigh(E_star) # get eigenvalues and eigenvectors of E_star
        
        # Adjustments to compensate for degeneracy:
        w_adj = w[:,1:6] # remove first eigenvector (the one with infinite extent)
        gamma_adj = gamma[1:6] # remove first eigenvalue (the one with infinite extent)
        Qprime = np.transpose(w_adj) # establish Q matrix with row-vectors corresponding to eigenvectors 
        Eprime = np.diag(gamma_adj) # form adjusted eigenvalue matrix

        lam, z = eigh(Qprime @ np.transpose(A) @ A @ np.transpose(Qprime), Eprime) # solve adjusted problem for new eigenvalues and eigenvectors
        
        # normalize eigenvectors
        z1 = z[:,3] / la.norm(z[:,3]) 
        z2 = z[:,4] / la.norm(z[:,4])

        # get directions of projected energy ellipsoid
        dir1 = A @ (np.transpose(Qprime)) @ z1 
        dir2 = A @ (np.transpose(Qprime)) @ z2
        angle = np.arctan2(dir1[1],dir1[0]) # radians, angle of ellipse rotation


        ext1 = 2*np.sqrt(2*J_max*lam[3]) #  width of ellipse projection
        ext2 = 2*np.sqrt(2*J_max*lam[4]) # height of ellipse projection

        rotation = angle*180/np.pi

        position = np.array([state_full[i,s1],state_full[i,s2]])
        # position[0] = state_full[i,s1]
        # position[1] = state_full[i,s2]

        # ext = np.zeros((2,1))
        # ext[0] = ext2/2*np.cos(angle+np.pi/2)
        # ext[1] = ext2/2*np.sin(angle+np.pi/2)

        # tipsright[:,i] = position  + ext
        # tipsleft[:,i] = position - ext

        # tipsright[:,i] = position + np.array([ext2/2*np.cos(angle-np.pi/2),ext2/2*np.sin(angle-np.pi/2)])
        # tipsleft[:,i] = position - np.array([ext2/2*np.cos(angle-np.pi/2),ext2/2*np.sin(angle-np.pi/2)])
        
        # ellipse plotting:
        ellipse = matplotlib.patches.Ellipse(xy=(position[0],position[1]), width=ext1, height=ext2, edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=rotation)

        ax.add_patch(ellipse)
        ellipses.append(ellipse)
            

        # ellipse_info is n x 5 array storing the x-coordinate of center, y-coordinate of center, semi-major axis, semi-minor axis, and rotation angle
        ellipse_info[i,0] = position[0]
        ellipse_info[i,1] = position[1]

        if ext1 > ext2: # extent 1 is semi-major axis
            ellipse_info[i,2] = ext1
            ellipse_info[i,3] = ext2
            ellipse_info[i,4] = np.arctan2(dir1[1],dir1[0])
        else: # extent 2 is semi-major axis
            ellipse_info[i,2] = ext2
            ellipse_info[i,3] = ext1
            ellipse_info[i,4] = np.arctan2(dir2[1],dir2[0])

        if ellipse_info[i,4] < 0:
            ellipse_info[i,4] = ellipse_info[i,4] + np.pi # correction so that ellipse rotation angle is between 0 and pi
            
        
    plt.legend(["Reference","Reachable Bounds"], loc="upper right")

    plt.show()


    return ellipses, ellipse_info


[ellipses,ellipse_info] = projection(STM_full,STT_full,state_full,J_max,'x','y')


[rows,columns] = ellipse_info.shape

def ellipse_tangent(ell1,ell2):
    h1 = ell1[0]
    k1 = ell1[1]
    a1 = ell1[2] / 2
    b1 = ell1[3] / 2
    r1 = ell1[4]

    h2 = ell2[0]
    k2 = ell2[1]
    a2 = ell2[2] / 2
    b2 = ell2[3] / 2
    r2 = ell2[4]
    
    def func(x):

        p1 = x[0]
        p2 = x[1]
        x1 = x[2]
        x2 = x[3]
        y1 = x[4]
        y2 = x[5]
        m = x[6]

        x[0] = (h1 + a1*np.cos(p1)*np.cos(r1) - b1*np.sin(p1)*np.sin(r1)) - x1
        x[1] = (k1 + a1*np.cos(p1)*np.sin(r1) + b1*np.sin(p1)*np.cos(r1)) - y1
        x[2] = (h2 + a2*np.cos(p2)*np.cos(r2) - b2*np.sin(p2)*np.sin(r2)) - x2
        x[3] = (k2 + a2*np.cos(p2)*np.sin(r2) + b2*np.sin(p2)*np.cos(r2)) - y2
        x[4] = m*(x2-x1) - (y2-y1)
        x[5] = ((-b1/a1)*(1/np.tan(p1)) + np.tan(r1)) / (1 + (b1/a1)*(1/np.tan(p1))*np.tan(r1)) - m
        x[6] = ((-b2/a2)*(1/np.tan(p2)) + np.tan(r2)) / (1 + (b2/a2)*(1/np.tan(p2))*np.tan(r2)) - m

        return x
    

    slope_guess = (k2 - k1) / (h2 - h1)
    guess = [0.2,0.2,h1,k1,h2,k2,0.1]
    solution = fsolve(func,guess)

    return solution


def tangent_lines(ell1,ell2,index):
    print(index)
    C = 10000 # scaling factor for ellipse
    pi = np.pi

    # ellipse 1 constants
    h1 = ell1[0] *C
    k1 = ell1[1] *C
    a1 = ell1[2]/2 * C
    b1 = ell1[3]/2 * C
    r1 = ell1[4]

    # ellipse 2 constants
    h2 = ell2[0] * C
    k2 = ell2[1] * C
    a2 = ell2[2]/2 * C
    b2 = ell2[3]/2 * C
    r2 = ell2[4]

    def cot(theta):
        return 1 / np.tan(theta)
    
    # define constraints for tangent line connecting two ellipses:
    con1 = lambda x: (h1 + a1*np.cos(x[0])*np.cos(r1) - b1*np.sin(x[0])*np.sin(r1)) - x[2]
    con2 = lambda x: (k1 + a1*np.cos(x[0])*np.sin(r1) + b1*np.sin(x[0])*np.cos(r1)) - x[4]
    con3 = lambda x: (h2 + a2*np.cos(x[1])*np.cos(r2) - b2*np.sin(x[1])*np.sin(r2)) - x[3]
    con4 = lambda x: (k2 + a2*np.cos(x[1])*np.sin(r2) + b2*np.sin(x[1])*np.cos(r2)) - x[5]
    con5 = lambda x: x[6]*(x[3]-x[2]) - (x[5]-x[4])
    con6 = lambda x: ((-b1/a1)*(cot(x[0])) + np.tan(r1)) / (1 + (b1/a1)*(cot(x[0]))*np.tan(r1)) - x[6]
    con7 = lambda x: ((-b2/a2)*(cot(x[1])) + np.tan(r2)) / (1 + (b2/a2)*(cot(x[1]))*np.tan(r2)) - x[6]

    nonlinear_constraints = ({'type': 'eq', 'fun': con1},
        {'type': 'eq', 'fun': con2},
        {'type': 'eq', 'fun': con3},
        {'type': 'eq', 'fun': con4},
        {'type': 'eq', 'fun': con5},
        {'type': 'eq', 'fun': con6},
        {'type': 'eq', 'fun': con7},)
    
    def objective_fun(x):
        return 0
    
    two_solutions_found = False
    one_solution_found = False
    jump = 1

    # ELLIPSE 1 PHI CALCULATIONS
    phi_ell1 = np.arctan(-b1/a1*np.tan(r1))

    if phi_ell1 < 0:
        phi1_ell1 = phi_ell1 + np.pi # make phi1_ell1 between 0 and pi
    else:
        phi1_ell1 = phi_ell1

    phi2_ell1 = phi1_ell1 + np.pi # make phi2_ell1 between pi and 2pi

    # ELLIPSE 2 PHI CALCULATIONS
    phi_ell2 = np.arctan(-b2/a2*np.tan(r2))
    if phi_ell2 < 0:
        phi1_ell2 = phi_ell2 + np.pi # make phi1_ell1 between 0 and pi
    else:
        phi1_ell2 = phi_ell2

    phi2_ell2 = phi1_ell2 + np.pi # make phi2_ell1 between pi and 2pi

    A = [
        (0, phi1_ell1),
        (phi1_ell1, np.pi),
        (np.pi, phi2_ell1),
        (phi2_ell1, 2 * np.pi)
    ]

    B = [
        (0, phi1_ell2),
        (phi1_ell2, np.pi),
        (np.pi, phi2_ell2),
        (phi2_ell2, 2 * np.pi)
    ]

    # Generate all combinations of one interval from A and one from B
    combinations = list(itertools.product(A, B))

    # rearrange so more likely combinations are in front
    int2 = combinations[15]
    int_replace = combinations[1]

    combinations[1] = int2
    combinations[15] = int_replace

    int2 = combinations[5]
    int_replace = combinations[2]

    combinations[2] = int2
    combinations[5] = int_replace

    int2 = combinations[10]
    int_replace = combinations[1]

    combinations[1] = int2
    combinations[10] = int_replace

    int2 = combinations[10]
    int_replace = combinations[3]

    combinations[3] = int2
    combinations[10] = int_replace


    solutions_found = 0
    i = 0

    while (solutions_found < 2) and (i < 16):
        p10 = (combinations[i][0][0] + combinations[i][0][1]) / 2
        p20 = (combinations[i][1][0] + combinations[i][1][1]) / 2
        x10 = (h1+a1*np.cos(r1))+0.05
        x20 = (h2+a2*np.cos(r2))+0.05
        y10 = (k1 + a1*np.sin(r1))+0.05
        y20 = (k2 + a2*np.sin(r1))+0.05
        m0 = (k2 - k1) / (h2 - h1)

        p1bnds = combinations[i][0]
        p2bnds = combinations[i][1]
        x1bnds = (None,None)
        x2bnds = (None,None)
        y1bnds = (None,None)
        y2bnds = (None,None)
        mbnds = (None,None)
        bnds = (p1bnds,p2bnds,x1bnds,x2bnds,y1bnds,y2bnds,mbnds)

        X0 = [p10, p20, x10, x20, y10, y20, m0]
        solution = minimize(objective_fun,X0, method='SLSQP',bounds = bnds, tol=1e-10,constraints=nonlinear_constraints,options={'maxiter': 100})

        # find way to change solution.success to false if tangent line crosses line connecting (h1,k1) and (h2,k2)
        if solution.success == True:
            print("BOUND INDEX: " + str(i))
            solutions_found = solutions_found + 1
            solution = solution.x
            for j in range(2,7):
                solution[j] = solution[j] / C

            if solutions_found == 1:
                solution1 = solution
            elif solutions_found == 2:
                solution2 = solution

        i = i + 1

    if solutions_found == 0: # no solutions found, just say both are most recent solution
        solution1 = solution.x
        solution2 = solution1
    elif solutions_found == 1: # one solution found, just say second solution is same as first
        solution2 = solution1


    # while two_solutions_found == False:
    #     two_solutions_found = True
    #     while one_solution_found == False:
    #         one_solution_found = True

    #         p10 = 3*pi/4
    #         p20 = 3*pi/4
    #         x10 = (h1-a1*np.cos(r1))
    #         x20 = (h2-a2*np.cos(r2))
    #         y10 = (k1 + a1*np.sin(r1))
    #         y20 = (k2 + a2*np.sin(r1))
    #         m0 = (k2 - k1) / (h2 - h1)

    #         p1bnds = (pi/2,pi)
    #         p2bnds = (pi/2,pi)
    #         x1bnds = (h1-a1,h1)
    #         x2bnds = (h2-a2,h2)
    #         y1bnds = (k1-a1,k1+a1)
    #         y2bnds = (k2-a2,k2+a2)
    #         mbnds = (-100,100)

    #         # bnds = (p1bnds,p2bnds,x1bnds,x2bnds,y1bnds,y2bnds,mbnds)

    #         x0 = np.array([p10,p20,x10,x20,y10,y20,m0])

    #         solution1 = minimize(objective_fun,x0, method='SLSQP',tol=1e-10,constraints=nonlinear_constraints,options={'maxiter': 250})
    #         solution1.success = True
    #         if solution1.success == True:
    #             one_solution_found = True
    #             solution1 = solution1.x
    #             for j in range(2,7):
    #                 solution1[j] = solution1[j] / C

    #     if solution1[2] > h1: # x connector is to the right of the center
    #         p10 = pi/4
    #         p20 = pi/4
    #         x10 = (h1 - a1*np.cos(r1))
    #         x20 = (h2 - a2*np.cos(r2))
    #         y10 = (k1 - a1*np.sin(r1))
    #         y20 = (k2 - a2*np.sin(r1))
    #         m0 = m0 = (k2 - k1) / (h2 - h1)
    #     else:
    #         p10 = pi/4
    #         p20 = pi/4
    #         x10 = (h1+a1*np.cos(r1))
    #         x20 = (h2+a2*np.cos(r2))
    #         y10 = (k1 + a1*np.sin(r1))
    #         y20 = (k2 + a2*np.sin(r1))
    #         m0 = (k2 - k1) / (h2 - h1)
        
    #     x0 = np.array([p10,p20,x10,x20,y10,y20,m0])

    #     p1bnds = (0.01,pi-0.01)
    #     p2bnds = (0.01,pi-0.01)
    #     x1bnds = (h1-a1,h1+a1)
    #     x2bnds = (h2-a2,h2+a2)
    #     y1bnds = (k1,k1+a1)
    #     y2bnds = (k2,k2+a2)
    #     mbnds = (-100,100)

    #     bnds = (p1bnds,p2bnds,x1bnds,x2bnds,y1bnds,y2bnds,mbnds)
    #     # solution2 = minimize(objective_fun,x0, method='SLSQP',tol=1e-10,bounds = bnds, constraints=nonlinear_constraints,options={'maxiter': 500})
    #     # solution2 = solution2.x


    #     # for j in range(2,7):
    #     #     solution2[j] = solution2[j] / C

     
    # # solution2 = solution1 # COMMENT OUT THIS LINE WHEN TRYING TO SEARCH FOR SOLUTION 2



        
    return solution1,solution2
    
fig, ax = plt.subplots()
ax = plt.gca()
ax.set_xlim([0.95, 1.15])
ax.set_ylim([-0.1, 0.1])
ax.set_aspect('equal', adjustable='box')

# limits = [120,122]
limits = [0,259]
solution1 = np.zeros((limits[1]-limits[0],7))
solution2 = np.zeros((limits[1]-limits[0],7))

for i in range(limits[0],limits[1]):
    ell1 = ellipse_info[i,:]
    ell2 = ellipse_info[i+1,:]

    if i == limits[0]:
        print("ELLIPSES: ")
        print(ell1)
        print(" ")
        print(ell2)

    solution1[i-limits[0],:],solution2[i-limits[0],:] = tangent_lines(ell1,ell2,i)

    ellipse = matplotlib.patches.Ellipse(xy=(ellipse_info[i,0],ellipse_info[i,1]), width=ellipse_info[i,2], height=ellipse_info[i,3], edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=ellipse_info[i,4]*180/np.pi)
    ax.add_patch(ellipse)

[rows,columns] = solution1.shape

for i in range(0,rows):
    x_points1 = np.array([solution1[i,2],solution1[i,3]])
    y_points1 = np.array([solution1[i,4],solution1[i,5]])
    plt.plot(x_points1,y_points1,linewidth= 2.5)

    x_points2 = np.array([solution2[i,2],solution2[i,3]])
    y_points2 = np.array([solution2[i,4],solution2[i,5]])
    plt.plot(x_points2,y_points2,linewidth= 2.5)


plt.xlabel("X [DU]")
plt.ylabel("Y [DU]")
plt.title("Projections with Tangent Lines")
plt.show()

############################
STM = STM_full[-1,:,:]
STT = STT_full[-1,:,:,:]

time = []
for i in range(len(state_full)):
    time.append(i/len(state_full))

# conduct E-matrix computation below, but do it in for-loop above to compute E-matrix at every time


# Compute the E matrix 
Matrix1 = np.block([[np.identity(6), np.zeros((6,6))],
    [-la.solve(STM[:6, 6:12], STM[:6, :6]), la.inv(STM[:6, 6:12])]])
Matrix2 = STT[12, :12, :12]
E = np.transpose(Matrix1) @ Matrix2 @ Matrix1 

E_star = np.block([[np.identity(6), np.identity(6)]]) @ E @ np.transpose(np.block([[np.identity(6), np.identity(6)]]))

# Determine eigenvalues of E matrix
# gamma is the list of eigenvalues in ascending order
# The normalized eigenvector corresponding to the eigenvalue gamma[i] is the column w[:,i]
gamma, w = eigh(E_star)


# To check eigenstuff, uncomment these lines
#print("Eigenvalue-Eigenvector pairs of E* are:")
#for i in range(6):
#    print(str(gamma[i]) + ", " + str(w[:,i]) + ", " + str(np.linalg.norm(w[:,i])))


a = w


for i in range(6):
    # print("The extent is " + str(la.norm(np.sqrt(2*J_max/gamma[i])*w[:,i])) + " and the direction is " + str(w[:,i]))
    a[:,i] = np.sqrt(2*J_max/gamma[i])*w[:,i]

# print(a[:,0])


################################
###### OBTAIN STATE DATA #######
################################
# Improvement: Make this a function that can just be called rather than commenting in/out

FileName_data = "./state_dataset.mat"

x_dataset = np.zeros((6,260,1000))
# Run if the file does not exist
if not os.path.isfile(FileName_data):
    for i in range(1000):
        rand_vec = np.reshape(np.hstack((np.array(w[:,1]), np.array(w[:,2]), np.array(w[:,3]), np.array(w[:,4]), np.array(w[:,5]))), (6,5)) @ np.random.standard_normal(5)
        rand_vec = np.reshape(rand_vec,(6,1))
        rand_vec = rand_vec/la.norm(rand_vec) #normalizes the vector
        scaling_factor = np.sqrt(2*J_max/(np.transpose(rand_vec) @ E_star @ rand_vec))
        dx_0 = rand_vec * scaling_factor
        costates_initial = la.inv(STM[:6,6:12]) @ (np.identity(6) - STM[:6,:6]) @ dx_0 
        x = np.zeros((6,260))
        for index in range(len(state_full)):
            dx = STM_full[index,:6,:6] @ dx_0 + STM_full[index,:6,6:12] @ costates_initial
            x[:,index] = state_full[index,:6] + np.reshape(dx, (6,))
        x_dataset[:,:,i] = x
        if i/1000 == 1 or i/10000 ==1 or i/20000 == 1:
            print(i)
    scipy.io.savemat(FileName_data, {"State_Data": x_dataset})

# load data
x_dataset = list(scipy.io.loadmat(FileName_data).values())[-1]











##################################
###### SAMPLED YELLOWPLOTS #######
##################################
# Improvement: Make this call a state_dataset function

# load data
FileName_data = "./state_dataset.mat"
x_dataset = list(scipy.io.loadmat(FileName_data).values())[-1]
fig = plt.figure()
# for 3d plot
# ax = fig.add_subplot(projection="3d")
# plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
# for 2d plot
plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
for i in range(1000):
    x = x_dataset[:,:,i]
    #color_random = list(np.random.choice(range(256), size=3)/256)
    color_random = [252/255, 227/255, 3/255]

    # For a 3d orbit plot
    #plt.plot(x[0,:], x[1,:], x[2,:], color=color_random)

    # For a 2d orbit plot
    plt.plot(x[0,:], x[1,:],color=color_random, alpha=1)

# for 2d orbit
plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
plt.plot(tipsoutside[0,:],tipsoutside[1,:], color = 'b')
plt.plot(tipsinside[0,:],tipsinside[1,:], color = 'b')
plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
# plt.plot(tipsright[:,0],tipsright[0:,1])
# plt.plot(tipsleft[:,0],tipsleft[0:,1])
plt.xlabel("X [DU]")
plt.ylabel("Y [DU]")
# for 3d orbit
#plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
#ax.set(
#    xlabel='X [DU]',
#    ylabel='Y [DU]',
#    zlabel='Z [DU]',
#)
#plt.savefig('3d_trajectories.png', dpi=500)
plt.show()





# DELETE STARTING HERE










""""

##################################
###### ELLIPSE YELLOWPLOTS #######
##################################
# Improvement: make this work based on method that Jackson wrote up
# Does not work currently - VERY close but not quite there

plt.figure()
#plt.xlim(0.985, 1.075)
#plt.ylim(-0.1, 0.1)
ax = plt.gca()
plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])

#for index in range(len(state_full)):
for index in range(1):
    A = STM_full[index, (0, 1), :6] + STM_full[index, (0, 1), 6:12] @ la.inv(STM[:6, 6:12]) @ (np.identity(6) - STM[:6, :6])
    # gamma is the list of eigenvalues in ascending order
    # The normalized eigenvector corresponding to the eigenvalue gamma[i] is the column w[:,i]
    gamma_A, w_A = eigh(A.T @ A, E_star)
    semiaxes = np.zeros((2,2))
    k=0

    for i in range(2): # Matt adjusted this from 6 to 2 to get it to run
        w_A[:,i] = w_A[:,i]/la.norm(w_A[:,i]) # normalize the eigenvectors

        
        if gamma_A[i] > 1e-10 and gamma_A[i] < 1:

            # This line will scale alpha so it is a semi-axis
            scaling_factor = np.sqrt(2*J_max/(np.transpose(w_A[:,i]) @ E_star @ w_A[:,i]))
            alpha = scaling_factor * A @ w_A[:,i]

            semiaxes[k,:] = alpha
            k+=1

    rotation = 180/np.pi*np.arctan2(semiaxes[0,1], semiaxes[0,0]) # in degrees

    ellipse = matplotlib.patches.Ellipse(xy=(state_full[index,0], state_full[index,1]), width=la.norm(semiaxes[0,:]), height=la.norm(semiaxes[1,:]), edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=rotation)
    ax.add_patch(ellipse)


#plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
plt.xlabel("X [DU]")
plt.ylabel("Y [DU]")
#plt.savefig('Ellipse_XY.pdf', format='pdf')
plt.show()

















"""


#################################
###### FILLED YELLOWPLOTS #######
#################################
# Improvement: remove completely after ellipse code because this is expensive and incomplete
"""
fig = plt.figure()
# for 3d plot
#ax = fig.add_subplot(projection="3d")
#plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
# for 2d plot
plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
dx_min = np.zeros((6,260))
dx_max = np.zeros((6,260))

#
for i in range(1000):
    rand_vec = np.reshape(np.hstack((np.array(w[:,1]), np.array(w[:,2]), np.array(w[:,3]), np.array(w[:,4]), np.array(w[:,5]))), (6,5)) @ np.random.standard_normal(5)
    rand_vec = np.reshape(rand_vec,(6,1))
    rand_vec = rand_vec/la.norm(rand_vec) #normalizes the vector
    scaling_factor = np.sqrt(2*J_max/(np.transpose(rand_vec) @ E_star @ rand_vec))
    dx_0 = rand_vec * scaling_factor
    costates_initial = la.inv(STM[:6,6:12]) @ (np.identity(6) - STM[:6,:6]) @ dx_0 
    x = np.zeros((6,260))
    dx_norm = []
    dlambda_norm = []
    for index in range(len(state_full)):
        dx = STM_full[index,:6,:6] @ dx_0 + STM_full[index,:6,6:12] @ costates_initial
        dlambda = STM_full[index,6:12,:6] @ dx_0 + STM_full[index,6:12,6:12] @ costates_initial
        dx_norm.append(la.norm(dx[:3]))
        dlambda_norm.append(la.norm([dlambda[3:]]))
        x[:,index] = state_full[index,:6] + np.reshape(dx, (6,))

    #x_dataset[:,:,i] = x
    dx_matrix = x[:,:] - np.reshape(state_full[:,:6], (6,260))
    for i in range(6):
        for j in range(260):
            if dx_matrix[i,j] < dx_min[i,j]:
                dx_min[i,j] = dx_matrix[i,j]
            if dx_matrix[i,j] > dx_max[i,j]:
                dx_max[i,j] = dx_matrix[i,j]

    #breakpoint()
    #x = np.reshape(x ,(index,6))
    #color_random = list(np.random.choice(range(256), size=3)/256)
    color_random = [252/255, 227/255, 3/255]

    # For a 3d orbit plot
    #plt.plot(x[0,:], x[1,:], x[2,:], color=color_random, alpha=0.2)

    # For a 2d orbit plot
    plt.plot(x[0,:], x[1,:],color=color_random, alpha=1)

    # For a displacement plot
    #plt.plot(time, dlambda_norm,color=color_random)

x_min = np.reshape(state_full[:,:6], (6,260)) + dx_min
x_max = np.reshape(state_full[:,:6], (6,260)) + dx_max

#breakpoint()
#plt.fill_between(state_full[:,0], x_min[1,:], x_max[1,:])
plt.plot(dx_min[0,:], dx_min[1,:])


#for displacement plot
#plt.hlines(0, 0, 1, color = [0,0,0])
#plt.xlabel("Time [TU]")
#plt.ylabel("Thrust Magnitude [DU/TU^2]")
# for 2d orbit
plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
plt.xlabel("X [DU]")
plt.ylabel("Y [DU]")
# for 3d orbit
#plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
#ax.set(
#    xlabel='X [DU]',
#    ylabel='Y [DU]',
#    zlabel='Z [DU]',
#)
#plt.savefig('YellowBounds_XY.pdf', format='pdf')
plt.show()
"""




# DELETE UP TO HERE

"""



########################
###### EIGENPLOT #######
########################
# Improvement: Just clean up a bit

fig = plt.figure()
# for 3d plot
#ax = fig.add_subplot(projection="3d")
#plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
# for 2d plot
#plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
plt.hlines(0, 0, 1, color = [0,0,0])
color_counter = np.zeros((1,6))
for i in [1, 2, 3, 4, 5]:
    rand_vec = w[:,i]
    rand_vec = np.reshape(rand_vec,(6,1))
    rand_vec = rand_vec/la.norm(rand_vec) #normalizes the vector
    scaling_factor = np.sqrt(2*J_max/(np.transpose(rand_vec) @ E_star @ rand_vec))
    dx_0 = rand_vec * scaling_factor
    costates_initial = la.inv(STM[:6,6:12]) @ (np.identity(6) - STM[:6,:6]) @ dx_0 
    x = np.zeros((6,260))
    dx_norm = []
    dlambda_norm = []
    for index in range(len(state_full)):
        dx = STM_full[index,:6,:6] @ dx_0 + STM_full[index,:6,6:12] @ costates_initial
        dlambda = STM_full[index,6:12,:6] @ dx_0 + STM_full[index,6:12,6:12] @ costates_initial
        dx_norm.append(la.norm(dx[:3]))
        dlambda_norm.append(la.norm([dlambda[3:]]))
        x[:,index] = state_full[index,:6] + np.reshape(dx, (6,))


    angles = [np.abs(np.dot(rand_vec[:,0],w[:,0])), np.abs(np.dot(rand_vec[:,0],w[:,1])), 
    np.abs(np.dot(rand_vec[:,0],w[:,2])), np.abs(np.dot(rand_vec[:,0],w[:,3])), 
    np.abs(np.dot(rand_vec[:,0],w[:,4])), np.abs(np.dot(rand_vec[:,0],w[:,5]))]

    #breakpoint()
    #x = np.reshape(x ,(index,6))
    #color_random = list(np.random.choice(range(256), size=3)/256)
    #color_random = [252/255, 227/255, 3/255]
    index_max = np.argmax(angles)
    if index_max == 1:
        color_random = '#FFC0CB'
        color_counter = color_counter + np.array([0, 1, 0, 0, 0, 0])
    elif index_max == 2:
        color_random = '#994F00'
        color_counter = color_counter + np.array([0, 0, 1, 0, 0, 0])
    elif index_max == 3:
        color_random = '#006CD1'
        color_counter = color_counter + np.array([0, 0, 0, 1, 0, 0])
    elif index_max == 4:
        color_random = '#E1BE6A'
        color_counter = color_counter + np.array([0, 0, 0, 0, 1, 0])
    elif index_max == 5:
        color_random = '#FF5F1F'
        color_counter = color_counter + np.array([0, 0, 0, 0, 0, 1])
    else:
        color_random = '#40B0A6'
        color_counter = color_counter + np.array([1, 0, 0, 0, 0, 0])
    # For a 3d orbit plot
    #plt.plot(x[0,:], x[1,:], x[2,:], color=color_random, alpha=0.2)

    # For a 2d orbit plot
    #plt.plot(x[0,:], x[1,:],color=color_random, alpha=1)

    # For a displacement plot
    #plt.plot(time, x[3,:] - np.reshape(state_full[:,3],(260,)), color=color_random)
    plt.plot(time, dlambda_norm, color=color_random)

print(color_counter)

plt.legend(["Ref","2nd", "3rd", "4th", "5th", "6th"])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[1].set_color('#FFC0CB')
leg.legend_handles[2].set_color('#994F00')
leg.legend_handles[3].set_color('#006CD1')
leg.legend_handles[4].set_color('#E1BE6A')
leg.legend_handles[5].set_color('#FF5F1F')
#leg.legend_handles[5].set_color('#40B0A6')


#for displacement plot
plt.xlabel("Time [TU]")
plt.ylabel("Thrust [DU/TU^2]")
# for 2d orbit
#plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
#plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
#plt.xlabel("X [DU]")
#plt.ylabel("Y [DU]")
# for 3d orbit
#ax.set(
#    xlabel='X [DU]',
#    ylabel='Y [DU]',
#    zlabel='Z [DU]',
#)
#plt.savefig('Eigen_Thrust.pdf', format='pdf')
plt.show()


fig = plt.figure()
ax = plt.gca()
xmin = -20
xmax = 20
ymin = -20
ymax = 20
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ellipse = matplotlib.patches.Ellipse(xy=[2,0], width=10, height=5, angle=30)
ax.add_patch(ellipse)
ellipse = matplotlib.patches.Ellipse(xy=[-6,-6], width=10, height=5, angle=30)
ax.add_patch(ellipse)
plt.show()
"""




#########################
###### COLORPLOT ########
#########################
"""
fig = plt.figure()
# for 3d plot
#ax = fig.add_subplot(projection="3d")
#plt.plot(state_full[:,0], state_full[:,1], state_full[:,2], color=[0,0,0])
# for 2d plot
#plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
color_counter = np.zeros((1,6))
for i in range(100):
    rand_vec = np.random.standard_normal(6)
    rand_vec = np.reshape(rand_vec,(6,1))
    rand_vec = rand_vec/la.norm(rand_vec) #normalizes the vector
    scaling_factor = np.sqrt(2*J_max/(np.transpose(rand_vec) @ E_star @ rand_vec))
    dx_0 = rand_vec * scaling_factor
    costates_initial = la.inv(STM[:6,6:12]) @ (np.identity(6) - STM[:6,:6]) @ dx_0 
    x = np.reshape(state_full[0,:6], (6,1)) + dx_0
    dx_norm = []
    dlambda_norm = []
    for index in range(len(state_full)):
        dx = STM_full[index,:6,:6] @ dx_0 + STM_full[index,:6,6:12] @ costates_initial
        dlambda = STM_full[index,6:12,:6] @ dx_0 + STM_full[index,6:12,6:12] @ costates_initial
        dx_norm.append(la.norm(dx[:3]))
        dlambda_norm.append(la.norm([dlambda[3:]]))
        x = np.hstack((x, np.reshape(state_full[index,:6], (6,1)) + dx))

    angles = [np.abs(np.dot(rand_vec[:,0],w[:,0])), np.abs(np.dot(rand_vec[:,0],w[:,1])), 
    np.abs(np.dot(rand_vec[:,0],w[:,2])), np.abs(np.dot(rand_vec[:,0],w[:,3])), 
    np.abs(np.dot(rand_vec[:,0],w[:,4])), np.abs(np.dot(rand_vec[:,0],w[:,5]))]

    index_max = np.argmax(angles)
    if index_max == 0:
        color_random = '#FF5F1F'
        color_counter = color_counter + np.array([1, 0, 0, 0, 0, 0])
    elif index_max == 1:
        color_random = '#FFC0CB'
        color_counter = color_counter + np.array([0, 1, 0, 0, 0, 0])
    elif index_max == 2:
        color_random = '#994F00'
        color_counter = color_counter + np.array([0, 0, 1, 0, 0, 0])
    elif index_max == 3:
        color_random = '#006CD1'
        color_counter = color_counter + np.array([0, 0, 0, 1, 0, 0])
    elif index_max == 4:
        color_random = '#E1BE6A'
        color_counter = color_counter + np.array([0, 0, 0, 0, 1, 0])
    else:
        color_random = '#40B0A6'
        color_counter = color_counter + np.array([0, 0, 0, 0, 0, 1])

    # For a 3d orbit plot
    #plt.plot(x[0,:], x[1,:], x[2,:], color=color_random, alpha=0.2)

    # For a 2d orbit plot
    #plt.plot(x[0,:], x[1,:],color=color_random, alpha=1)

    # For a displacement plot
    plt.plot(time, dx_norm,color=color_random)

print(color_counter)

plt.legend(["1st", "2nd", "3rd", "4th", "5th", "6th"])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[0].set_color('#FF5F1F')
leg.legend_handles[1].set_color('#FFC0CB')
leg.legend_handles[2].set_color('#994F00')
leg.legend_handles[3].set_color('#006CD1')
leg.legend_handles[4].set_color('#E1BE6A')
leg.legend_handles[5].set_color('#40B0A6')


#for displacement plot
plt.hlines(0, 0, 1, color = [0,0,0])
plt.xlabel("Time [TU]")
plt.ylabel("Thrust Magnitude [DU/TU^2]")
# for 2d orbit
#plt.plot(state_full[:,0], state_full[:,1], color=[0,0,0])
#plt.legend(["Reference", "Reachable Bounds"], loc="upper right")
#plt.xlabel("X [DU]")
#plt.ylabel("Y [DU]")
# for 3d orbit
#ax.set(
#    xlabel='X [DU]',
#    ylabel='Y [DU]',
#    zlabel='Z [DU]',
#)
#plt.savefig('Thrust_Magnitude.eps', format='eps')
plt.show()
"""











#########################
###### VALIDATION #######
#########################
# Improvement: Create a more robist continuation scheme

# iterate the state to improve accuracy (doesn't improve accuracy currently)
#state_ics = state_iterate(np.array(ics[:6],ndmin=2).T, 1E-3, 0.001, T_final)
#print(str(state_ics))
"""
costates_guess = np.array([[0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000]])
cost_inherent, unused_parameter_costates = true_cost(np.array(ics[:6],ndmin=2).T, 0, 1E-13, 0.01, T_final, costates_guess)
u_inherent = np.sqrt(2*cost_inherent/T_final)

print("Inherent Cost is " + str(cost_inherent) + " DU^2/TU^2 or "+ str(T_final*0.5*u_inherent**2))
print("Inherent u is " + str(u_inherent) + " DU/TU")

dx_mag = []
J_linear = []
J_computed = []
J_true = []
costates_guess = np.array([[0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000], [0.000000000000000000]])
#for i in [600, 700, 800, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]:
#for i in [100000, 10000, 1000, 700, 500, 400, 300, 260, 225, 200, 175, 150, 135, 110, 100]:
for i in [500000, 50000, 5000, 1000, 800, 500]:
#for i in [5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 1000000000, 10000000000]:
#for i in [7500, 10000, 20000, 30000, 31000, 32500, 34000, 35500, 37000, 38500, 40000, 44000, 50000, 70000]:
    dx = np.array(w[:,3]) * 1/i
    dx_mag.append(la.norm(dx))
    J_linear.append(0.5*np.dot(dx, np.dot(dx,E_star))) # This line is in place of 0.5xtranspose(a)xE*xa where a is 6x1 and E* is 6x6

    # compute the true cost
    cost_computed, costates_guess = true_cost(np.array(ics[:6],ndmin=2).T + np.array(w[:,3],ndmin=2).T * 1/i, u_inherent, 1E-13, 0.01, T_final, costates_guess)
    J_computed.append(cost_computed)

J_difference_computed = 100 * abs(np.array(J_linear) - np.array(J_computed)) / np.array(J_computed)

#print(str(state[:,:6]))
#print("Differences in final and initial state are " + str(100 * (np.array(ics[:6])-np.array(state[:,:6])) / (np.array(ics[:6])+np.array(state[:,:6]))) + " %" )

dx_mag_plot = [dx_mag[0], dx_mag[1], dx_mag[2], dx_mag[-1]]
J_linear_plot = [J_linear[0], J_linear[1], J_linear[2], J_linear[-1]]
J_computed_plot = [J_computed[0], J_computed[1], J_computed[2], J_computed[-1]]
abs_error = abs(np.array(J_linear_plot) - np.array(J_computed_plot))

plt.figure()
plt.loglog(dx_mag_plot, abs_error,'X', color=[0, 0, 0])
plt.xlabel('|dx| [DU]')
plt.ylabel('Absolute Error in J (DU^2/TU^3)')
#plt.savefig('Absolute_Error.pdf', format='pdf')
plt.show()

plt.figure()
plt.loglog(dx_mag_plot, J_linear_plot, 'o', markersize=15, color=[252/255, 227/255, 3/255])
plt.loglog(dx_mag_plot, J_computed_plot, 'X', color=[0, 0, 0])
plt.legend(['Estimate', 'Computed'])
plt.xlabel('|dx| [DU]')
plt.ylabel('J (DU^2/TU^3)')
#plt.savefig('Linear_Computed_Cost.pdf', format='pdf')
plt.show()


plt.figure()
plt.loglog(dx_mag / la.norm(ics[:6]), J_difference_computed, 'bX')
plt.xlabel('|dx|/|x|')
plt.ylabel('% Relative Error in J')
plt.show()
"""

