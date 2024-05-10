# hello

import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
from scipy import integrate as int


# create orbit variables
path_str ="orbitFiles/DRO_11.241_days.p"   
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
DRO_11 = pickle.load(f1)
f1.close()

path_str ="orbitFiles/DRO_13.0486_days.p"   
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
DRO_13 = pickle.load(f1)
f1.close()

path_str ="orbitFiles/L1_S_10.003_days.p"  
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
L1_10 = pickle.load(f1)
f1.close()

path_str ="orbitFiles/L1_S_13.0094_days.p"  
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
L1_13 = pickle.load(f1)
f1.close()

path_str ="orbitFiles/L2_S_6.0066_days.p"   
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
L2_6 = pickle.load(f1)
f1.close()

path_str ="orbitFiles/L2_S_12.05_days.p"  
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
L2_12 = pickle.load(f1)
f1.close()


# establish constants
M_m = 7.349*(10)**22 # mass of moon
M_e = 5.97219*(10)**24 # mass of earth
d = (3.8444*10**5)*1000 # distance between earth and moon in m
R_m = 1737400 # radius of moon
G = 6.6743*10**(-11) # Gravitational constant
pi = math.pi # pi
x_com = (M_m*d) / (M_m + M_e) # center of mass position
m_position = d - x_com # moon distance from barycenter


class orbit_eval:
  def __init__(self,orbit):
    self.orbit = orbit # orbit is a pickle file

  def orbit_volume(self,angle,A):
    # evaluates how well the spacecraft views volume of moon below certain orbit
    # angle is the scope of the telescope in degrees (ex: 2 deg by 2 deg means angle)
    # A is the altitude of orbits you are considering in meters

    state = self.orbit['state'] # gather position and velocity data
    dimensions = np.shape(state)
    rows, columns = dimensions # obtain size of state
    r_satellite = state[:,0:3]*1000
    time = self.orbit['t'] # get time
    phi = (angle) * math.pi / 180 / 2
    Va = (4/3)*pi*(R_m+A)**3

    r_moon = []
    for i in range(rows):
      r_moon.append([m_position,0,0]) # list describing moon position

    r_moon = np.array(r_moon) # convert moon list to array
    r = r_satellite - r_moon #  array describing satellite distance from moon

    V = []
    r_mag_vec = []

    for i in range(0,rows):
      r_mag = (r[i,0]**2 + r[i,1]**2 + r[i,2]**2)**(1/2)
      r_mag_vec.append(r_mag)

      if r_mag < R_m + A:
        # Satellite is within max altitude
        if r_mag*math.sin(phi) < R_m:
          
          # scenario 1 for within altitude
          alpha_fake = math.asin(math.sin(phi)*r_mag/R_m)
          alpha = pi - alpha_fake
          theta = pi - alpha - phi
          L = R_m*math.sin(theta)
          h = r_mag - R_m*math.cos(theta)
          V_cone = (pi/3)*(L**2)*h
          V_cap = (pi/3)*(R_m**3)*(2+math.cos(theta))*(1-math.cos(theta))**2

          scenario = '1within'
          volume_viewed = V_cone - V_cap

        else:
          # scenario 2 for within altitude
          
          alpha = math.asin((math.sin(phi)/(R_m+A))*r_mag)
          theta = pi - alpha - phi

          if theta < (pi/2):
            tau = math.acos(R_m / (R_m + A))
            omega = math.acos(R_m/r_mag)
            gamma = tau+omega

            if gamma < (pi/2):
              # scenario 2a for within altitude

              V_frontcapsmall = (pi/3)*(R_m+A)**3 * (2+math.cos(theta))*(1-math.cos(theta))**2
              V_frontcapbig = (pi/3)*(R_m+A)**3 * (2+math.cos(gamma))*(1-math.cos(gamma))**2
              b = (R_m+A)*math.sin(theta)
              a = (R_m+A)*math.sin(theta)
              h = (R_m+A)*(math.cos(theta) - math.cos(gamma))
              V_conical = (pi/3)*h*(a**2 + a*b + b**2)
              V_s2 = V_frontcapbig - V_frontcapsmall - V_conical

              L = R_m*math.sin(omega)
              h = r - R_m*math.cos(omega)
              V_cone = (pi/3)*(L**2)*h
              V_cap = (pi/3)*(R_m)**3*(2+math.cos(omega))*(1-math.cos(omega))**2
              V_s1 = V_cone - V_cap

              delta = (pi/2) - omega
              S = math.sin(theta)*(R_m+A) / math.sin(phi)
              L_big = S*math.sin(phi)
              h_big = S*math.cos(phi)
              V_bigcone = (pi/3)*(L_big**2)*h_big
              L_small = h_big(math.tan(phi))
              V_smallcone = (pi/3)*(L_small**2)*h_big
              V_s3 = V_bigcone - V_smallcone

              scenario = '2within'
              volume_viewed = V_s1 + V_s2 + V_s3

            else: 
              # scenario 2b for within altitude

              alpha = math.asin(math.sin(phi)*r/(R_m+A))
              omega = math.acos(R_m/r_mag)
              delta = pi/2 - omega
              tau = theta - omega
              q = math.asin(R_m / (R_m+A))
              f = pi/2 - q
              u = pi - omega - f
              a = (R_m+A)*math.sin(u)
              h_small = (R_m+A)*math.sin(theta)
              h_big = r_mag - h_small
              b = h_big*math.tan(delta)
              
              V_frontcap = (pi/3)*(R_m+A)**3 * (2+math.cos(theta))*(1-math.cos(theta))**2
              V_backcap = (pi/3)*(R_m+A)**3 * (2+math.cos(u))*(1-math.cos(u))**2

              h = (R_m + A)*math.cos(theta) + (R_m+A)*math.cos(u)
              V_conical = (pi/3)*h*(a**2 + a*b + b**2)
              Vs3 = (4/3)*pi(R_m+A)**3 - V_frontcap - V_backcap - V_conical

              L = (R_m + A)*math.sin(theta)
              Vs2 = (pi/3)*(L**2)*h_big - (pi/3)*(b**2)*h_big

              omega = math.acos(R_m/(R_m+A))
              L = R_m*math.sin(omega)
              h = r_mag - R_m*math.cos(omega)
              V_cone = (pi/3)*L**2*h
              V_cap = (pi/3)*(R_m)**3*(2+math.cos(omega))*(1-math.cos(omega))**2
              Vs1 = V_cone - V_cap

              scenario = '2within'
              volume_viewed = Vs1 + Vs2 + Vs3
          
          else:
            # SCENARIO 2C for within altitude
            delta = math.atan(R_m/r_mag)
            omega = (pi/2) - delta
            tau = math.acos(R_m / (R_m+A))
            gamma = pi - tau - omega

            alpha = math.asin(math.sin(phi)*r_mag / (R_m + A))
            theta = pi - alpha - phi
            beta = pi - theta - gamma
            
            V_bigcap = (pi/3)*(R_m+A)**3 *(2+math.cos(beta+gamma))*(1-math.cos(beta+gamma))**2
            V_smallcap = (pi/3)*(R_m+A)**3 * (2+math.cos(gamma))*(1-math.cos(gamma))**2

            L1 = (R_m + A)*math.sin(beta + gamma)
            d = (R_m + A)*math.cos(beta + gamma)
            a = (R_m + A)*math.sin(gamma)
            b = (r_mag + d)*math.tan(delta)
            h1 = (R_m+A)*math.cos(gamma) - d
            V_conical = (pi/3)*h1*(a**2 + a*b + b**2)

            Vs3 = V_bigcap - V_smallcap - V_conical

            V_bigcone = (pi/3)*(L1**2)*(r_mag+d)
            V_smallcone = (pi/3)*(b**2)*(r_mag+d)

            Vs2 = V_bigcone - V_smallcone

            L = R_m*math.sin(omega)
            h = r_mag - R_m*math.cos(omega)
            V_cone = (pi/3)*L**2*h
            V_cap = (pi/3)*(R_m)**3*(2+math.cos(omega))*(1-math.cos(omega))**2

            Vs1 = V_cone - V_cap

            scenario = '2within'
            volume_viewed = Vs1 + Vs2 + Vs3

      
      else:
        # OUTSIDE ALTITUDE
        if r_mag*math.sin(phi) < R_m:

          # scenario 1
          alpha_mfake = math.asin(math.sin(phi)*r_mag/R_m)
          alpha_afake = math.asin(math.sin(phi)*r_mag/(R_m+A))

          alpha_m = pi - alpha_mfake
          alpha_a = pi - alpha_afake

          theta_m = pi - phi - alpha_m
          theta_a = pi - phi - alpha_a

          V_cap_m = (pi/3)*(R_m)**3 * (2+math.cos(theta_m))*(1-math.cos(theta_m))**2
          V_cap_a = (pi/3)*(R_m+A)**3 * (2+math.cos(theta_a))*(1-math.cos(theta_a))**2

          a = R_m*math.sin(theta_m)
          b = (R_m + A)*math.sin(theta_a)

          h = (R_m+A)*math.cos(theta_a) - R_m*math.cos(theta_m)
          V_conical = (pi/3)*h*(a**2 + a*b + b**2)

          scenario = '1Outside'
          volume_viewed = V_cap_a + V_conical - V_cap_m

        

        elif (r_mag*math.sin(phi) < R_m + A):
          
          # scenario 2, in between case where view spans moon but not R_m+A
          # find spherical cap volume in front
          gamma = math.asin(R_m/r_mag)
          delta = (pi/2) - gamma
          V_front = (pi/3)*(R_m**3)*(2+math.cos(delta))*(1-math.cos(delta))**2

          # find cone section volume
          b = R_m*math.sin(delta)
          omega = math.acos(R_m/(R_m+A))
          alpha = pi - delta - omega
          a = (R_m+A)*math.sin(alpha)
          h = R_m*math.cos(delta) + (R_m+A)*math.cos(alpha)
          V_cone = (pi/3)*h*(a**2 + a*b + b**2)

          # find back-spherical cap volume
          V_back = (pi/3)*((R_m+A)**3)*(2+math.cos(alpha))*(1-math.cos(alpha))**2

          sigma = math.asin((r_mag*math.sin(phi)) / (R_m + A))
          sigma = pi - sigma # correction of alpha to account for arcsin
          tao = pi - phi - sigma
          epsilon = 2*sigma - pi

          if (tao + epsilon) > (pi/2):

            # find spherical cap information
            zeta = pi - tao - epsilon
            V_spherical = ((pi/3)*(R_m+A)**3)*(4-(2+math.cos(tao))*(1-math.cos(tao))**2 - (2+math.cos(zeta))*(1-math.cos(zeta))**2)

            # find partial cone information
            b = (R_m+A)*math.sin(tao)
            a = (R_m+A)*math.sin(zeta)
            h = (R_m+A)*(math.cos(tao)+math.cos(zeta))
            V_conical = (pi/3)*h*(a**2 + a*b + b**2)

            V_slit = V_spherical - V_conical # volume of space between cone section and sphere section

      
            scenario = '2outside'
            volume_viewed = Va - V_slit - V_front - V_cone - V_back



          else:
            # find spherical cap information

            V_spherical = ((pi/3)*(R_m+A)**3)*((2+math.cos(epsilon+tao))*(1-math.cos(epsilon+tao))**2 - (2+math.cos(tao))*(1-math.cos(tao))**2)

            # find cone information
            b = (R_m+A)*math.sin(tao)
            a = (R_m+A)*math.sin(epsilon+tao)
            h = (R_m+A)*(math.cos(tao) - math.cos(epsilon+tao))
            V_conical = (pi/3)*h*(a**2 + a*b + b**2)

            V_slit = V_spherical - V_conical # volume of space between cone section and sphere section

            scenario = '2outside'
            volume_viewed = Va - V_slit - V_front - V_cone - V_back

            

        else:
          
          # scenario 3, view spans entire moon and altittude
          gamma = math.asin(R_m/r_mag)
          delta = (pi/2) - gamma
          V_front = (pi/3)*(R_m**3)*(2+math.cos(delta))*(1-math.cos(delta))**2 # spherical cap volume in front

          # find cone section volume
          b = R_m*math.sin(delta)
          omega = math.acos(R_m/(R_m+A))
          alpha = pi - delta - omega
          a = (R_m+A)*math.sin(alpha)
          h = R_m*math.cos(delta) + (R_m+A)*math.cos(alpha)
          V_cone = (pi/3)*h*(a**2 + a*b + b**2)

          # find back-spherical cap volume
          V_back = (pi/3)*((R_m+A)**3)*(2+math.cos(alpha))*(1-math.cos(alpha))**2

          volume_viewed = Va - V_front - V_cone - V_back
      
      Volume = ((4/3)*pi)*((R_m+A)**3 - (R_m)**3)



      V.append(volume_viewed)
    
    Volume = ((4/3)*pi)*((R_m+A)**3 - (R_m)**3)
    V_percent_viewed = []

    for i in range(rows):
      V_percent_viewed.append(V[i] / Volume)
      #if i == 180:
        #print('Fraction of Volume: ' + str(V[i] / Volume))

    return time,V_percent_viewed,r_mag_vec
  
  def angular_diameter(self,angle,A):
    # volume solution using steradians
    # angle is the field of view
    # A is the altitude of orbits
    # ang_diam is a list containing the angular diameter of the moon + altitude at all points in time
    # ang_diam_frac is a list containing the fraction of the angular diameter of the moon + altitude you can see at any point in time

    state = self.orbit['state'] # gather position and velocity data
    dimensions = np.shape(state)
    rows, columns = dimensions # obtain size of state
    r_satellite = state[:,0:3]*1000
    time = self.orbit['t'] # get time
    phi = (angle) * math.pi / 180 / 2
    d = 2*(R_m + A) # diameter of moon + altitude

    fixed_diam = phi*2 # angular diameter of view phi*2
    ang_diam = []
    ang_diam_frac = []

    r_moon = []
    for i in range(rows):
      r_moon.append([m_position,0,0]) # list describing moon position

    r_moon = np.array(r_moon) # convert moon list to array
    r = r_satellite - r_moon #  array describing satellite distance from moon

    V = []
    r_mag_vec = []

    for i in range(0,rows):
      r_mag = (r[i,0]**2 + r[i,1]**2 + r[i,2]**2)**(1/2)
      a_diam = 2*math.asin(d/(2*r_mag))
      # a_diam = 2*np.pi*(1-np.sqrt(r_mag**2-d**2/4)/r_mag)
      a_diam_frac = fixed_diam / a_diam
      r_mag_vec.append(r_mag)
      ang_diam.append(a_diam)

      if a_diam <= fixed_diam: # object + altitude is in full view
        ang_diam_frac.append(1)
      else:
        ang_diam_frac.append(a_diam_frac)

    return time,ang_diam_frac
  
  def orbit_ranker(self,orbit_files,orbit_names,angle,A,metric):
    # orbit_files is a list of orbit pickle files
    # orbit_names is the names of the orbits
    # angle is the filed of view
    # A is the altitude
    # metric is the metric by which the orbits are evaluated, which should be a string that is the same as the name of the function

    avg = []

    for i in range(0,len(orbit_files)):

      if metric == "orbit_volume":
        orbit = orbit_files[i]
        [time,values,r_mag_vec] = orbit.orbit_volume(angle,A) # obtain volumes for metric throughout orbit

      elif metric == "angular_diameter":
        orbit = orbit_files[i]
        [time,values] = orbit.angular_diameter(angle,A) # obtain angular diameter for metric throughout orbit

      # convert time from pickle file to 1D array
      timevec = []
      for j in range(1,len(time)):
        timevec.append(time[j][0]) 
      time = timevec

      # obtain time vector of equally-spaced points
      total_points = math.ceil(time[len(time)-1] / (time[1] - time[0]))
      end_time = time[len(time)-1]
      time = np.array(time) 

      # create corresponding values for metric for each time using interpolation
      values = np.array(values[0:(len(values)-1)])
      tvals = np.linspace(0,end_time,total_points+10)
      yInt = np.interp(tvals,time,values)

      # find average of metric
      yInt = np.array(yInt)
      avg.append(np.mean(yInt))

    avg = np.array(avg) # convert list to numpy array

    # establish ranking (updated)
    indices = np.argsort(avg)
    ranking = np.empty(len(indices), dtype=object)

    for i in range(len(indices)):
      index = indices[i]
      ranking[i] = orbit_names[index]
      
    return avg,indices,ranking

print(" ")


DRO_11 = orbit_eval(DRO_11)
DRO_13 = orbit_eval(DRO_13)
L1_10 = orbit_eval(L1_10)
L1_13 = orbit_eval(L1_13)
L2_6 = orbit_eval(L2_6)
L2_12 = orbit_eval(L2_12)



A = 500000 # lunar altitude
angle = 2 # width of view in degrees
[time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A)
[time,ang_diam_frac] = DRO_11.angular_diameter(angle,A)

[time11,ang_diam_frac11] = DRO_11.angular_diameter(angle, A)
[time13,ang_diam_frac13] = DRO_13.angular_diameter(angle, A)
[time12,ang_diam_frac12] = L2_12.angular_diameter(angle, A)


orbits_files = [DRO_11,DRO_13,L2_12]
orbit_names = ["DRO_11","DRO_13","L2_12"]

# [avg,indices,ranking] = DRO_11.orbit_ranker(orbits_files,orbit_names,angle,A,"angular_diameter")
# print(avg)
# print(ranking)

plt.plot(time11,ang_diam_frac11,label = 'DRO11')
plt.plot(time13,ang_diam_frac13,label = 'DRO13')
plt.plot(time12,ang_diam_frac12,label = 'L2 12')
plt.xlabel("Time (s)")
plt.ylabel("Fraction of Angular Diameter")
plt.legend()
plt.show()

plt.plot(time,V_percent_viewed,label = 'Fraction of Volume')
plt.plot(time,ang_diam_frac,label = 'Fraction of Angular Diameter')

plt.xlabel("Time (s)")
plt.ylabel("Fraction")
plt.title("Spacecraft in Orbit Evaluation")
plt.legend() 

plt.show()
