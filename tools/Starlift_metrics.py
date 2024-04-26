import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
from scipy import integrate as int

path_str ="orbitFiles/DRO_11.241_days.p"   # change this
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
retrograde = pickle.load(f1)
f1.close()

path_str ="orbitFiles/L2_S_6.0066_days.p"   # change this
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
halo = pickle.load(f1)
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

  def view_eval(self,angle,A):
    # evaluates how well the spacecraft views volume of moon below certain orbit
    # angle is the scope of the telescope in degrees (ex: 2 deg by 2 deg)
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

halo = orbit_eval(retrograde)
A = 300000 # lunar altitude
angle = 2 # width of view in degrees
[time,V_percent_viewed,r_mag_vec] = halo.view_eval(angle,A)

plt.plot(time,V_percent_viewed)
plt.xlabel("Time (s)")
plt.ylabel("Fraction of Volume Viewed")
plt.title("Fraction of Available Volume Viewed by Spacecraft in Halo Orbit")

plt.show()

# plotting distance from moon center over time

plt.plot(time,r_mag_vec)
plt.axhline(y=1737400, color='r', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Distance from Moon's Center (m)")
plt.title("Spacecraft in Retrograde Orbit - Distance from Moon Center")

plt.show()