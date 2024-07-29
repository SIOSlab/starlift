import numpy as np
import sys
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
import plot_tools
import gmatTools
import pdb


# ~~~~~PROPAGATE THE DYNAMICS~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de440')

# Parameters
t_mjd = Time(57727, format='mjd', scale='utc')
days = 30
days_can = unitConversion.convertTime_to_canonical(days * u.d)
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star
moon_r_can = 1-mu_star  # Radius of the Moon
moon_r = (unitConversion.convertPos_to_dim(moon_r_can)).to('AU')
earth_r_can = mu_star  # Radius of the Earth
earth_r = (unitConversion.convertPos_to_dim(earth_r_can)).to('AU')

# Initial condition in non dimensional units in rotating frame R [pos, vel, T/2]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period
# IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]  # DRO, 0.9697497-day period

# Convert the velocity to inertial from R
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar = np.array([IC[0], IC[2], vI[1], days_can])

# Propagate the dynamics in the CRTBP model
states, times = orbitEOMProp.statePropCRTBP(freeVar, mu_star)
pos = states[:, 0:3]
vel = states[:, 3:6]

breakpoint()

# Convert to AU
pos_au = unitConversion.convertPos_to_dim(pos).to('AU')


# ~~~~~PLOT SOLUTION AND GMAT FILE IN THE INERTIAL FRAME~~~~

# Obtain CRTBP data from GMAT
file_name = "gmatFiles/CRTBP_rot.txt"
gmat_km, gmat_time = gmatTools.extract_pos(file_name)
gmat_posrot = np.array((gmat_km * u.km).to('AU'))

# Convert to I frame from R frame
gmat_posinert = np.zeros([len(gmat_time), 3])
for ii in np.arange(len(gmat_time)):
    gmat_posinert[ii, :] = frameConversion.rot2inertP(gmat_posrot[ii, :], gmat_time[ii], gmat_time[0])

# Plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(pos_au[:, 0], pos_au[:, 1], pos_au[:, 2], color='blue', label='Propagated CRTBP')
ax.plot(moon_r*np.cos(times), moon_r*np.sin(times), 0, color='gray', label='Moon')
ax.plot(earth_r*np.cos(times), earth_r*np.sin(times), 0, color='green', label='Earth')
# ax.plot(gmat_posinert[:, 0], gmat_posinert[:, 1], gmat_posinert[:, 2], color='red', label='GMAT Orbit')
limit = 0.003
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
plt.title('CRTBP in the Inertial (I) Frame')
plt.legend()

# # Save
# plt.savefig('CRTBP L2.png')


# ~~~~~ANIMATION~~~~~

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Collect animation data
data_sc = np.array([pos_au[:, 0], pos_au[:, 1], pos_au[:, 2]])
data_Earth = np.array([earth_r*np.cos(times), earth_r*np.sin(times), np.zeros(len(times))])
data_Moon = np.array([moon_r*np.cos(times), moon_r*np.sin(times), np.zeros(len(times))])

# Initialize the first point for each body
line_sc, = ax.plot(data_sc[0, 0:1], data_sc[1, 0:1], data_sc[2, 0:1], color='blue', label='Orbit')
line_Earth, = ax.plot(data_Earth[0, 0:1], data_Earth[1, 0:1], data_Earth[2, 0:1], color='green', label='Earth')
line_Moon, = ax.plot(data_Moon[0, 0:1], data_Moon[1, 0:1], data_Moon[2, 0:1], color='gray', label='Moon')

interval = unitConversion.convertTime_to_canonical(days * u.d) / 100  # Fixed time interval for each frame


def next_frame(times, interval):
    t0 = 0
    idx = 1
    frame_indices = []
    while idx < len(times):
        diff = times[idx] - t0
        if diff >= interval:
            frame_indices.append(idx)
            t0 = times[idx]
        idx += 1
    return frame_indices


def animate(i):
    idx = frame_indices[i]
    if idx == 0:
        return
    line_sc.set_data(data_sc[0, :idx], data_sc[1, :idx])  # Set the x and y positions
    line_sc.set_3d_properties(data_sc[2, :idx])  # Set the z position
    line_Earth.set_data(data_Earth[0, :idx], data_Earth[1, :idx])
    line_Earth.set_3d_properties(data_Earth[2, :idx])
    line_Moon.set_data(data_Moon[0, :idx], data_Moon[1, :idx])
    line_Moon.set_3d_properties(data_Moon[2, :idx])


frame_indices = next_frame(times, interval)
ani = animation.FuncAnimation(fig, animate, frames=len(frame_indices), interval=1, repeat=True)

ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
plt.legend()
plt.title('CRTBP model in the I frame')

# # Save
# writergif = animation.PillowWriter(fps=30)
# ani.save('CRTBP L2.gif', writer=writergif)

plt.show()
