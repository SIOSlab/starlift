
import os.path
import numpy as np
import pickle
import math
from scipy.integrate import tplquad
from numpy import *
from matplotlib import pyplot as plt


# ~~~~RETRIEVE DATA FROM FILE~~~~

path_str = "orbitFiles/L2_S_6.0066_days.p"  # Change to whichever orbit you'd like
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")  # "rb" means "read binary"
orbit_data = pickle.load(f1)  # open the pickled file (and unpickle it)
f1.close()  # close the file

# ~~~~DEFINE CONSTANTS~~~~

barycenter_to_earth = -4641  # x coordinate [km]
barycenter_to_moon = 379764  # x coordinate [km]
earth_to_moon = 384405  # x coordinate [km]
earth_radius = 6378  # [km]
moon_radius = 1738  # [km]


class VisibilityMetric:

    def __init__(self, orbit):
        self.orbit = orbit  # Orbit is the UNPICKLED data

    def visible_altitude(self, y_fov, z_fov, altitude):
        # Calculates the percentage of the chosen altitude that is visible from the spacecraft's orbit.
        # Returns a vector (percent_visible) with that percentage at each orbital position.
        # y_fov is the field of view in the y direction, in degrees.
        # z_fov is the field of view in the z direction, in degrees.
        # altitude is the altitude from the moon that you want to observe, in kilometers.

        # ~~~~CALL THE COMPONENTS~~~~
        # state: position in km and velocity km/s stored as rows
        # t: times states are defined at in s
        # te: orbit period in s
        # x_lpoint: km from the Earth-Moon barycenter

        x_position = np.array(self.orbit["state"][:, 0])  # From barycenter
        # this calls the first element in the first array of 'state'
        y_position = np.array(self.orbit["state"][:, 1])
        z_position = np.array(self.orbit["state"][:, 2])
        x_velocity = np.array(self.orbit["state"][:, 3])
        y_velocity = np.array(self.orbit["state"][:, 4])
        z_velocity = np.array(self.orbit["state"][:, 5])
        L2_point = np.array(self.orbit["x_lpoint"])
        time = np.array(self.orbit["t"])

        alt_radius = altitude + moon_radius

        x_from_moon = [abs(x - barycenter_to_moon) for x in x_position]
        y_from_moon = y_position  # [km]
        z_from_moon = z_position  # [km]
        sc_to_moon = []  # Magnitude of the distance between the spacecraft and the moon

        i = 0
        for x in x_from_moon:
            sc_to_moon.append(np.sqrt(x_from_moon[i] ** 2 + y_from_moon[i] ** 2 + z_from_moon[i] ** 2))  # [km]
            i = i + 1
        sc_to_moon = np.array(sc_to_moon)

        # ~~~~VOLUME OF THE MOON~~~~
        # (Not really necessary, just for practice...)

        r1_moon, r2_moon = 0, moon_radius
        theta1_moon, theta2_moon = 0, 2*np.pi
        phi1_moon, phi2_moon = 0, np.pi

        def diff_volume1(phi, theta, r):
            return (r**2)*sin(phi)

        volume_moon = tplquad(diff_volume1, r1_moon, r2_moon, lambda r: theta1_moon, lambda r: theta2_moon,
                              lambda r, theta: phi1_moon, lambda r, theta: phi2_moon)[0]

        # ~~~~VOLUME OF THE DESIRED ALTITUDE (no moon included)~~~~

        r1_alt, r2_alt = 0, alt_radius
        theta1_alt, theta2_alt = 0, 2*np.pi
        phi1_alt, phi2_alt = 0, np.pi

        volume_alt = (tplquad(diff_volume1, r1_alt, r2_alt, lambda r: theta1_alt, lambda r: theta2_alt,
                              lambda r, theta: phi1_alt, lambda r, theta: phi2_alt)[0]) - volume_moon

        # ~~~~VOLUME OF DESIRED ALTITUDE~~~~

        target_volume = []
        i = 0
        for x in sc_to_moon:
            if sc_to_moon[i] < alt_radius:
                x1, x2 = 0, abs(sc_to_moon[i] - moon_radius)
            else:
                x1, x2 = abs(sc_to_moon[i] - alt_radius), abs(sc_to_moon[i] - moon_radius)
            y1, y2 = lambda x: -np.tan(math.radians(y_fov / 2)) * x, lambda x: np.tan(math.radians(y_fov / 2)) * x
            z1, z2 = lambda x, y: -np.tan(math.radians(z_fov / 2)) * x, lambda x, y: np.tan(math.radians(z_fov / 2)) * x
            target_volume.append(tplquad(lambda x, y, z: 1, x1, x2, y1, y2, z1, z2)[0])  # [km]
            i = i+1
        target_volume = np.array(target_volume)
        # The second argument in the result is an estimate of the error

        # ~~~~PERCENT VISIBLE~~~~
        # Percent of the desired altitude that is visible from the spacecraft

        percent_visible = (target_volume/volume_alt)*100
        plt.plot(time, percent_visible)
        plt.ylabel('Percentage of the altitude that is visible from the spacecraft')
        plt.xlabel('Time [s]')
        plt.show()

    def plot_fov(self, y_fov, z_fov, altitude, i):
        # Plots the moon, the target altitude, the orbit, and the field of view that the orbit sees.
        # i is the position of the orbit that you want to plot the field of view from.

        # ~~~~CALL THE COMPONENTS~~~~
        # state: position in km and velocity km/s stored as rows
        # t: times states are defined at in s
        # te: orbit period in s
        # x_lpoint: km from the Earth-Moon barycenter

        x_position = np.array(self.orbit["state"][:, 0])  # From barycenter
        # this calls the first element in the first array of 'state'
        y_position = np.array(self.orbit["state"][:, 1])
        z_position = np.array(self.orbit["state"][:, 2])
        x_velocity = np.array(self.orbit["state"][:, 3])
        y_velocity = np.array(self.orbit["state"][:, 4])
        z_velocity = np.array(self.orbit["state"][:, 5])
        L2_point = np.array(self.orbit["x_lpoint"])
        time = np.array(self.orbit["t"])

        alt_radius = altitude + moon_radius

        x_from_moon = [abs(x - barycenter_to_moon) for x in x_position]
        y_from_moon = y_position  # [km]
        z_from_moon = z_position  # [km]
        sc_to_moon = []  # Magnitude of the distance between the spacecraft and the moon

        k = 0
        for x in x_from_moon:
            sc_to_moon.append(np.sqrt(x_from_moon[k] ** 2 + y_from_moon[k] ** 2 + z_from_moon[k] ** 2))  # [km]
            k = k + 1
        sc_to_moon = np.array(sc_to_moon)

        # Set up the figure
        fig1 = plt.figure(1)
        ax3D = plt.axes(projection='3d')  # Call a 3D plot
        ax3D.set_xlabel('X Position [km]')  # Label axes
        ax3D.set_ylabel('Y Position [km]')
        ax3D.set_zlabel('Z Position [km]')

        """
        # Limits for DRO
        ax3D.axes.set_xlim3d(left=300000, right=500000)
        ax3D.axes.set_ylim3d(bottom=-100000, top=100000)
        ax3D.axes.set_zlim3d(bottom=-100000, top=100000)
        
        # Limit for moon only
        ax3D.axes.set_xlim3d(left=375000, right=385000)
        ax3D.axes.set_ylim3d(bottom=-5000, top=5000)
        ax3D.axes.set_zlim3d(bottom=-5000, top=5000)
        """

        # Plot the orbit
        ax3D.scatter3D(x_position, y_position, z_position, s=5)  # Plot the orbit

        # Plot the surface of the moon
        phi = np.linspace(0, 2*np.pi, 500)  # Angle phi
        theta = np.linspace(0, np.pi, 500)  # Angle theta

        x_moon = barycenter_to_moon + (moon_radius * np.outer(np.cos(phi), np.sin(theta)))
        y_moon = moon_radius * np.outer(np.sin(phi), np.sin(theta))
        z_moon = moon_radius * np.outer(np.ones(np.size(phi)), np.cos(theta))

        ax3D.plot_surface(x_moon, y_moon, z_moon, color='lightgreen')

        # Plot the surface of the desired altitude
        x_alt = barycenter_to_moon + (alt_radius * np.outer(np.cos(phi), np.sin(theta)))
        y_alt = alt_radius * np.outer(np.sin(phi), np.sin(theta))
        z_alt = alt_radius * np.outer(np.ones(np.size(phi)), np.cos(theta))

        ax3D.plot_surface(x_alt, y_alt, z_alt, alpha=0.1)

        # Plot the field of view from the spacecraft
        # Currently for 1 orbital position at a time
        # NOTE: This section doesn't work quite yet, and is still being debugged...a separate version that utilizes
        # matrices is currently being iterated in Matlab.

        y_line1 = []
        y_line2 = []
        z_line1 = []
        z_line2 = []

        x_fov1 = []  # Final rotated vector
        x_fov2 = []
        x_fov12 = []
        x_fov21 = []
        y_fov1 = []
        y_fov2 = []
        z_fov1 = []
        z_fov2 = []

        # i = 50  # Position index (of the orbit)

        x_line = np.linspace(x_from_moon[i], barycenter_to_moon, num=len(x_from_moon))  # Line from SC to moon
        j = 0  # Index of each point in the field of view vectors

        for x in x_line:  # For each element in the x position vector, calculate the y and z vectors

            # Non-rotated vectors, pointed towards the negative x direction
            y_line1.append((np.tan(math.radians(y_fov / 2)) * (x_line[j]-x_position[i])) + y_position[i])
            y_line2.append((-np.tan(math.radians(y_fov / 2)) * (x_line[j]-x_position[i])) + y_position[i])
            z_line1.append((np.tan(math.radians(z_fov / 2)) * (x_line[j]-x_position[i])) + z_position[i])
            z_line2.append((-np.tan(math.radians(z_fov / 2)) * (x_line[j]-x_position[i])) + z_position[i])

            # Angles to rotate the vectors
            theta_fov = np.arctan(z_from_moon[i]/x_from_moon[i])  # Rotation about the y-axis [rad]
            phi_fov = np.arctan(y_from_moon[i]/x_from_moon[i])  # Rotation about the z-axis [rad]

            # Rotate the x vectors
            x_temp1 = (((x_line[j]-x_position[i]) * np.cos(theta_fov)) - ((z_line1[j]-z_position[i]) * np.sin(theta_fov))
                       + x_position[i])  # First rotation, about the y-axis
            x_fov1.append(((x_temp1-x_position[i]) * np.cos(phi_fov)) - ((y_line1[j]-y_position[i]) * np.sin(phi_fov))
                          + x_position[i])  # Second rotation, about the z-axis

            x_temp2 = (((x_line[j]-x_position[i]) * np.cos(theta_fov)) - ((z_line2[j]-z_position[i]) * np.sin(theta_fov))
                       + x_position[i])
            x_fov2.append(((x_temp2-x_position[i]) * np.cos(phi_fov)) - ((y_line2[j]-y_position[i]) * np.sin(phi_fov))
                          + x_position[i])

            x_temp12 = (((x_line[j]-x_position[i]) * np.cos(theta_fov)) - ((z_line2[j]-z_position[i]) * np.sin(theta_fov))
                        + x_position[i])
            x_fov12.append(((x_temp12-x_position[i]) * np.cos(phi_fov)) - ((y_line1[j]-y_position[i]) * np.sin(phi_fov))
                           + x_position[i])

            x_temp21 = (((x_line[j]-x_position[i]) * np.cos(theta_fov)) - ((z_line1[j]-z_position[i]) * np.sin(theta_fov))
                        + x_position[i])
            x_fov21.append(((x_temp21-x_position[i]) * np.cos(phi_fov)) - ((y_line2[j]-y_position[i]) * np.sin(phi_fov))
                           + x_position[i])

            # Rotate the z vectors (only rotation about y-axis)
            z_fov1.append(((x_line[j]-x_position[i]) * np.sin(theta_fov)) + ((z_line1[j]-z_position[i]) * np.cos(theta_fov))
                          + z_position[i])  # Rotate z vector 1 (positive)
            z_fov2.append(((x_line[j]-x_position[i]) * np.sin(theta_fov)) + ((z_line2[j]-z_position[i]) * np.cos(theta_fov))
                          + z_position[i])  # Rotate z vector 2 (negative)

            # Rotate the y vectors (only rotation about z-axis)
            y_fov1.append(((x_line[j] - x_position[i]) * np.sin(phi_fov)) + ((y_line1[j] - y_position[i]) * np.cos(phi_fov))
                          + y_position[i])  # Rotate y vector 1
            y_fov2.append(((x_line[j] - x_position[i]) * np.sin(phi_fov)) + ((y_line2[j] - y_position[i]) * np.cos(phi_fov))
                          + y_position[i])  # Rotate y vector 2

            j = j+1

        ax3D.plot(x_fov1, y_fov1, z_fov1, linewidth=2, color='blue')
        ax3D.plot(x_fov2, y_fov2, z_fov2, linewidth=2, color='red')
        ax3D.plot(x_fov21, y_fov2, z_fov1, linewidth=2, color='green')
        ax3D.plot(x_fov12, y_fov1, z_fov2, linewidth=2, color='purple')

        # Show the plot
        plt.show()


orbit = VisibilityMetric(orbit_data)  # Creates an object in the class

y_fov = 1
z_fov = 1
altitude = 560

orbit.visible_altitude(y_fov, z_fov, altitude)

orbital_position_index = 1
orbit.plot_fov(y_fov, z_fov, altitude, orbital_position_index)
