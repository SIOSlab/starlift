import numpy as np
import astropy.units as u
import sys
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.coordinates.solar_system import get_body_barycentric_posvel
sys.path.insert(1, 'tools')
import gmatTools
import frameConversion
import plot_tools
import unitConversion

t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d  # + 1*u.yr/4
t_start = Time(57727, format='mjd', scale='utc')
C_I2G = frameConversion.inert2geo(t_start, t_veq)

# ~~~~~MOON~~~~~

# Get GMAT data
file_name = "gmatFiles/Moon_EMInert.txt"  # MJ2000Eq centered at EM barycenter
moon_inert, times = gmatTools.extract_pos(file_name)  # Array in km
moon_inert = np.array((moon_inert * u.km).to('AU'))  # Array in AU

file_name = "gmatFiles/Moon_EMRot.txt"  # Rotating with the moon centered at EM barycenter
moon_rot, _ = gmatTools.extract_pos(file_name)  # Array in km
moon_rot = np.array((moon_rot * u.km).to('AU'))  # Array in AU

file_name = "gmatFiles/Moon_GMEc.txt"  # Rotating with the moon centered at EM barycenter
moon_gmec, _ = gmatTools.extract_pos(file_name)  # Array in km
moon_gmec = np.array((moon_gmec * u.km).to('AU'))  # Array in AU

# Compare GMAT GMEc with astropy GMEc (conclusion: not the same)
moon_astro_gmec = np.zeros([len(times), 3])
moon_astro_inert = np.zeros([len(times), 3])
for ii in np.arange(len(times)):
    moon_astro_icrs = get_body_barycentric_posvel('Moon', times[ii])[0].get_xyz().to('AU').value
    moon_astro_gmec[ii, :] = (frameConversion.icrs2gmec(moon_astro_icrs * u.AU, times[ii])).to('AU')
    _, _, moon_astro_inert[ii, :] = frameConversion.getSunEarthMoon(times[ii], C_I2G)

# Convert rot to inert 2 different ways
moon_inert_frameconvert = np.zeros([len(times), 3])
moon_inert_manual = np.zeros([len(times), 3])
for ii in np.arange(len(times)):
    # Convert GMAT rot to inert using Python function
    C_I2R_astro = frameConversion.inert2rot(times[ii], times[0])
    C_R2I_astro = C_I2R_astro.T
    moon_inert_frameconvert[ii, :] = C_R2I_astro @ moon_rot[ii, :]

    # DCM in GMAT
    angle = np.arccos(np.dot(moon_rot[0], moon_rot[ii])/(np.linalg.norm(moon_rot[0])*np.linalg.norm(moon_rot[ii])))
    C_I2R_gmat = frameConversion.rot(angle, 3)
    C_R2I_gmat = C_I2R_gmat.T
    moon_inert_manual[ii, :] = C_R2I_gmat @ moon_rot[ii, :]


    # # Get the difference
    # DCM_diff = C_I2R_astro - C_I2R_gmat
    # plt.scatter(times[ii].value, DCM_diff[0, 0])
    # plt.scatter(times[ii].value, DCM_diff[0, 1])
    # plt.scatter(times[ii].value, DCM_diff[1, 0])
    # plt.scatter(times[ii].value, DCM_diff[1, 1])


title = 'Moon'
body_names = ['Raw gmat gmec', 'Astropy gmec']
fig, ax = plot_tools.plot_bodies(moon_gmec, moon_astro_gmec, body_names=body_names, title=title)

breakpoint()


# # ~~~~~SPACECRAFT~~~~~
#
# file_name = "gmatFiles/DebugSC_ICRF.txt"
# sc_icrf, times_icrf = gmatTools.extract_pos(file_name)  # Array in km
# sc_icrf = np.array((sc_icrf * u.km).to('AU'))  # Array in AU
#
# file_name = "gmatFiles/DebugSC_Rot.txt"
# sc_rot, times_rot = gmatTools.extract_pos(file_name)
# sc_rot = np.array((sc_rot * u.km).to('AU'))  # Array in AU
#
# file_name = "gmatFiles/DebugSC_H.txt"
# pos_H, vel_H, times_H = gmatTools.extract_posvel(file_name)  # In km, km/s, ModJulian
# pos_H = unitConversion.convertPos_to_canonical(pos_H*u.km)
# vel_H = unitConversion.convertVel_to_canonical(vel_H*(u.km/u.s))
#
# # Convert H frame to I frame
# t_equinox = Time(51544.5, format='mjd', scale='utc')
# t_veq = t_equinox + 79.3125*u.d
# C_I2G = frameConversion.inert2geo(times_H[0], t_veq)
# sc_I = np.zeros([len(times_H), 3])
# vel_I = np.zeros([len(times_H), 3])
#
# for ii in np.arange(len(times_H)):
#      # Array in AU and AU/day
#      sc_I[ii, :], vel_I[ii, :] = frameConversion.convertSC_H2I(pos_H[ii, :], vel_H[ii, :], times_H[ii], C_I2G)
#
# # # Convert rotating frame to I frame
# # sc_I = np.zeros([len(times_rot), 3])
# # for ii in np.arange(len(times_rot)):
# #      C_I2R = frameConversion.inert2rot(times_rot[ii], times_rot[0])
# #      C_R2I = C_I2R.T
# #      sc_I[ii, :] = C_R2I @ sc_rot[ii, :]
#
# title = 'Spacecraft'
# body_names = ['ICRF', 'Rotating', 'Converted to I from H']
# fig, ax = plot_tools.plot_bodies(sc_icrf, sc_rot, sc_I, body_names=body_names, title=title)


# # ~~~~~FULL FORCE~~~~~
#
# file_name = "gmatFiles/FF_ICRF.txt"
# FF_icrf, _ = gmatTools.extract_pos(file_name)
#
# file_name = "gmatFiles/FF_rot.txt"
# FF_rot, times = gmatTools.extract_pos(file_name)
#
# # Convert to I frame
# FF_i = np.zeros([len(times), 3])
# for ii in np.arange(len(times)):
#      C_I2R = frameConversion.inert2rot(times[ii], times[0])
#      C_R2I = C_I2R.T
#      FF_i[ii, :] = C_R2I @ FF_rot[ii, :]
#
# title = 'FF'
# body_names = ['ICRF', 'Rotating', 'Converted to I']
# fig, ax = plot_tools.plot_bodies(FF_icrf, FF_rot, FF_i, body_names=body_names, title=title)
