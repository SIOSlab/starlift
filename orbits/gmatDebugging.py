import numpy as np
import astropy.units as u
import sys
from astropy.time import Time
from astropy.coordinates.solar_system import get_body_barycentric_posvel
sys.path.insert(1, 'tools')
import gmatTools
import frameConversion
import plot_tools
import unitConversion

# ~~~~~MOON~~~~~

file_name = "gmatFiles/DebugICRF.txt"  # Centered at Earth
moon_icrf, times = gmatTools.extract_pos(file_name)  # Array in km
moon_icrf = np.array((moon_icrf * u.km).to('AU'))  # Array in AU

file_name = "gmatFiles/DebugRot.txt"
moon_rot, times = gmatTools.extract_pos(file_name)
moon_rot = np.array((moon_rot * u.km).to('AU'))  # Array in AU

file_name = "gmatFiles/DebugH.txt"
moonpos_H, moonvel_H, times_H = gmatTools.extract_posvel(file_name)  # In km, km/s, ModJulian
moonpos_H = (moonpos_H*u.km).to('AU')
moonpos_H_can = unitConversion.convertPos_to_canonical(moonpos_H)
moonvel_H_can = unitConversion.convertVel_to_canonical(moonvel_H*(u.km/u.s))

# Convert H frame to I frame
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d
C_I2G = frameConversion.inert2geo(times_H[0], t_veq)
C_G2I = C_I2G.T
moonpos_I = np.zeros([len(times_H), 3])
moonvel_I = np.zeros([len(times_H), 3])

for ii in np.arange(len(times_H)):
     # Array in AU and AU/day
     moonpos_I[ii, :], moonvel_I[ii, :] = frameConversion.convertSC_H2I(moonpos_H_can[ii, :], moonvel_H_can[ii, :], times_H[ii], C_I2G)

# Convert to I frame
moon_i = np.zeros([len(times), 3])
for ii in np.arange(len(times)):
     C_I2R = frameConversion.inert2rot(times[ii], times[0])
     C_R2I = C_I2R.T
     moon_i[ii, :] = C_R2I @ moon_rot[ii, :]

title = 'Moon'
body_names = ['EarthICRF', 'Rotating', 'Moon H frame', 'Converted to I from H', 'Converted to I from R']
fig, ax = plot_tools.plot_bodies(moon_icrf, moon_rot, np.array(moonpos_H), moonpos_I, moon_i, body_names=body_names, title=title)

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
