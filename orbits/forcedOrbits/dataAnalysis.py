import numpy as np
import sys
import os
import astropy.units as u
import astropy.constants as const
from matplotlib import pyplot as plt

# automate saving this info

Fts = np.array([50.115, 50.125, 50.25, 50.5, 51, 55, 60, 70, 90, 107])*u.mN
t_burns = [1.652*u.d, 14.359*u.hr, 1.656*u.hr, 35.893*u.min, 15.759*u.min, 2.872*u.min, 1.42*u.min, 42.37*u.s, 21.127*u.s, 14.814*u.s]
pos_errors = np.array([2557.914, 957.654, 152.415, 93.497, 73.374, 64.429, 63.568, 48.008, 47.727, 47.765])*u.km

t_burns_plt = np.zeros(len(t_burns))
for ii in np.arange(len(t_burns)):
    t_burns_plt[ii] = t_burns[ii].to_value(u.s)

plt.figure(1)
plt.yscale('log')
plt.plot(Fts.value, t_burns_plt)
plt.xlabel('Force [mN]')
plt.ylabel('Total Burn Time [s]')

plt.figure(2)
plt.yscale('log')
plt.plot(Fts.value, pos_errors.value)
plt.xlabel('Force [mN]')
plt.ylabel('Max Position Error [km]')

plt.show()
breakpoint()
