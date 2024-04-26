# tests volume case
import sys
sys.path.insert(1, sys.path[0][0:-10])
print(sys.path)
from metricTests import Starlift_metrics

import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
from scipy import integrate as int
import unittest


path_str ="orbitFiles/DRO_11.241_days.p"   
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")
DRO_11 = pickle.load(f1)
f1.close()

DRO_11 = Starlift_metrics.orbit_eval(DRO_11)

A = 20000000 # lunar altitude
angle = 42 # width of view in degrees
[time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A)

def testVolume():
    # tests that the volumes outputted from orbit_volume are correct 
    # based on SolidWorks cross-references for possible scenarios
    # scenarios are:
    # collides
    # back altitude
    # front altitude
    # full view

    passes = 0
    angle = 2
    A = 500000
    [time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A)
    if V_percent_viewed[1] > 0.066 and V_percent_viewed[1] < 0.0665:
        print("Collides scenario passed")
        passes = passes + 1
    else:
        print("Collides scenario failed")

    # back altitude scenario
    angle = 3.8
    A = 500000
    [time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A)
    if V_percent_viewed[1] > 0.485 and V_percent_viewed[1] < 0.486:
        print("back altitude scenario passed")
        passes = passes + 1
    else:
        print("back altitude scenario failed")

    # front altitude scenario
    angle = 42
    A = 20000000
    [time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A)
    if V_percent_viewed[1] > 0.974 and V_percent_viewed[1] < 0.976:
        print("front altitude scenario passed")
        passes = passes + 1
    else:
        print("front altitude scenario failed")

    # full view scenario
    angle = 6
    A = 500000
    [time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A)
    if V_percent_viewed[1] > 0.721 and V_percent_viewed[1] < 0.723:
        print("full view scenario passed")
        passes = passes + 1
    else:
        print("full view scenario failed")

    if passes == 4:
        print("ALL SCENARIOS PASSED")

testVolume()

    
            
