import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
import pdb


# Parameters
mu_star = 1.215059*10**(-2)
