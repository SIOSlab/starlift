import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import GCRS, ICRS

# From JPL Horizons
# TU = 27.321582 d
# DU = 384400 km
# m_moon = 7.349x10^22 kg
# m_earth = 5.97219x10^24 kg


# =============================================================================
# Unit conversions
# =============================================================================

# converting times
def convertTime_to_canonical(self,dimTime):
    """Convert array of times from dimensional units to canonical units
    
    Method converts the times inside the array from the given dimensional
    unit (doesn't matter which, it converts to units of days in an
    intermediate step) into canonical units of the CR3BP. 1 month = 2 pi TU
    where TU are the canonical time units.
    
    Args:
        dimTime (float n array):
            Array of times in some time unit

    Returns:
        canonicalTime (float n array):
            Array of times in canonical units
    """
    dimTime = dimTime.to('day')/27.321582
    canonicalTime = dimTime.value * (2*np.pi)
    
    return canonicalTime

def convertTime_to_dim(self,canonicalTime):
    """Convert array of times from canonical units to unit of years
    
    Method converts the times inside the array from canonical units of the
    CR3BP into year units. 1 month = 2 pi TU where TU are the canonical time
    units.
    
    Args:
        canonicalTime (float n array):
            Array of times in canonical units

    Returns:
        dimTime (float n array):
            Array of times in units of days
    """
    
    canonicalTime = canonicalTime / (2*np.pi)
    dimTime = canonicalTime * u.day * 27.321582
    
    return dimTime

# converting distances
def convertPos_to_canonical(self,dimPos):
    """Convert array of positions from dimensional units to canonical units
    
    Method converts the positions inside the array from the given dimensional
    unit (doesn't matter which, it converts to units of AU in an
    intermediate step) into canonical units of the CR3BP. (3.844000E+5*u.km).to('m') = 1 DU
    where DU are the canonical position units.
    
    Args:
        dimPos (float n array):
            Array of positions in some distance unit

    Returns:
        canonicalPos (float n array):
            Array of distance in canonical units
    """
    
    dimPos = dimPos.to('m')
    DU2m = (3.844000E+5*u.km).to('m')
    canonicalPos = (dimPos/DU2m).value
    
    return canonicalPos

def convertPos_to_dim(self,canonicalPos):
    """Convert array of positions from canonical units to dimensional units
    
    Method converts the positions inside the array from canonical units of
    the CR3BP into units of AU. (3.844000E+5*u.km).to('m') = 1 DU
    
    Args:
        canonicalPos (float n array):
            Array of distance in canonical units

    Returns:
        dimPos (float n array):
            Array of positions in units of m
    """
    DU2m = (3.844000E+5*u.km).to('m')
    dimPos = canonicalPos * DU2m
    
    return dimPos

# converting velocity
def convertVel_to_canonical(self,dimVel):
    """Convert array of velocities from dimensional units to canonical units
    
    Method converts the velocities inside the array from the given dimensional
    unit (doesn't matter which, it converts to units of AU/yr in an
    intermediate step) into canonical units of the CR3BP.
    
    Args:
        dimVel (float n array):
            Array of velocities in some speed unit

    Returns:
        canonicalVel (float n array):
            Array of velocities in canonical units
    """
    
    dimVel = dimVel.to('m/d')
    DU2m = (3.844000E+5*u.km).to('m')
    TU2d = 27.321582*u.day
    canonicalVel = (dimVel/DU2m*TU2d).value / (2*np.pi)
    
    return canonicalVel

def convertVel_to_dim(self,canonicalVel):
    """Convert array of velocities from canonical units to dimensional units
    
    Method converts the velocities inside the array from canonical units of
    the CR3BP into units of m/s.
    
    Args:
        canonicalVel (float n array):
            Array of velocities in canonical units

    Returns:
        dimVel (float n array):
            Array of velocities in units of m/s
    """
    
    DU2m = (3.844000E+5*u.km).to('m')
    TU2d = 27.321582*u.day
    canonicalVel = canonicalVel * (2*np.pi)
    dimVel = canonicalVel * DU2m/TU2d
    
    return dimVel
