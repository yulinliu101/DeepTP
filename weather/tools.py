'''
Module's author : Jarry Gabriel
Date : June, July 2016
Some Algorithms was made by : Malivai Luce, Helene Piquet
This module handle different tools
'''

from pyproj import Proj, Geod
import numpy as np

# Projections
wgs84=Proj("+init=EPSG:4326")
epsg3857=Proj("+init=EPSG:3857")
g=Geod(ellps='WGS84')


# Returns pressure from altitude (ft)
def press(alt):
    z = alt/3.28084
    return 1013.25*(1-(0.0065*z)/288.15)**5.255

# Returns the closest lvl from levels with altitude (atl)
def proxilvl(alt , lvls):
    p = press(alt)
    levels = np.array(sorted(lvls.keys()))
    return levels[np.abs(levels - p).argmin()]
    
# def proxy(val, lvl1, lvl2):
#        if (abs(val - lvl1) < abs(val - lvl2)):
#            return lvl1
#        else:
#            return lvl2
#    p = press(alt)
#    levels = sorted(lvls.keys())
#    if p < levels[0]:
#        return levels[0]
#    else:
#        for i, el in enumerate(levels[1:]):
#            if p < el:
#                return proxy(p, levels[i-1], el)

#        return levels[-1]