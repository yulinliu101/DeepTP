# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-08-17 15:17:56
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-08-31 15:52:26

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patch
from descartes.patch import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint

def plot_daily_wx(nested_polygon_list, **kwargs):
    plt.figure(figsize=(10,6))
    m = Basemap(llcrnrlon = -128,llcrnrlat = 22.,urcrnrlon = -63,urcrnrlat = 52,projection='merc')
    m.drawmapboundary(fill_color='#8aeaff')
    m.bluemarble()
    m.fillcontinents(color='#c5c5c5', lake_color='#8aeaff')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.drawstates(linewidth=0.1)


    feature_grid = kwargs.get('feature_grid', None)
    flight_tracks_arr = kwargs.get('flight_tracks_arr', None)
    if feature_grid is None:
        pass
    else:
        xgrid, ygrid = m(feature_grid[..., 0], feature_grid[..., 1])
        _ = plt.plot(xgrid, ygrid, 'o', ms = 0.1, color = 'r')

    if flight_tracks_arr is None:
        pass
    else:
        xtrack, y_track = m(flight_tracks_arr[..., 0], flight_tracks_arr[..., 1])
        _ = plt.plot(xtrack, y_track, 'o-', ms = 5, color = 'b')
    
    bad_poly = []
    i = -1
    for hr_poly in nested_polygon_list:
        i += 1
        j = -1
        for mpoly in hr_poly:
            j += 1
            if mpoly.geom_type is 'MultiPolygon':
                k = -1
                for poly in mpoly:
                    k += 1
                    try:
                        x,y = m(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1])
                        xy = list(zip(x, y))
                        de_Poly = plt_patch.Polygon( xy, facecolor='yellow', alpha=0.9)
                        plt.gca().add_patch(de_Poly)
                    except Exception as err:
                        print(i,j,k)
                        print('MultiPolygon:%s, location: (%d, %d, %d)'%(err, i, j, k))
                        bad_poly.append(poly)
                        pass
            elif mpoly.geom_type is 'Polygon':
                try:
                    x,y = m(mpoly.exterior.coords.xy[0], mpoly.exterior.coords.xy[1])
                    xy = list(zip(x, y))
                    de_Poly = plt_patch.Polygon( xy, facecolor='yellow', alpha=0.9)
                    plt.gca().add_patch(de_Poly)
                except Exception as err:
                    print('Polygon:%s, location: (%d, %d)'%(err,i,j))
                    bad_poly.append(poly)
                    pass
            else:
                print(mpoly.geom_type)
    plt.title('NCWF convective weather polygons')
    return bad_poly