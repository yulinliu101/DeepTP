# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-10-16 16:36:20
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-10-16 16:43:30

from matplotlib.patches import Polygon
import pyproj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle
from utils import g
# Borrow from https://stackoverflow.com/questions/8161144/drawing-ellipses-on-matplotlib-basemap-projections
class Basemap(Basemap):
    def ellipse(self, x0, y0, a, b, n, ax=None, **kwargs):
        """
        Draws a polygon centered at ``x0, y0``. The polygon approximates an
        ellipse on the surface of the Earth with semi-major-axis ``a`` and 
        semi-minor axis ``b`` degrees longitude and latitude, made up of 
        ``n`` vertices.

        For a description of the properties of ellipsis, please refer to [1].

        The polygon is based upon code written do plot Tissot's indicatrix
        found on the matplotlib mailing list at [2].

        Extra keyword ``ax`` can be used to override the default axis instance.

        Other \**kwargs passed on to matplotlib.patches.Polygon

        RETURNS
            poly : a maptplotlib.patches.Polygon object.

        REFERENCES
            [1] : http://en.wikipedia.org/wiki/Ellipse


        """
        ax = kwargs.pop('ax', None) or self._check_ax()
        g = pyproj.Geod(a=self.rmajor, b=self.rminor)
        # Gets forward and back azimuths, plus distances between initial
        # points (x0, y0)
        azf, azb, dist = g.inv([x0, x0], [y0, y0], [x0+a, x0], [y0, y0+b])
        tsid = dist[0] * dist[1] # a * b

        # Initializes list of segments, calculates \del azimuth, and goes on 
        # for every vertex
        seg = [self(x0+a, y0)]
        AZ = np.linspace(azf[0], 360. + azf[0], n)
        for i, az in enumerate(AZ):
            # Skips segments along equator (Geod can't handle equatorial arcs).
            if np.allclose(0., y0) and (np.allclose(90., az) or
                np.allclose(270., az)):
                continue

            # In polar coordinates, with the origin at the center of the 
            # ellipse and with the angular coordinate ``az`` measured from the
            # major axis, the ellipse's equation  is [1]:
            #
            #                           a * b
            # r(az) = ------------------------------------------
            #         ((b * cos(az))**2 + (a * sin(az))**2)**0.5
            #
            # Azymuth angle in radial coordinates and corrected for reference
            # angle.
            azr = 2. * np.pi / 360. * (az + 90.)
            A = dist[0] * np.sin(azr)
            B = dist[1] * np.cos(azr)
            r = tsid / (B**2. + A**2.)**0.5
            lon, lat, azb = g.fwd(x0, y0, az, r)
            x, y = self(lon, lat)

            # Add segment if it is in the map projection region.
            if x < 1e20 and y < 1e20:
                seg.append((x, y))

        poly = Polygon(seg, **kwargs)
        ax.add_patch(poly)

        # Set axes limits to fit map region.
        self.set_axes_limits(ax=ax)

        return poly
        
def get_cov_ellipse_wh(cov, nstd = 3):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][1], eigvecs[:,0][0]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return width, height, theta

def plot_fp_act(FP_ID, 
                IAH_BOS_FP_utilize, 
                IAH_BOS_ACT_track, 
                IAH_BOS_FP_track, 
                feed_track = None, 
                pred_track = None, 
                pred_track_mu = None,
                pred_track_cov = None,
                k = 9, 
                nstd = 3,
                sort = True):
    # TODO: plot error ellipse
    ori_lat = 29.98333333; ori_lon = -95.33333333
    des_lat = 42.36666667; des_lon = -71
    
    fig = plt.figure(figsize=(8,6))
    m = Basemap(llcrnrlon = -100,llcrnrlat = 27,urcrnrlon = -68,urcrnrlat = 46,projection='merc')
    m.drawmapboundary(fill_color='#8aeaff')
    m.fillcontinents(color='#c5c5c5', lake_color='#8aeaff')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.drawstates(linewidth=0.5)
    m.drawparallels(np.arange(10.,35.,5.))
    m.drawmeridians(np.arange(-120.,-80.,10.))
    
    x1, y1 = m(ori_lon, ori_lat)
    x2, y2 = m(des_lon, des_lat)
    plt.plot(x1,y1, 'r*', ms = 15, zorder = 3)
    plt.plot(x2,y2, 'r*', ms = 15, zorder = 3)

    fid_fp1 = IAH_BOS_FP_utilize.loc[IAH_BOS_FP_utilize.FLT_PLAN_ID == FP_ID, 'FID'].values
    print('%d flights filed flight plan %s'%(fid_fp1.shape[0], FP_ID))
    plot_track = IAH_BOS_ACT_track.loc[IAH_BOS_ACT_track.FID.isin(fid_fp1)]
    plot_fp = IAH_BOS_FP_track.loc[IAH_BOS_FP_track.FLT_PLAN_ID_REAL == FP_ID]
    x_fp, y_fp = m(plot_fp.LONGITUDE.values, plot_fp.LATITUDE.values)
    
    feed_x, feed_y = m(feed_track.Lon.values, feed_track.Lat.values)
    feed, = plt.plot(feed_x, feed_y, 'o-', ms = 4, linewidth = 3, color='g', label = 'feed tracks', zorder = 4)
    
    for gpidx, gp in plot_track.groupby('FID'):
        x,y = m(gp.Lon.values, gp.Lat.values)
        actual, = plt.plot(x,y,'--', linewidth = 2, color='b', label = 'Actual Tracks', zorder = 3)
    fp, = plt.plot(x_fp, y_fp, '-', linewidth = 2, color='r', label = 'Flight Plans', zorder = 1)
    
    if pred_track is not None:
        if sort:
            x, y = m(pred_track[k][pred_track[k][:,3].argsort()][:, 1], pred_track[k][pred_track[k][:,3].argsort()][:, 0])
        else:
            x, y = m(pred_track[k, :, 1], pred_track[k, :, 0])
        pred_fig, = plt.plot(x,y, 'o--', ms = 3, zorder = 4)
    if pred_track_mu is not None:
        if sort:
            x, y = m(pred_track_mu[k][pred_track_mu[k][:,3].argsort()][:, 1], 
                     pred_track_mu[k][pred_track_mu[k][:,3].argsort()][:, 0])
        else:
            x, y = m(pred_track_mu[k][:, 1], pred_track_mu[k][:, 0])
        plt.plot(x,y, 'mo--', ms = 4, zorder = 4)

    if pred_track_cov is not None:
        for t in range(pred_track_mu[k].shape[0]):
            lon_a = np.sqrt(pred_track_cov[k, t, 1, 1]) * nstd
            lat_b = np.sqrt(pred_track_cov[k, t, 0, 0]) * nstd
            # assume independency
#             cov_width, cov_height, cov_theta = get_cov_ellipse_wh(predicted_tracks_cov[k, t, :2, :2], nstd = 3)
            centre_lon, centre_lat = (pred_track_mu[k, t, 1], pred_track_mu[k, t, 0])
            poly = m.ellipse(centre_lon, 
                             centre_lat, 
                             lon_a,
                             lat_b, 
                             50, 
                             facecolor='green', zorder=4, alpha=0.25)

    plt.show()
    return plot_track, plot_fp