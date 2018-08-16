import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import numpy as np

def plot_grb_data(fig,wdata,Nx,Ny,lat_0=38.5339273887,lon_0=261.81191642,lat_1=12.19,lon_1=-133.459,title='',unit='m/s',cmap=plt.get_cmap('RdYlGn')):
    # fig = plt.figure(figsize=(16,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # create polar stereographic Basemap instance.
    # http://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html for grid description
    m = Basemap(projection='stere',lon_0=lon_0,lat_0=lat_0,lon_1=lon_1,lat_1=lat_1,lat_ts=lat_0,\
                llcrnrlat=16.281,urcrnrlat=55.481,\
                llcrnrlon=-126.138,urcrnrlon=-57.383,\
                rsphere=6371200.,resolution='l',area_thresh=10000)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    # draw parallels.
    parallels = np.arange(0.,90,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(180.,360.,10.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    lons, lats = m.makegrid(Nx, Ny) # get lat/lons of Ny by Nx evenly space grid.
    x, y = m(lons, lats) # compute map proj coordinates.
    # draw filled contours.
    clevs = np.linspace(np.min(wdata), np.max(wdata), num=100)
    # cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)
    cs = m.contour(x,y,wdata,clevs,cmap=cmap)
    cs2 = m.contourf(x,y,wdata,clevs,cmap=cmap)
    # add colorbar.
    cbar = m.colorbar(cs2,location='bottom',pad="5%")
    cbar.set_label(unit)

    plt.title(title)
    return

def plot_msg(fig,msg,title,unit='m/s',cmap=plt.get_cmap('RdYlGn')):
    Nx = msg.Nx
    Ny = msg.Ny
    lat_1 = msg.latitudeOfFirstGridPointInDegrees
    lon_1 = msg.longitudeOfFirstGridPointInDegrees
    # lat_0 = 90.0#data.latitudeOfSouthernPoleInDegrees
    # lon_0 = 0.0#data.longitudeOfSouthernPoleInDegrees
    lat_0,lon_0 = (38.5339273887, 261.81191642)
    data = msg.values

    plot_grb_data(fig,data,Nx,Ny,lat_0,lon_0,lat_1,lon_1,title,unit,cmap)
    return
