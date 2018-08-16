# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:39:24 2015

@author: Emmanuel Boidot
"""

import traceback
import sys
import os
import subprocess as commands

import pygrib

VERBOSE = 0

DATADIR = 'data'
WINDDATADIR='winddata'
RADARDATADIR='raddata'

path = '/media/liuyulin101/YulinLiu/WindData'

WEATHERDATADIR = path + '/DATA/weather_data'
FILTEREDWEATHERDATADIR = path + '/DATA/filtered_weather_data'
RAPDATADIR=WEATHERDATADIR+'/rap'
NAMDATADIR=WEATHERDATADIR+'/nam'
NAMANLDATADIR=WEATHERDATADIR+'/namanl'
GFSDATADIR=WEATHERDATADIR+'/gfs'

def get_fname(dsrc,depTime,mforecast):
    """
    Returns the name of the grib file on the NCDC server at the time requested
    
    Args:
        * 'dsrc' (string): type of source files for weather: either namanl or rap
        * 'DT (string): date and time of the grib file requested
        * 'mforecast' (string): time horizon of the weather forecast requested
    
    Outputs:
        * 'fname' (string): name of the requested grib file on the NCDC server
    """
    mdate = depTime[0:8]
    mtime = depTime[8:12]
    fname = dsrc + '/'
    fname = fname + ('rap_130_' if dsrc=='rap' else ('nam_218_' if dsrc=='nam' else 'namanl_218_'))
    fname = fname + mdate +'_'+mtime+'_'+mforecast
    fname = fname + ('.grb2' if dsrc=='rap' else '.grb')
    return fname

def get_data_at_lonlat(grb,lon, lat, dlat = 0.05, dlon = 0.05):
    """
    Decodes the grib message at the position specified

    
    Args:
        * 'grb' (grib object): the grib message to decode
        * 'lon' (float): lontitude of the point in degrees
        * 'lat' (float): latitude of the point in degrees
        * 'dl' (float): step of increment in degrees
    
    Outputs:
        * 'val' (list, float): values of the grib message at proximity of the point
    """
    val = grb.data(lat1 = lat - dlat, lat2 = lat + dlat, lon1 = lon - dlon, lon2 = lon+dlon)[0]
    # [1] and [2] are lat and lon, respectively
    #val = test.data(grb,lat1=lat-dl,lat2=lat+dl,lon1=lon-dl,lon2=lon+dl)[0]
    #The function looks for value in a square centered on (lon, lat). If there is no values in the square,
    #its size is increased (by dl in degrees) until at least one value is found inside    
    while (len(val) == 0):
        dlat += 0.05
        dlon += 0.05
        val = grb.data(lat1 = lat - dlat, lat2 = lat + dlat, lon1 = lon - dlon, lon2 = lon+dlon)[0]
        #val, lats, lons = test.data(grb,lat1=lat-dl,lat2=lat+dl,lon1=lon-dl,lon2=lon+dl)
    return val

def download_and_filter_data(depTime,forecast='000',filetype='rap',verbose=True):
    """
    Downloads the grib file from the NCDC server on the local disk
    
    Args:
        * 'DT' (string): date and time of the last weather bulletin available
        * 'forecast' (string): time horizon of the weather forecast requested
        * 'filetype' (string): type of source files for weather: either namanl or rap
        * 'verbose' (boolean): if True, print some directory paths
    """
    if not filetype in ['rap','nam','namanl']:
        raise NameError("type should be either 'rap' or 'nam'")
        return
    
    date = depTime[0:8]
    time = depTime[8:12]    
    
    ufile = DATADIR+'/wdata_'+date+'_'+time+'_u.grb'+('2' if filetype=='rap' else '')
    vfile = DATADIR+'/wdata_'+date+'_'+time+'_v.grb'+('2' if filetype=='rap' else '')
    rfile = DATADIR+'/rdata_'+date+'_'+time+'_tisrgrd.grb'+('2' if filetype=='rap' else '')

    #if not (os.path.exists(ufile) and os.path.exists(vfile) and os.path.exists(rfile)):
    if filetype in ['rap', 'nam']:
        fname = get_fname(filetype,depTime,forecast)[4:]
    else: 
        fname = get_fname(filetype,depTime,forecast)[7:]

    if not os.path.exists(FILTEREDWEATHERDATADIR+'/'+filetype+'/'+fname):
        if not os.path.exists(WEATHERDATADIR+'/'+filetype+'/'+fname):
            print('Downloading file %s ...'%fname)
            if filetype=='rap':
                mycmd = ('wget -P '+RAPDATADIR+' ftp://nomads.ncdc.noaa.gov/RUC/13km/%s/%s/'+fname)%(date[:-2],date)
            elif filetype=='namanl':
                mycmd = ('wget -P '+NAMANLDATADIR+' ftp://nomads.ncdc.noaa.gov/NAM/analysis_only/%s/%s/'+fname)%(date[:-2],date)
            elif filetype=='nam':
                mycmd = ('wget -P '+NAMDATADIR+' http://nomads.ncdc.noaa.gov/data/meso-eta-hi/%s/%s/'+fname)%(date[:-2],date)
            # only for last few days
            elif filetype=='gfs': 
                mycmd = ('wget -P '+GFSDATADIR+' ftp://nomads.ncep.noaa.gov:9090/dods/gfs/gfs%s/gfs%s_%sz')%(mdate,mdate,mforecast[1:])
            commands.getstatusoutput(mycmd)
        else:
#            print('I have the file %s'%fname)
            pass
            
        mycmd = 'grib_filter '+os.path.dirname(__file__)+'/filter_for_wind_and_radar_grouped '+WEATHERDATADIR+'/'+filetype+'/'+fname+';'
        mycmd = mycmd+'mv '+FILTEREDWEATHERDATADIR+'/temp.grb'+('2' if filetype=='rap' else '')+' '+FILTEREDWEATHERDATADIR+'/'+filetype+'/'+fname
        commands.getstatusoutput(mycmd)
    else:
        # print('I already have the filtered file %s'%fname)
        pass

    if verbose:
        print('-'*80)
        lookup_file(ufile)
        print('-'*80)
        lookup_file(vfile)
        print('-'*80)

    return os.path.join(os.getcwd(), ufile), os.path.join(os.getcwd(), vfile), os.path.join(os.getcwd(), rfile)


def grib_is_valid(gid,keys):
    for k in keys.keys():
        try:
            g = grib_get(gid,k)
        except err:
            g = None
        if g is not None:
            g_str = g if g is str else str(g)
            if g not in keys[k]:
                # print '/'.join(keys[k])+', '+g_str
                # print 'NO'
                return False
            # print str(k)+': '+g_str+'    '+'/'.join(keys[k])+' ---> OK'
    # print gid
    return True

def release_all_gids(gid_list):
    for gid in gid_list:
        try:
            grib_release(gid)
        except GribInternalError:
            print(sys.stderr,err.msg)
    return

def get_keys_from_gid(gid,verbose=False):
    iterid = grib_keys_iterator_new(gid,'ls')
    #
    res = []
    while grib_keys_iterator_next(iterid):
        keyname = grib_keys_iterator_get_name(iterid)
        # keyval = grib_get_string(iterid,keyname)
        res.append(keyname)
        if verbose:
            print(keyname,keyval)
    #
    grib_keys_iterator_delete(iterid)
    return res

# ['edition', 'centre', 'date', 'dataType', 'gridType', 'stepRange', 'typeOfLevel', 'level', 'shortName', 'packingType']
def get_keys_name(gid_list,keys=['paramId','name','nameECMF','level','typeOfLevel']):
    for gid in gid_list:
        t = ", ".join([str(grib_get(gid,k)) for k in keys])
        print(gid, t)

def get_gids_from_file(fname,header=False):
    f = open(fname) 
    mcount = grib_count_in_file(f)
    gid_list = [grib_new_from_file(f,header) for i in range(mcount)]
    f.close()
    return gid_list

def lookup_file(fname):
    gid_list = get_gids_from_file(fname,True)
    get_keys_name(gid_list)
    release_all_gids(gid_list)

def get_message_from_gids_with_keys(gid_list,keys,releaseOthers=False):
    res = []
    for gid in gid_list:
        if grib_is_valid(gid,keys):
            if VERBOSE==1:
                print('!! %d is a valid message !!'%gid)
            res.append(gid)
        else:
            if VERBOSE==1:
                print('   %d is not a valid message'%gid)
            # gid_list.remove(gid)
            if releaseOthers:
                grib_release(gid)
    return res

def get_message_from_file_with_keys(fname,keys):
    gid_list = get_gids_from_file(fname)
    return get_message_from_gids_with_keys(gid_list,keys)


def get_data(dtype='u',date='20150429',time='0000',keys={'shortName':['u','v','tisrgrd'],'typeOfLevel':['isobaricInhPa'],'level':[125]},dsrc='nam',verbose=False):
    if verbose:
        print(dtype,date,time,keys['level'],dsrc)

    if dtype in ['u','v','tisrgrd']:
        download_and_filter_data(date,time,dsrc,False)
    else:
        # TODO
        # download_nonwind_data(date,time,dsrc,verbose)
        raise NameError("data filter only implemented for wind data")
        return

    # fname = 'rap_130_' if type=='rap' else 'namanl_218_'
    # fname = fname + date +'_'+time+'_000'
    # fname = fname + ('.grb2' if type=='rap' else '.grb')

    # all_gids = get_gids_from_file(dsrc+'/'+fname)
    wfname = 'data/'+('r' if dtype=='tisrgrd' else 'w')+'data_'+date+'_'+time+'_'+dtype+'.grb'+('1' if dsrc=='nam' else '2')  

    gid_list = get_gids_from_file(wfname)
    if verbose:
        print(gid_list)

    # wid = get_message_from_gids_with_keys(gid_list,keys)
    # if verbose:
    #     print wid
    # wid = wid[0]
    wid = []
    for gid in gid_list:
        if grib_is_valid(gid,keys):
            wid.append(gid)
            break
    if len(wid)>0:
        wid = wid[0]
        if verbose:
            print(wid)
    else:
        raise NameError("No message corresponding to the desired keys")

    wdata = grib_get_values(wid)
    if verbose:
        print(wdata)

    Nx = grib_get_elements(wid,'Nx',0)
    Ny = grib_get_elements(wid,'Ny',0)
    wdata.shape = (Ny,Nx)

    # TODO
    # find real lat_0, lon_0
    lat_0 = grib_get_elements(wid,'latitudeOfSouthernPoleInDegrees',0)[0]
    lon_0 = grib_get_elements(wid,'longitudeOfSouthernPoleInDegrees',0)[0]
    # print lat_0, lon_0
    lat_0,lon_0 = (38.5339273887, 261.81191642)
    # print lat_0, lon_0

    lat_1 = grib_get_elements(wid,'latitudeOfFirstGridPointInDegrees',0)[0]
    lon_1 = grib_get_elements(wid,'longitudeOfFirstGridPointInDegrees',0)[0]

    title = ''
    if dtype=='u':
        title = 'U component of wind'
    elif dtype == 'v':
        title = 'V component of wind'
    else:
        title = str(grib_get_elements(wid,'name',0)[0])

    title = title+' for period ending '+ str(int(grib_get_elements(wid,'dataDate',0)[0]))\
        +' '+str(int(grib_get_elements(wid,'dataTime',0)[0]))+' at level 125 hPa'

    release_all_gids(gid_list)
    return wdata,Nx,Ny,lat_0,lon_0,lat_1,lon_1,title

