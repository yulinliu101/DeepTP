'''
Module's author : Jarry Gabriel
Date : June, July 2016
Some Algorithms was made by : Malivai Luce, Helene Piquet

Module Modified by: Yulin Liu
Date: 10/25/2016

This module handle weather methods
''' 

import timeMethods
from weather.gributils import *
import tools
import math
import numpy as np
import zipfile

# path1 = '/media/liuyulin101/YulinLiu/WindData'
path1 = '/home/liuyulin/Desktop/DATA/filtered_weather_data/namanl/'
# path2 = '/home/liuyulin/Desktop/WindFeature/'


class GetWindSpeed:
    def __init__(self, departureTime, arrivalTime):
        self.departureTime = departureTime
        self.arrivalTime = arrivalTime
        self.meteoTime, self.winds, self.lvls, self.final_timeTags, self.final_timeTags_array = self.getWeather()
        
    def getWeather(self):
        """
        Initial: ['000','001','002','003','006']
        Valid: ['000','001','006']
        Final: ['001','006']
        """
        datasrc = path1
        AllFileName = os.listdir(datasrc)
        datatype = 'namanl'
        meteoTime = timeMethods.getLastWeatherDate(self.departureTime, 'namanl')
        timeTags = timeMethods.getTimeTags(datatype, self.departureTime, self.arrivalTime, meteoTime)
        fnames = []
        for mforecast in timeTags:
            fnames.append(get_fname(datasrc, timeMethods.dateFromDT(meteoTime), mforecast))
        
        # Check if file exists
        valid_timeTags = []
        valid_fnames = []
        for fidx in range(len(fnames)):
            if fnames[fidx][-32:] in AllFileName:
                valid_timeTags.append(timeTags[fidx])
                valid_fnames.append(fnames[fidx])
            else:
                pass
        valid_timeTags_array = np.array(list(map(int,valid_timeTags)))
        
        # Only Load files from nearest departure time forecast
        DepTimeTag = timeMethods.getTimeTag((self.departureTime - meteoTime).seconds, valid_timeTags_array)        
        Final_timeTags = valid_timeTags[DepTimeTag:]
        
        lvls = {}
        grbs = pygrib.open(valid_fnames[DepTimeTag])
        uin = grbs.select(shortName='u', typeOfLevel='isobaricInhPa', level = lambda l: l >= 150 and l <= 1000)
        vin = grbs.select(shortName='v', typeOfLevel='isobaricInhPa', level = lambda l: l >= 150 and l <= 1000)
        grbs.close()
        winds = [(uin, vin)]
#        for vfidx in range(DepTimeTag, len(valid_timeTags)):
#            grbs = pygrib.open(valid_fnames[vfidx])
#            uin = grbs.select(shortName='u', typeOfLevel='isobaricInhPa')
#            vin = grbs.select(shortName='v', typeOfLevel='isobaricInhPa')
#            grbs.close()
#            winds.append((uin, vin))

        try:
            for k in range(len(uin)):
                lvls[uin[k].level] = k
        except UnboundLocalError:
            pass
        
        Final_timeTags_array = np.array(list(map(int,Final_timeTags)))
        return meteoTime, winds, lvls, Final_timeTags, Final_timeTags_array
        
    def getWind(self, lon ,lat, alt, cur_time, lonl, latl):
        azimuth = tools.g.inv(lonl, latl, lon, lat)[0]
        # return the forward [0] and back [1] azimuths, and distance between [3] two points
#        diffTime = (cur_time - self.meteoTime).seconds
#        tag = timeMethods.getTimeTag(diffTime, self.final_timeTags_array)
        try:
            lvl= tools.proxilvl(alt, self.lvls)
            i_lvl = int(self.lvls[lvl])
            u = get_data_at_lonlat(self.winds[0][0][i_lvl], lon, lat)
            v = get_data_at_lonlat(self.winds[0][1][i_lvl], lon, lat)
            
            u = np.mean(u)
            v = np.mean(v)
            windSpeed = u * math.sin(azimuth * math.pi / 180) + v * math.cos(azimuth * math.pi / 180)
        except:
            windSpeed = 999999
        # m/s    
        return windSpeed

    
# Original Codes
# Package Everything into a Class    
    
#def getWeather(departureTime, arrivalTime):
#    datasrc = path + '/DATA/filtered_weather_data/namanl'
#    datatype = 'namanl'
#
#    ahead_time = datetime.timedelta(hours=1)
#    meteoTime = timeMethods.getLastWeatherDate(departureTime - ahead_time, 'namanl')
#    
##    if not os.path.exists(datasrc):
##        os.mkdir(datasrc)
#
#    timeTags = timeMethods.getTimeTags(datatype, departureTime, arrivalTime, meteoTime)
#    fnames = []
#    for mforecast in timeTags:
#        # Since all files are correctly downloaded, this line is commented out
##        download_and_filter_data(timeMethods.dateFromDT(meteoTime), mforecast, datatype, False)
#        fnames.append(get_fname(datasrc, timeMethods.dateFromDT(meteoTime), mforecast))
#        
#    winds, lvls, valid_timeTags = getWeatherData(fnames, timeTags)
#    
#    return meteoTime, winds, lvls, valid_timeTags
#
## Sets the weather data to the citypair flight
#def getWeatherData(fnames, timeTags):
#    winds = []
#    valid_tags = []
#    lvls = {}
#    for fidx in range(len(fnames)):
#        try:
#            grbs = pygrib.open(os.path.join(os.getcwd(), fnames[fidx]))
#            uin = grbs.select(shortName='u', typeOfLevel='isobaricInhPa')
#            vin = grbs.select(shortName='v', typeOfLevel='isobaricInhPa')
#            grbs.close()
#            winds.append((uin, vin))
#            
#            valid_tags.append(timeTags[fidx])
#            # those files are in the directory of ../filtered_weather_data/namanl
#        except:
#            pass
##            try:
##                grbs = pygrib.open(os.path.join(os.getcwd(), fnames[min(fidx + 1, len(fnames) - 1)]))
##            except IOError:
##                print('No weather Datas')
##                break
##        uin = grbs.select(shortName='u', typeOfLevel='isobaricInhPa')
##        vin = grbs.select(shortName='v', typeOfLevel='isobaricInhPa')
##        grbs.close()
##        winds.append((uin, vin))
#    try:
#        for k in range(len(uin)):
#            lvls[uin[k].level] = k
#    except UnboundLocalError:
#        pass
#    return winds, lvls, valid_tags
#
## Returns wind speed in km/h at (lon, lat) and level lvl with and position time t, meteoDT, last last position (lonl, latl), winds,
#def getWind(lon ,lat, alt, t , meteoDT, lonl, latl, timeTags, winds, lvls):
#    azi = tools.g.inv(lonl, latl, lon, lat)[0]
#    # return the forward [0] and back [1] azimuths, and distance between [3] two points
#    tag = timeMethods.getTimeTag(timeMethods.deltaTime(t, meteoDT), timeTags)
#    
#    lvl= tools.proxilvl(alt, lvls)
#    windSpeed = getWindSpeed(lon ,lat , azi, tag, int(lvls[lvl]), winds, meteoDT)
#    return windSpeed
#
# # Defines methods to get windspeed km/hr
#def getWindSpeed(lon, lat, azimuth, tag, i_lvl, winds, meteoDT):
#    t = meteoDT
#    u = get_data_at_lonlat(winds[tag][0][i_lvl], lon, lat)
#    v = get_data_at_lonlat(winds[tag][1][i_lvl], lon, lat)
#    u = sum(u) / len(u)
#    v = sum(v) / len(v)
#    delta = u * math.sin(azimuth * math.pi / 180) * 3.6 + v * math.cos(azimuth * math.pi / 180) * 3.6
#    return delta
#
#
