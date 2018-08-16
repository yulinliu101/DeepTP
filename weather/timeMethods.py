'''
Module's author : Jarry Gabriel
Date : June, July 2016
Some Algorithms was made by : Malivai Luce, Helene Piquet
This module handle time methods
'''

import datetime
import numpy as np
## Returns algebrics delta in seconds between date a and date b
# Deprecated
#def deltaTime(a,b):   
#    d = a - b
#    ds = d.seconds
#    dd = d.days
#    return int(ds+(3600*24)*dd)

# Returns a datetime.datetime object from a date string in the format YYYYMMDDHHMM
def dateFromString(date):

    return datetime.datetime.strptime(date, '%Y%m%d%H%M%S')

# Returns a date string in the format YYYYMMDDHHMM rrom a datetime.datetime object
def dateFromDT(date):

    return datetime.datetime.strftime(date, '%Y%m%d%H%M%S')

# Returns the time of the last weather bulletin available at the time of the flight planning,
# using the departure date and the type of source file
def getLastWeatherDate(dt, dsrc):

    date = dt.date()
    time = dt.time()
    if dsrc=='namanl':
        newHour = time.hour - time.hour%6
        time = time.replace(hour=newHour, minute=0)
        return datetime.datetime.combine(date,time)
    elif dsrc=='rap':
        time = time.replace(minute=0)
        return datetime.datetime.combine(date,time)

# Returns the time horizons for weather forecast with source type, distance of the longest route,
# departure date and time of the last weather bulletin available
def getTimeTags(dsrc, depTime, arrTime, meteoTime):
    # depTime: actual departure time
    # meteoTime = timeMethods.getLastWeatherDate(dt - ahead_time, 'namanl'))
    # instead of using avspeed to calculate duration, use the actual arrival time
#    avspeed = 630
#    flightime = datetime.timedelta(minutes=int(60*distmax/avspeed))
    timeHorizon = arrTime - meteoTime
#    timeHorizon2 = depTime - meteoTime
    delta = int(timeHorizon.seconds/3600)
#    delta2 = int(timeHorizon2.seconds/3600)
    
    if(dsrc == 'namanl'):
        if delta >= 6:
            return ['000', '001', '002', '003', '006']
        elif delta >= 3:
            return ['000', '001', '002', '003']
        else:
            return ['00' + str(k) for k in range(delta + 1)]
      
    if(dsrc == 'rap'):
        tags = []
        for k in range(min(25,delta + 1)):
            if(k < 10):
                tags.append('00' + str(k))
            else:
                tags.append('0' + str(k))
        return tags


# Returns the time tag which correspond to a deltatime
def getTimeTag(deltatime, timeTags_array):        
    # deltatime is the time difference (sec) between meteological cycle time and current postion time
    # timeTags_array is sth like [0,1,2,3,6]
    return np.abs(timeTags_array - deltatime/3600).argmin()
    
#    def proxy(deltatime, t1, t2):
#        if (abs(3600*int(t1) - deltatime) < abs(3600*int(t2) - deltatime)):
#            return t1
#        else:
#            return t2
#    last = '000'
#    for tag in timeTags:
#        if (int(deltatime) == 3600*int(tag)):
#            return timeTags.index(tag)
#        elif (int(deltatime) < 3600*int(tag)):
#            return timeTags.index(proxy(deltatime, tag, last))
#        last = tag
#    return len(timeTags) - 1
#    
#
#