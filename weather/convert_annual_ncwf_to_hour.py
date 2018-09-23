# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-09-18 15:46:32
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-09-18 16:11:48

import numpy as np


ncwf_arr_dir = '../../DATA/NCWF/gridded_storm.npz'
wx_pointer = np.load(ncwf_arr_dir)

ncwf_wx_arr = wx_pointer['ncwf_arr']
start_time = wx_pointer['start_time']
wx_unique_alt = wx_pointer['unique_alt']
smallgrid = wx_pointer['smallgrid']


np.savez_compressed('../../DATA/NCWF/gridded_storm_hourly/ncwf_info_file.npz', 
                    start_time = start_time,
                    unique_alt = wx_unique_alt,
                    smallgrid = smallgrid)
for idx in range(start_time.shape[0]):
    if idx % 500 == 0:
        print(idx)
    fname = '%d_%s_%s_%s00Z.npz'%(start_time[idx, 0], 
                               str(start_time[idx, 1]).zfill(2), 
                               str(start_time[idx, 2]).zfill(2), 
                               str(start_time[idx, 3]).zfill(2))
    np.savez_compressed('../../DATA/NCWF/gridded_storm_hourly/%s'%fname, 
                        ncwf_arr = ncwf_wx_arr[idx])
print("Done!")