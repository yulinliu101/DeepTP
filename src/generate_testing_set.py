# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-10-08 15:33:11
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-10-08 15:37:06

import numpy as np
import pandas as pd

def generate_testing_set(actual_track_datapath = '../../DATA/DeepTP/processed_flight_tracks.csv',
                         flight_plan_datapath = '../../DATA/DeepTP/processed_flight_plans.csv',
                         flight_plan_utilize_datapath = '../../DATA/DeepTP/IAH_BOS_Act_Flt_Trk_20130101_1231.CSV',
                         testing_fid = [20130118900394, 20130426357386, 20130713836889, 20130810273857, 20131109716864],
                         num_feed_pnt = 20,
                         testing_track_dir = '../../DATA/DeepTP/test_flight_tracks.csv',
                         testing_fp_dir = '../../DATA/DeepTP/test_flight_plans.csv',
                         ):

    act_track_data = pd.read_csv(actual_track_datapath, header = 0)
    FP_track = pd.read_csv(flight_plan_datapath)
    FP_utlize = pd.read_csv(flight_plan_utilize_datapath, header = 0, usecols = [19,1])

    testing_track = act_track_data.loc[act_track_data.FID.isin(testing_fid), 
                                       ['FID', 
                                        'Elap_Time', 
                                        'Lat', 
                                        'Lon', 
                                        'Alt', 
                                        'Speed', 
                                        'DT', 
                                        'Dist', 
                                        'CumDist']]
    testing_track_head20 = testing_track.groupby('FID').head(num_feed_pnt).reset_index(drop = True)
    testing_fp = np.empty((0, 4))
    end_idx = 0
    for tmpfid in testing_fid:
        flp_id = FP_utlize.loc[FP_utlize.FID == tmpfid, 'FLT_PLAN_ID'].values[0]
        fp_arr = FP_track.loc[FP_track.FLT_PLAN_ID == flp_id, ['LONGITUDE', 'LATITUDE']].values
        id_arr = np.array([[tmpfid, flp_id]] * fp_arr.shape[0], dtype = object)
        fp_arr = np.concatenate((fp_arr, id_arr), axis = 1)
        
        testing_fp = np.concatenate((testing_fp, fp_arr), axis = 0)
    testing_fp = pd.DataFrame(testing_fp, columns=['LONGITUDE', 'LATITUDE', 'FLT_PLAN_ID', 'FLT_PLAN_ID_REAL'])

    if testing_fp_dir is not None:
        testing_fp.to_csv(testing_fp_dir, index = False)
    if testing_track_dir is not None:
        testing_track_head20.to_csv(testing_track_dir, index = False)
    
    return testing_track_head20, testing_fp
