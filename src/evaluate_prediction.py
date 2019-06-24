# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-10-22 14:31:13
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2019-06-23 20:44:21

import numpy as np
import pandas as pd
from visualize_samples import plot_fp_act
import pickle
from scipy.interpolate import interp1d
from utils import g
import matplotlib.pyplot as plt

pred, predicted_tracks_cov, buffer_total_logprob, buffer_pi_prob, predicted_matched_info = pickle.load(open('sample_results/all_lite_samp_mu_cov_test_s2_w80_batch0.pkl', 'rb'))

class evaluate_prediction:
    def __init__(self, 
                 pred_results_datapath_list,
                 actual_track_datapath = '../../DATA/DeepTP/processed_flight_tracks.csv',
                 flight_plan_datapath = '../../DATA/DeepTP/processed_flight_plans.csv',
                 flight_plan_utilize_datapath = '../../DATA/DeepTP/IAH_BOS_Act_Flt_Trk_20130101_1231.CSV',
                 feed_track_datapath = '../../DATA/DeepTP/test_flight_tracks_all.csv',
                 feed_fp_datapath = '../../DATA/DeepTP/test_flight_plans_all.csv',
                 n_mix = 3,
                 search_pwr = 2,
                 pred_dt = 120.
                 ):
        self.pred_results_datapath_list = pred_results_datapath_list
        self.actual_track_datapath = actual_track_datapath
        self.flight_plan_datapath = flight_plan_datapath
        self.flight_plan_utilize_datapath = flight_plan_utilize_datapath
        self.feed_track_datapath = feed_track_datapath
        self.feed_fp_datapath = feed_fp_datapath

        self.n_mix = n_mix
        self.search_pwr = search_pwr
        self.pred_dt = pred_dt

        self.preds, \
            self.pred_covs, \
                self.pred_logprobs, \
                    self.act_track_data, \
                        self.FP_track, \
                            self.FP_utlize, \
                                self.feed_data, \
                                    self.feed_fp = self._load_tracks()

    def _load_tracks(self):
        act_track_data = pd.read_csv(self.actual_track_datapath, header = 0)
        FP_track = pd.read_csv(self.flight_plan_datapath)
        FP_utlize = pd.read_csv(self.flight_plan_utilize_datapath, header = 0, usecols = [19,1])
        feed_data = pd.read_csv(self.feed_track_datapath, header = 0)
        feed_fp = pd.read_csv(self.feed_fp_datapath, header = 0)
        self.n_feed = feed_data.groupby('FID').FID.count().values[0] - 1

        act_track_data['cumDT'] = act_track_data.groupby('FID').DT.transform(pd.Series.cumsum)
        feed_data['cumDT'] = feed_data.groupby('FID').DT.transform(pd.Series.cumsum)

        preds = []
        pred_covs = []
        pred_logprobs = []
        for pfile in self.pred_results_datapath_list:
            with open(pfile, 'rb') as pfilein:
                pred, predicted_tracks_cov, buffer_total_logprob, _, _ = pickle.load(pfilein)
            preds.append(pred)
            pred_covs.append(predicted_tracks_cov)
            pred_logprobs.append(buffer_total_logprob)

        preds = np.concatenate(preds, axis = 0)
        pred_covs = np.concatenate(pred_covs, axis = 0)
        pred_logprobs = np.concatenate(pred_logprobs, axis = 0)

        return preds, pred_covs, pred_logprobs, act_track_data, FP_track, FP_utlize, feed_data, feed_fp


    def _best_sequence_idx(self,
                           buffer_total_logprob,
                           ):
        idx = self.n_mix**(self.search_pwr)
        n_predictions = buffer_total_logprob.shape[0]//idx
        best_seq_idx = []
        for i in range(n_predictions):
            best_seq_idx.append(np.argmax(buffer_total_logprob[i*idx:(i+1)*idx]) + i*idx)
        return best_seq_idx

    def _resample_interpolate_ground_truth(self):
        # resample ground truth to make it equal time interval as the predictions
        ground_truth = self.act_track_data.loc[self.act_track_data.FID.isin(self.feed_fp.FLT_PLAN_ID.unique())].reset_index(drop = True)
        ground_truth = ground_truth.drop(index = ground_truth.groupby('FID').head(self.n_feed).index)

        int_ground_truth_arr = self._interpolation(ground_truth)
        return int_ground_truth_arr

    def _interpolation(self,
                       track_dataframe):
        new_series = []
        i = 0

        for idx, gp in track_dataframe.groupby('FID'):
            i += 1
            # Interpolated in terms of time
            # dold = gp.CumDist.values
            told = gp.cumDT.values
            xold = gp.Lon.values
            yold = gp.Lat.values
            zold = gp.Alt.values

            f1 = interp1d(told, xold, kind = 'linear')
            f2 = interp1d(told, yold, kind = 'linear')
            f3 = interp1d(told, zold, kind = 'linear')

            tnew = np.arange(told[0],told[-1], step = self.pred_dt)
            xnew = f1(tnew)
            ynew = f2(tnew)
            znew = f3(tnew)
            new_series.append(np.stack([ynew, xnew, znew], axis = 1))
        # new_series = np.array(new_series)

        return new_series

    def prediction_error(self, 
                         predictions,
                         ground_truth = None,
                         beam_search = True,
                         resample_and_interpolation = True):
        if beam_search:
            best_seq_idx = self._best_sequence_idx(self.pred_logprobs)
            predictions = predictions[best_seq_idx, ] # shape of [n_seq, n_time, 6|--> lat lon alt cumT latspd lonspd]
        if ground_truth is not None:
            self.ground_truth = ground_truth.copy()
        else:
            if resample_and_interpolation:
                self.ground_truth = self._resample_interpolate_ground_truth() # list of arrays with shape of [n_time, 3]
            else:
                raise ValueError("No ground truth!")
        
        avg_horizontal_err = []
        avg_vertical_err = []
        all_horizontal_err = []
        all_vertical_err = []
        for i in range(len(self.ground_truth)):
            n_pnt = min(self.ground_truth[i].shape[0], predictions[i].shape[0] - self.n_feed - 1)
            # print(n_pnt)
            _, _, dist = g.inv(self.ground_truth[i][:n_pnt, 1], 
                               self.ground_truth[i][:n_pnt, 0], 
                               predictions[i][self.n_feed:self.n_feed+n_pnt, 1], 
                               predictions[i][self.n_feed:self.n_feed+n_pnt, 0])

            alt_dist = 100*(self.ground_truth[i][:n_pnt, 2] - predictions[i][self.n_feed:self.n_feed+n_pnt, 2]) # ft.
            
            all_horizontal_err += list(dist/1852)
            all_vertical_err += list(alt_dist)

            avg_horizontal_err.append(np.mean(np.abs((dist/1852)))) # in nmi
            avg_vertical_err.append(np.mean(np.abs(alt_dist)))
            # avg_horizontal_err.append(np.sqrt(np.mean((dist/1852)**2))) # in nmi
            # avg_vertical_err.append(np.sqrt(np.mean(alt_dist**2)))
        
        return np.array(avg_horizontal_err), np.array(avg_vertical_err), np.array(all_horizontal_err), np.array(all_vertical_err)

    def prediction_coverage(self, 
                            n_std,
                            predictions,
                            prediction_cov,
                            ground_truth = None,
                            beam_search = True,
                            resample_and_interpolation = True):
        if beam_search:
            best_seq_idx = self._best_sequence_idx(self.pred_logprobs)
            predictions = predictions[best_seq_idx, ] # shape of [n_seq, n_time, 6|--> lat lon alt cumT latspd lonspd]
            predictions_cov = np.sqrt(prediction_cov[best_seq_idx, ]) # shape of [n_seq, n_time - n_feed-1, 5,5|--> lat lon alt latspd lonspd]
        if ground_truth is not None:
            self.ground_truth = ground_truth.copy()
        else:
            if resample_and_interpolation:
                self.ground_truth = self._resample_interpolate_ground_truth() # list of arrays with shape of [n_time, 3]
            else:
                raise ValueError("No ground truth!")
        
        n_horizotal_cover = []
        n_vertical_cover = []
        n_full_cover = []

        percentage_horizotal_cover = []
        percentage_vertical_cover = []
        percentage_full_cover = []

        total_pts = 0
        for i in range(len(self.ground_truth)):
            n_pnt = min(self.ground_truth[i].shape[0], predictions[i].shape[0] - self.n_feed - 1)

            _cond_lat_rhs = (self.ground_truth[i][:n_pnt, 0] <= (predictions[i][self.n_feed:self.n_feed+n_pnt, 0] + predictions_cov[i][:n_pnt, 0, 0] * n_std)) # lat
            _cond_lat_lhs = (self.ground_truth[i][:n_pnt, 0] >= (predictions[i][self.n_feed:self.n_feed+n_pnt, 0] - predictions_cov[i][:n_pnt, 0, 0] * n_std)) # lat
            _cond_lon_rhs = (self.ground_truth[i][:n_pnt, 1] <= (predictions[i][self.n_feed:self.n_feed+n_pnt, 1] + predictions_cov[i][:n_pnt, 1, 1] * n_std)) # lon
            _cond_lon_lhs = (self.ground_truth[i][:n_pnt, 1] >= (predictions[i][self.n_feed:self.n_feed+n_pnt, 1] - predictions_cov[i][:n_pnt, 1, 1] * n_std)) # lon
            _cond_alt_rhs = (self.ground_truth[i][:n_pnt, 2] <= (predictions[i][self.n_feed:self.n_feed+n_pnt, 2] + predictions_cov[i][:n_pnt, 2, 2] * n_std)) # alt
            _cond_alt_lhs = (self.ground_truth[i][:n_pnt, 2] >= (predictions[i][self.n_feed:self.n_feed+n_pnt, 2] - predictions_cov[i][:n_pnt, 2, 2] * n_std)) # alt

            _horizontal_cond = (_cond_lat_lhs & _cond_lat_rhs & _cond_lon_lhs & _cond_lon_rhs)
            _vertical_cond = (_cond_alt_rhs & _cond_alt_lhs)
            _full_cond = (_horizontal_cond & _vertical_cond)
            
            n_horizotal_cover.append(_horizontal_cond.sum())
            percentage_horizotal_cover.append(_horizontal_cond.sum()/n_pnt)

            n_vertical_cover.append(_vertical_cond.sum())
            percentage_vertical_cover.append(_vertical_cond.sum()/n_pnt)

            n_full_cover.append(_full_cond.sum())
            percentage_full_cover.append(_full_cond.sum()/n_pnt)

            total_pts += n_pnt
        
        return (np.array(percentage_horizotal_cover), 
                np.array(percentage_vertical_cover), 
                np.array(percentage_full_cover), 
                sum(n_horizotal_cover)/total_pts, 
                sum(n_vertical_cover)/total_pts,
                sum(n_full_cover)/total_pts)

    def plot_hist(self, 
                  all_hor_err,
                  avg_horizontal_err,
                  all_alt_err,
                  avg_vertical_err):
        fig, axs = plt.subplots(2, 2, figsize=(10,6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(wspace = 0.2, hspace = 0.35)
        axs = axs.ravel()
        _ = axs[0].hist(all_hor_err, 50, range = (0, 200), density = True)
        _ = axs[0].set_title('Horizontal Error (All)')
        _ = axs[0].set_xlabel('Distance/ nmi')
        _ = axs[1].hist(avg_horizontal_err, 50, range = (0, 200), density = True)
        _ = axs[1].set_title('Horizontal Error (Flight)')
        _ = axs[1].set_xlabel('Distance/ nmi')
        _ = axs[2].hist(all_alt_err, 25, range = (-150, 150), density = True)
        _ = axs[2].set_title('Vertical Error (All)')
        _ = axs[2].set_xlabel('Distance/ FL')
        _ = axs[3].hist(avg_vertical_err, 25, range = (0, 150), density = True)
        _ = axs[3].set_title('Vertical Error (Flight)')
        _ = axs[3].set_xlabel('Distance/ FL')
        return