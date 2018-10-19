# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-10-16 13:52:03
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-10-18 21:05:45

import numpy as np

def calculate_next_pnt_kf(current_state, 
                          current_cov,
                          measure_state,
                          measure_cov,
                          delta_t,
                          Q_err = 1e-6,
                          Q_scale_factor = 10,
                          validation_gate = 0.8,
                          maneuver_thres = 0.3,
                          Kalman = True):
    # current_state has shape: [n_seq, n_state_var]
    # current_cov has shape: [n_seq, n_state_var, n_state_var]
    # measure_state has shape: [n_seq, n_state_var]
    # measure_cov has shape: [n_seq, n_state_var, n_state_var]

    if not Kalman:
        return measure_state, measure_cov, np.zeros((measure_state.shape[0], 1))

    n_seq, n_state_var = current_state.shape
    I = np.zeros(shape = (n_seq, n_state_var, n_state_var))
    I[:, ] = np.eye(n_state_var, n_state_var)

    Q = I.copy()
            
    Q[:, :2, :2] = Q[:, :2, :2] * Q_err * 1000
    Q[:, 2, 2] = 1.
    Q[:, 3:, 3:] = Q[:, 3:, 3:] * Q_err

    current_state = current_state.reshape(n_seq, n_state_var, 1)
    measure_state = measure_state.reshape(n_seq, n_state_var, 1)

    next_state_pred, next_cov_pred, A = _process_model(current_state,
                                                 current_cov,
                                                 delta_t,
                                                 Q)

    next_state, next_cov, residual, S_inv, kf_gain, mahal_error_sq = _update(next_state_pred,
                                                                              next_cov_pred,
                                                                              measure_state,
                                                                              measure_cov)
    logprob_measure = np.zeros((n_seq, 1))
    gate_idx = np.where(mahal_error_sq >= validation_gate)
    maneuver_idx = np.where((mahal_error_sq < validation_gate) & 
                            (mahal_error_sq >= maneuver_thres))
    # print(gate_idx)
    # print(maneuver_idx)
    # initialize output_state and cov
    output_state = next_state.reshape(n_seq, n_state_var).copy()
    output_cov = next_cov.copy()
    Q_time_holder = np.zeros((n_seq, n_state_var, n_state_var))

    if gate_idx[0].size == 0:
        pass
    else:
        # return predicted points for validation gates
        output_state[gate_idx[0], :] = next_state_pred[gate_idx[0], :, 0]
        output_cov[gate_idx[0], :, :] = next_cov_pred[gate_idx[0], :, :]
        logprob_measure[gate_idx[0]] = -9.
    
    if maneuver_idx[0].size == 0:
        pass
    else:
        # rescale Q matrix for maneuver points
        Q_scale = Q[maneuver_idx[0], :, :] * Q_scale_factor
        next_man_state_pred, next_man_cov_pred, A = _process_model(current_state[maneuver_idx[0]],
                                                                     current_cov[maneuver_idx[0]],
                                                                     delta_t,
                                                                     Q_scale)
        next_man_state, next_man_cov, _, _, _, _ = _update(next_man_state_pred,
                                                                next_man_cov_pred,
                                                                measure_state[maneuver_idx[0]],
                                                                measure_cov[maneuver_idx[0]])
        output_state[maneuver_idx[0], :] = next_man_state.reshape(-1, n_state_var)
        output_cov[maneuver_idx[0], :, :] = next_man_cov


    
    return output_state, output_cov, logprob_measure
    

def _process_model(current_state,
                   current_cov,
                   delta_t,
                   Q):
    A = np.array([[1., 0., 0., delta_t, 0], 
                 [0., 1., 0., 0., delta_t],
                 [0., 0., 1., 0., 0.],
                 [0., 0., 0., 1., 0.],
                 [0., 0., 0., 0., 1.]])
    # B = np.array([[0., 0.5 * delta_t**2, 0.], 
    #               [0., 0., 0.5 * delta_t**2], 
    #               [delta_t, 0, 0], 
    #               [0, delta_t, 0], 
    #               [0, 0, delta_t]])
    # u = np.zeros((n_seq, 3, 1))
    # u[:, :, 0] = [alt_spd, lat_accr, lon_accr]

    next_state_pred = A @ current_state
    next_cov_pred = A @ current_cov @ (A.T) # shape of [n_seq, n_state_var, n_state_var]
    next_cov_pred += Q
    next_cov_pred[:, :3, 3:] = 0.
    next_cov_pred[:, 3:, :3] = 0.
    return next_state_pred, next_cov_pred, A

def _update(next_state_pred,
            next_cov_pred,
            measure_state,
            measure_cov,
            ):
    
    H = np.zeros((next_cov_pred.shape[0], next_cov_pred.shape[-1], next_cov_pred.shape[-1]))
    H[:, ] = np.eye(5)

    S = H @ next_cov_pred @ np.transpose(H, axes = [0, 2, 1]) + measure_cov

    I = np.zeros(shape = S.shape)
    I[:, ] = np.eye(S.shape[-1])
    # print(np.linalg.cond(S))

    # S_inv = np.linalg.solve(S, I)

    # use diagonal approximation to avoicd numerical instability
    S_diag_inv = 1/np.diagonal(S, axis1=1, axis2=2)
    S_inv = np.zeros(S.shape)
    S_inv[:, range(S.shape[-1]), range(S.shape[-1])] = S_diag_inv

    residual = H @ (measure_state - next_state_pred) # shape of [n_seq, n_state_var, 1]

    kf_gain = (next_cov_pred @ np.transpose(H, [0,2,1])) @ S_inv
    next_state = next_state_pred  + kf_gain @ (residual)

    next_cov = next_cov_pred - kf_gain @ H @ next_cov_pred
    next_cov = (next_cov + np.transpose(next_cov, [0, 2, 1]))/2
    next_cov[:, :3, 3:] = 0.
    next_cov[:, 3:, :3] = 0.
    # next_cov = (I - kf_gain) @ next_cov_pred @ np.transpose(I - kf_gain, axes = [0, 2, 1]) + kf_gain @ measure_cov @ np.transpose(kf_gain, [0, 2, 1])
    # mahal_error_sq = np.transpose(residual[:, :2, ], axes = [0, 2, 1]) @ np.linalg.solve(next_cov_pred[:, :2, :2], I[:, :2, :2]) @ residual[:, :2, ]
    # mahal_error_sq = np.abs(residual[:, 0, 0])/np.sqrt(next_cov_pred[:, 0, 0])+np.abs(residual[:, 1, 0])/np.sqrt(next_cov_pred[:, 1, 1])
    mahal_error_sq = np.abs(residual[:, 0, 0])+np.abs(residual[:, 1, 0])
    return next_state, next_cov, residual, S_inv, kf_gain, mahal_error_sq.flatten()

def RTS_smoother(batch_kf_state,
                 batch_kf_cov,
                 batch_Q,
                 A):
    # batch_kf_state has shape of [n_seq, n_time, n_state_var]
    # batch_kf_cov has shape of [n_seq, n_time, n_state_var, n_state_var]
    # batch_Q has shape of [n_seq, n_time, n_state_var, n_state_var]
    # A is the state transition matrix

    n_seq, n_time, n_state_var = batch_kf_state.shape
    rts_states = [batch_kf_state[:, -1, :]]
    rts_covs = [batch_kf_cov[:, -1, :, :]]
    for t in range(n_time - 2, -1, -1):
        current_cov = batch_kf_cov[:, t, :, :] # shape of [n_seq, n_state_var, n_state_var]
        current_state = batch_kf_state[:, t, :, None] # shape of [n_seq, n_state_var, 1]
        I = np.zeros(current_cov.shape)
        I[:, ] = np.eye(n_state_var)
        # predict
        next_cov_pred = A @ current_cov @ A.T + batch_Q[:, t, :, :]
        next_state_pred = A @ current_state
        # update
        current_C = current_cov @ A.T @ np.linalg.solve(next_cov_pred, I)
        state_rts = batch_kf_state[:, t, :, None] + current_C @ (batch_kf_state[:, t+1, :, None] - next_state_pred) # shape of [n_seq, n_state_var, 1]
        cov_rts = batch_kf_cov[:, t, :, :] + current_C @ (batch_kf_cov[:, t+1, :, :] - next_cov_pred) @ np.transpose(current_C, axes = [0, 2, 1])

        rts_states.append(state_rts.reshape(n_seq, n_state_var))
        rts_covs.append(cov_rts)
    rts_states.reverse()
    rts_covs.reverse()
    rts_states = np.transpose(np.array(rts_states), axes = [1,0,2])
    rts_covs = np.transpose(np.array(rts_covs), axes = [1,0,2,3])
    return rts_states, rts_covs