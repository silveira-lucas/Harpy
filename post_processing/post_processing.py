import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import h5py
# import time
from numba import jit

#%%


def rotor_moment(wt, BEM):
    
    f_gravity = np.zeros( (len(BEM.t_vec), wt.NB, len(wt.z), 3) )
    f_inertia = np.zeros( (len(BEM.t_vec), wt.NB, len(wt.z), 3) )
    f_blade = np.zeros( (len(BEM.t_vec), wt.NB, len(wt.z), 3) )
    M_blade = np.zeros( (len(BEM.t_vec), wt.NB, 3))
    M_rot = np.zeros( (len(BEM.t_vec), 3))
    M_root = np.zeros((len(BEM.t_vec), wt.NB, 3))
    
    for i_t in range(len(BEM.t_vec)):
        
        print(BEM.t_vec[i_t])
        
        wt.q = BEM.q[i_t, :]
        wt.q_dot = BEM.q_dot[i_t, :]
        if (BEM.t_vec[i_t]!=BEM.t_vec[-1]):
            wt.q_ddot = BEM.q_ddot[i_t+1, :]
        else:
            wt.q_ddot = BEM.q_ddot[i_t, :]
        
        wt.theta = BEM.theta[i_t]
        wt.Omega = BEM.Omega[i_t]
        
        for i_b in range(wt.NB):
            # Initial azimuth angle
            wt.eta = np.pi + (2.*i_b*np.pi)/wt.NB
            
            wt.reference_matrices()
            wt.A_02 = (wt.A_01.T @ wt.A_12.T).T
            wt.A_03 = (wt.A_01.T @ wt.A_12.T @ wt.A_23.T ).T
            wt.A_04 = (wt.A_01.T @ wt.A_12.T @ wt.A_23.T @ wt.A_34.T ).T
            wt.A_24 = (wt.A_23.T @ wt.A_34.T).T
            
            for i_z in range(len(wt.z)):
                fg_0 = np.array([0, 0, wt.m[i_z]*wt.g])
                f_gravity[i_t, i_b, i_z, :] = wt.A_04 @ fg_0
    
            # blade deflection acceleration (its reference of frame)
            ur_ddot = np.zeros( (len(wt.z), 3) )
            ur_ddot[:, 0] = wt.q_ddot[(wt.n_m*i_b)+0:(wt.n_m*i_b)+wt.n_m] @ wt.phi_x[:, :]
            ur_ddot[:, 1] = wt.q_ddot[(wt.n_m*i_b)+0:(wt.n_m*i_b)+wt.n_m] @ wt.phi_y[:, :]
            ur_ddot[:, 2] = 0.
            
            for i_d in range(3):
                f_inertia[i_t, i_b, :, i_d] = wt.m*ur_ddot[:, i_d]
            
            f_blade[i_t, i_b, :, :] = BEM.f_aero[i_t, i_b, :, :] + f_gravity[i_t, i_b, :, :] - f_inertia[i_t, i_b, :, :]
            
            M_blade[i_t, i_b, :] = wt.A_34.T @ np.trapz( np.cross(BEM.r_b4[i_t, i_b, :, :], f_blade[i_t, i_b, :, :]), BEM.r_b4[i_t, i_b, :, 2], axis=0)
    
    M_rot = np.sum(M_blade, axis=1)
    P_harpy = M_rot[:, 1] * BEM.Omega
    
    return f_blade, f_gravity, f_inertia, M_rot, P_harpy

def lift_drad(wt, BEM):
    l = np.zeros((len(BEM.t_vec), wt.NB, len(wt.z)))
    d = np.zeros((len(BEM.t_vec), wt.NB, len(wt.z)))
    for i_t in range(len(BEM.t_vec)):
        print(BEM.t_vec[i_t])
        for i_b in range(wt.NB):
            for i_z in range(len(wt.z)):
                l[i_t, i_b, i_z] =   np.sin(BEM.phi[i_t, i_b, i_z]) * BEM.f_aero[i_t, i_b, i_z, 0] + np.cos(BEM.phi[i_t, i_b, i_z]) * BEM.f_aero[i_t, i_b, i_z, 1]
                d[i_t, i_b, i_z] = - np.cos(BEM.phi[i_t, i_b, i_z]) * BEM.f_aero[i_t, i_b, i_z, 0] + np.sin(BEM.phi[i_t, i_b, i_z]) * BEM.f_aero[i_t, i_b, i_z, 1]
    return l, d

def M_root(f, r_b4, t_vec, NB):
    r = np.zeros(r_b4.shape)
    r[:, :, :, 2] = r_b4[:, :, :, 2] - 2.8
    
    M_root = np.zeros((len(t_vec), NB, 3))
    for i_t in range(len(t_vec)):
        print(t_vec[i_t])
        for i_b in range(NB):
            M_root[i_t, i_b, :] = np.trapz( np.cross(r[i_t, i_b, :, :], f[i_t, i_b, :, :]), r[i_t, i_b, :, 2], axis=0)

    return M_root