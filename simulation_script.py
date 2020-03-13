from os import system as sys
import platform
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import pickle
from scipy.interpolate import UnivariateSpline

import custom_functions as cf
from wind_box import WindBox
from wind_turbine import WindTurbine
from wind_turbine_aerodynamics import UnsteadyBem

t_start = time.time()

#%%

case = 'case_111'
seed = -1
save_data_on = True

dynamic_stall_on = True
dynamic_inflow_on = True

#%%

t_min = 0.
t_max = 1000.0
delta_t = 0.02

t_vec = np.arange(t_min, t_max+delta_t, delta_t)

#%% Wind turbine

# Wind turbine characteristics file
wt_file = './turbine_data/WT_general_properties.xlsx'

# Wind turbine object constructor
wt = WindTurbine.construct(wt_file=wt_file, n_modes=4)

# Modify some of the turbine characteristics
wt.Omega_min = wt.lmbda * wt.u_cut_in / wt.R # no minimum limit for Omega
wt.Omega = 0.673 # [rad/s]
wt.yaw = 0.
# wt.tilt = 0.
# wt.cone = 0.

wt.is_stiff = False

wt.downwind_azimuth()
wt.initilialise()
M = wt.mass_matrix() 
C = wt.gyro_matrix()
K = wt.stiffness_matrix()
M1 = np.linalg.inv(M)


#%%

# phi_x_dot = np.zeros(wt.phi_x.shape)
# phi_y_dot = np.zeros(wt.phi_y.shape)
# for i_m in range(wt.n_m):
#     phi_x_spl = UnivariateSpline(wt.z, wt.phi_x[i_m, :])
#     phi_y_spl = UnivariateSpline(wt.z, wt.phi_x[i_m, :])
#     pass

for i_m in range(wt.n_m):
    # phi_x_spl = UnivariateSpline(wt.z, wt.phi_x[i_m, :])
    # phi_y_spl = UnivariateSpline(wt.z, wt.phi_y[i_m, :])
    
    fig = plt.figure()
    plt.plot(wt.z, wt.phi_x[i_m, :], label=r'$\phi_x$')
    plt.plot(wt.z, wt.phi_y[i_m, :], label=r'$\phi_y$')
    # plt.plot(wt.z, phi_x_spl.derivative(n=1)(wt.z), label=r'$\phi_x$ (spline)')
    # plt.plot(wt.z, phi_y_spl.derivative(n=1)(wt.z), label=r'$\phi_y$ (spline)')
    plt.xlabel(r'z [m]')
    plt.ylabel(r'$\phi$ [-]')
    plt.title(r'mode %i'%(i_m+1))
    plt.grid('on')
    plt.legend()
    plt.tight_layout()

#%% Wind

wind = WindBox.construct(wt, wt_file, t_max, delta_t*10, 8.0, turbulence_generator='windsimu_x32', shear_format='power_law', alpha=0.2, t_ramp=0., seed=seed)

#%%

BEM = UnsteadyBem(wt, t_vec)

def ode(t, i_t, q, q_dot):
    
    # Aerodynaic simulation
    BEM.unsteady_bem(wt, wind, t, i_t, q, q_dot, dynamic_inflow_on, dynamic_inflow_on)
    
    # Deformation
    if (wt.is_stiff):
        wt.q_ddot = np.zeros(q.shape)
    else:
        wt.reinitilialise()
        M = wt.mass_matrix()
        C = wt.gyro_matrix()
        K = wt.stiffness_matrix()
        F = wt.force_vector()            
        wt.q_ddot = np.linalg.solve(M, (F - C@q_dot - K@q))
    
    # Shaft azimuth angle
    if (t!=BEM.t_vec[-1] and t==BEM.t_vec[i_t]):
        delta_t = BEM.t_vec[i_t + 1] - t
        BEM.theta[i_t+1] = BEM.theta[i_t] + wt.Omega * delta_t
        
        # no variation in Omega yet
        BEM.Omega[i_t+1] = BEM.Omega[i_t]
    
    return wt.q_ddot

#%%

y0 = np.zeros((len(wt.q),))
dy0 = np.zeros((len(wt.q),))

BEM.q, BEM.q_dot, BEM.q_ddot = cf.rgkn(ode, t_vec, y0, dy0)

t_end = time.time()
print('The simulation took %0.3f [s]'%(t_end-t_start))
 
#%% Saving the data

folder = './' + case + '/'

if (save_data_on):
    sys('mkdir '+folder)
    
    with open(folder+case+'.pkl', 'wb') as f:
        pickle.dump(['wt', 'BEM'], f)
        pickle.dump(wt, f)
        pickle.dump(BEM, f)

#with open(folder+case+'.pkl', 'rb') as f:
#    _ = pickle.load(f)
#    wt = pickle.load(f)
#    BEM = pickle.load(f)

