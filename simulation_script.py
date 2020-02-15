from os import system as sys
import platform
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import pickle

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
t_max = 1000.
delta_t = 0.02

t_vec = np.arange(t_min, t_max+delta_t, delta_t)

#%% Wind turbine

# Wind turbine characteristics file
wt_file = './turbine_data/WT_general_properties.xlsx'

# Wind turbine object constructor
wt = WindTurbine.construct(wt_file=wt_file)

# Modify some of the turbine characteristics
wt.Omega_min = wt.lmbda * wt.u_cut_in / wt.R # no minimum limit for Omega
wt.Omega = 0.673 # [rad/s]
wt.is_stiff = False

wt.downwind_azimuth()
wt.initilialise()
M = wt.mass_matrix() 
C = wt.gyro_matrix()
K = wt.stiffness_matrix()
M1 = np.linalg.inv(M)

#%% Wind

wind = WindBox.construct(wt, wt_file, t_max, delta_t*10, 8.0, turbulence_generator='windsimu_x32', shear_format='power_law', alpha=0.2, t_ramp=0., seed=seed)

#%%

BEM = UnsteadyBem(wt, t_vec)

def ode(t, i_t, q, q_dot):
    
    q_ddot = BEM.unsteady_bem(wt, wind, t, i_t, q, q_dot, dynamic_inflow_on, dynamic_inflow_on)

    return q_ddot

#%%

y0 = np.zeros((len(wt.q),))
dy0 = np.zeros((len(wt.q),))

y, dy, ddy = cf.rgkn(ode, t_vec, y0, dy0)

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

#%% End

t_end = time.time()
print('The simulation took %0.4f [s], Dave' %(t_end-t_start))
