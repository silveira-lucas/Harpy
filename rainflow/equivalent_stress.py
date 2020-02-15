import numpy as np

#%%

def eq_stress(N, sigma_ar, b, N_B):
    '''
    Calculates the equivalent constant amplitude stress level that causes the
    same life as the variable history if applied for the same number of cycles.
    
    The equation follows eq. 9.37 on Mechanical Behavior of Materials:
    Engineering Methods for Deformation, Fracture, and Fatigue, fourth edition,
    2007, Dowling, N.E. ISBN 9780131863125
    
    Parameters
    ----------
    N : np.ndarray[:], dtype='float'
        Number of cycles at each level. Half cycles are possible on rainflow
        counting.
    sigma_ar : np.ndarray[:], dtype='float'
               Equivalent completely reversed stress amplitude computed from
               the stress amplitude and mean.
    b : float
        Stress life curve exponent
        eq: sigma_ar = sigma_f_prime * (2*N_f)**b
    N_B : float
          number of rainflow cycles. Half cycles are possible on rainflow
          counting.
    
    Returns
    -------
    sigma_aq : float
               Equivalent constant amplitude stress level that causes the
               same life as the variable history if applied for the same number
               of cycles
    '''
    
    sigma_aq = (np.sum(N*sigma_ar**(-1./b))/N_B)**(-b)
    
    return sigma_aq