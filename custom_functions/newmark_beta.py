import numpy as np
from scipy.linalg import solve

#%%

def newmark_beta(M, C, K, t, y0, dy0):
    '''
    Parameters
    ----------
    ode : python function
    t : numpy.ndarray [:], dtype='float'
        [s] time simulation array
    y0 : numpy.ndarray [:], dtype='float'
         [-] ode variable y initial conditions
    dy0 : numpy.ndarray [:], dtype='float'
          [:] ode variable first derivative inital conditions
    
    Returns
    -------
    y : numpy.ndarray [:, :], dtype='flaot'
        ode variable as a function of [time, y_dimension]
    dy : numpy.ndarray [:, :], dtype='flaot'
        ode variable first derivative as a function of [time, y_dimension]
    ddy : numpy.ndarray [:, :], dtype='flaot'
        ode variable first derivative as a function of [time, y_dimension]
    '''
    
    gamma = 0.5
    beta = 0.25
    
    y = np.zeros( (len(t), len(y0)) )
    dy = np.zeros( (len(t), len(y0)) )
    ddy = np.zeros( (len(t), len(y0)) )
    
    y[0, :] = y0
    dy[0, :] = dy0
    # ddy[0, :] = solve(M, F-)
    
    
    
    pass