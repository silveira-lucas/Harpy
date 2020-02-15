import numpy as np

#%%

def sig2ext(sig):
    '''
    Extract the local extrema in the time history signal. When equal values are
    encountered, only the latest one is preserved.
    
    Parameters
    ----------
    sig : numpy.ndarray[:], dtype='float'
          Time history signal
    
    Returns
    -------
    ext : numpy.ndarray[:], dtype='float'
          Local extrema in the history signal.
    tp : numpy.ndarray[:], dtype='bool'
         Boolean array in which a True indicates a local extremum and False
         otherwise
    '''

    # Boolean array of length len(sig)
    tp = np.array([True for i in range(len(sig))])
    
    # Check for local extremum
    for i in range(1,len(sig)-1):
        if (sig[i]>=sig[i-1] and sig[i]>sig[i+1]):
            tp[i] = True
        elif (sig[i]<=sig[i-1] and sig[i]<sig[i+1]):
            tp[i] = True
        else:
            tp[i] = False
    
    # Geting extrema values of signal, i.e. where tp[:] == True
    ext = sig[tp]
    
    return (ext, tp)