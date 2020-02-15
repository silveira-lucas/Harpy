"""
-------------------------------------------------------------------------------
Rainflow counting function
Reproduced from Jennifer Rinker - 2015, with minor modifications on the output
format.
Original at:
https://gist.github.com/jennirinker/688a917ccb7a9c14e78f 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------------------
"""

from numpy import fabs as fabs
import numpy as np

#%%

def rainflow(array_ext,
             flm=0, l_ult=1e16, uc_mult=0.5):
    '''
    Rainflow counting of a signal's turning points with Goodman correction.
    Reproduced from Jennifer Rinker, 2015, with minor modifications on the
    return format. Original at:
    https://gist.github.com/jennirinker/688a917ccb7a9c14e78f
    
    Parameters
    ----------
    flm : numpy.ndarray[:], dtype='float'
          Array of turning points
    flm : float. Optional, default=0
          Fixed-load mean.
    l_ult : float. Optional, default=1e16
            Ultimate load
    uc_mult : float. Optional, default=0.5
              partial-load scaling
    
    Returns
    -------
    load amplitude : numpy.ndarray[:], dtype='float'
    range mean : numpy.ndarray[:], dtype='float'
    cycle count : numpy.ndarray[:], dtype='float'
    Goodman-adjusted amplitude : numpy.ndarray[:], dtype='float'
    Goodman-adjusted range with flm = 0 : numpy.ndarray[:], dtype='float'
    '''

    flmargin = l_ult - fabs(flm)            # fixed load margin
    tot_num = array_ext.size                # total size of input array
    array_out = np.zeros((5, tot_num-1))    # initialize output array
    
    pr = 0                                  # index of input array
    po = 0                                  # index of output array
    j = -1                                  # index of temporary array "a"
    a  = np.empty(array_ext.shape)          # temporary array for algorithm
    
    # loop through each turning point stored in input array
    for i in range(tot_num):
        
        j += 1                  # increment "a" counter
        a[j] = array_ext[pr]    # put turning point into temporary array
        pr += 1                 # increment input array pointer
        
        while ((j >= 2) & (fabs( a[j-1] - a[j-2] ) <= \
                fabs( a[j] - a[j-1]) ) ):
            lrange = fabs( a[j-1] - a[j-2] )
              
            # partial range
            if j == 2:
                mean      = ( a[0] + a[1] ) / 2.
                adj_range = lrange * flmargin / ( l_ult - fabs(mean) )
                adj_zero_mean_range = lrange * l_ult / ( l_ult - fabs(mean) )
                a[0]=a[1]
                a[1]=a[2]
                j=1
                if (lrange > 0):
                    array_out[0,po] = lrange
                    array_out[1,po] = mean
                    array_out[2,po] = adj_range
                    array_out[3,po] = uc_mult
                    array_out[4,po] = adj_zero_mean_range
                    po += 1
                
            # full range
            else:
                mean      = ( a[j-1] + a[j-2] ) / 2.
                adj_range = lrange * flmargin / ( l_ult - fabs(mean) )
                adj_zero_mean_range = lrange * l_ult / ( l_ult - fabs(mean) )
                a[j-2]=a[j]
                j=j-2
                if (lrange > 0):
                    array_out[0,po] = lrange
                    array_out[1,po] = mean
                    array_out[2,po] = adj_range
                    array_out[3,po] = 1.00
                    array_out[4,po] = adj_zero_mean_range
                    po += 1
                    
    # partial range
    for i in range(j):
        lrange    = fabs( a[i] - a[i+1] );
        mean      = ( a[i] + a[i+1] ) / 2.
        adj_range = lrange * flmargin / ( l_ult - fabs(mean) )
        adj_zero_mean_range = lrange * l_ult / ( l_ult - fabs(mean) )
        if (lrange > 0):
            array_out[0,po] = lrange
            array_out[1,po] = mean
            array_out[2,po] = adj_range
            array_out[3,po] = uc_mult
            array_out[4,po] = adj_zero_mean_range
            po += 1  
            
    # get rid of unused entries
    array_out = array_out[:,:po]
    
    return (array_out[0, :]/2., array_out[1, :], array_out[3, :], array_out[2, :]/2., array_out[4,:])