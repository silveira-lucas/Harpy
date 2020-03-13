import numpy as np
from scipy import linalg
from numba import jit

#%%
   
def euler_int(ode, t, y0, dy0): # return y, dy, ddy
    y = np.zeros( (len(t),len(y0)) )
    dy = np.zeros( (len(t),len(y0)) )
    ddy = np.zeros( (len(t),len(y0)) )
    
    y[0, :] = y0
    dy[0, :] = dy0
    ddy[0, :] = ode(0.0, 0, y[0, :], dy[0, :])
    
    h = np.diff(t)
    for i_t in range(len(t)-1):
        
        print('time = %0.2f'%t[i_t])
        
        ddy[i_t+1, :] = ode(t[i_t], i_t, y[i_t, :], dy[i_t, :])
        dy[i_t+1, :] = dy[i_t, :] + ddy[i_t+1, :] * h[i_t]
        y[i_t+1, :] = y[i_t, :] + dy[i_t+1, :] * h[i_t]
    
    return y, dy, ddy

def rgkn(ode, t, y0, dy0): # return y, dy, ddy
    '''
    Runge Runge–Kutta-Nyström method. Integrates numerically a second order
    system of differential equations of the form:
        
        y'' = ode(t, y, y')
    
    The algorithm was modified to allow passing the time-step counter i_t to
    the differential equation, ode.
    
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
    
    y = np.zeros( (len(t), len(y0)) )
    dy = np.zeros( (len(t), len(y0)) )
    ddy = np.zeros( (len(t), len(y0)) )
    
    y[0, :] = y0
    dy[0, :] = dy0
    ddy[0, :] = ode(0.0, 0, y[0, :], y[0, :])
    
    h = np.diff(t)
    for i_t in range(len(t)-1):
        
        A = (h[i_t]/2.) * ddy[i_t, :]
        b = (h[i_t]/2.) * (dy[i_t, :]+(1./2.)*A)
        B = (h[i_t]/2.) * ode(t[i_t]+(h[i_t]/2.), i_t, y[i_t, :]+b, dy[i_t, :]+A)
        C = (h[i_t]/2.) * ode(t[i_t]+(h[i_t]/2.), i_t, y[i_t, :]+b, dy[i_t, :]+B)
        d = h[i_t] * (dy[i_t, :] + C)
        D = (h[i_t]/2.) * ode(t[i_t+1], i_t+1, y[i_t, :]+d, dy[i_t, :]+(2*C))
        
        y[i_t+1, :] = y[i_t, :] + h[i_t] * (dy[i_t, :]+ (1./3.)*(A+B+C))
        dy[i_t+1, :] = dy[i_t, :] + (1./3.)*(A+2*B+2*C+D)
        ddy[i_t+1, :] = ode(t[i_t+1], i_t+1, y[i_t+1, :], dy[i_t+1, :])

    return y, dy, ddy

def rotate_tensor(angle, axis): # return A_01
    '''
    Generates the transformation tensor A_01 from two frames of reference,
    where the reference frame 1 is obtained rotating the given angle
    around the given axis of the reference frame 0.
    
    Parameters
    ----------
    angle : float
            [rad] angle between the two reference frames along the given axis
    axis : str
           axis of rotation: 'x', 'y' or 'z'
    
    Returns
    -------
    A_01 : numpy.ndarray[:, :], dtype='float'
           transformation tensor from the reference frame 0 to the reference
           frame 1
    '''
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    if (axis=='x'):
        A_01 = np.array([[1., 0., 0.],
                         [0., cos_angle, sin_angle],
                         [0., -sin_angle, cos_angle]], dtype=float)
    elif (axis=='y'):
        A_01 = np.array([[cos_angle, 0., -sin_angle],
                         [0., 1., 0.],
                         [sin_angle, 0., cos_angle]], dtype=float)
    elif (axis=='z'):
        A_01 = np.array([[cos_angle, sin_angle, 0.],
                         [-sin_angle, cos_angle, 0.],
                         [0., 0., 1.]], dtype=float)
    else:
        print("I am sorry Dave, I am afraind I cannot do that. The axis of rotation must be either 'x', 'y' or 'z'.")
    
    return A_01

def omega_tensor(omega): # return Omega
    # omega is a 3d vector
    Omega = np.array([[0., -omega[2], omega[1]],
                      [omega[2], 0., -omega[0]],
                      [-omega[1], omega[0], 0.]])
    return Omega
    
def print_comment(string, ident, max_columns=79):
    '''
    The function manipulates the string so it can be printed considering 
    the maximum of carachters per line defined taking into account the 
    identation of the funcitons.
    
    Parameters
    ----------
    string : string
             python string to be formated
    ident : string
            python string containing the identation inside the function
    max_columns : integer, optional
                  maximum number of character per line. The default number 
                  is 79, as recommended by the PEP8 stadard.
                  
    Returns
    -------
    comment : string
              string formated to the maximum of line length desired
    '''
    words = string.split()
    
    comment = ''
    line_length = int(0)
    for word in words:    
        if (len(comment)==0):
            comment += ident + word
            line_length = len(ident) + len(word)
        elif (line_length + len(' ') + len(word) < max_columns):
            comment += ' ' + word
            line_length += len(' ') + len(word)            
        else:
            comment += ' ' + '\n' + ident + word
            line_length = len(ident) + len(word)
    return comment

@jit
def interp_3d(x, y, z, x_vec, y_vec, z_vec, f):
    '''
    This method calculates the trilinear interpolation on a 3D regular and
    equaly spaced grid. The code uses the algorith bellow:
    
    f(x, y, z) = a0 + a1*x + a2*y + a3*z + a4*x*y + a5*x*z + a6*y*z + 
               + a7*x*y*z
    
    where a_j is given by:
    
    |1 x0 y0 z0 x0*y0 x0*z0 y0*z0 x0*y0*z0|   |a0|   |f(x0, y0, z0)|
    |1 x1 y0 z0 x1*y0 x1*z0 y0*z0 x1*y0*z0|   |a1|   |f(x1, y0, z0)|
    |1 x0 y1 z0 x0*y1 x0*z0 y1*z0 x0*y1*z0|   |a2|   |f(x0, y1, z0)|
    |1 x1 y1 z0 x1*y1 x1*z0 y1*z0 x1*y1*z0| . |a3| = |f(x1, y1, z0)|
    |1 x0 y0 z1 x0*y0 x0*z1 y0*z1 x0*y0*z1|   |a4|   |f(x0, y0, z1)|
    |1 x1 y0 z1 x1*y0 x1*z1 y0*z1 x1*y0*z1|   |a5|   |f(x1, y0, z1)|
    |1 x0 y1 z1 x0*y1 x0*z1 y1*z1 x0*y1*z1|   |a6|   |f(x0, y1, z1)|
    |1 x1 y1 z1 x1*y1 x1*z1 y1*z1 x1*y1*z1|   |a7|   |f(x1, y1, z1)|
    
    Parameters
    ----------
    f : numpy.ndarray [:, :, :], dtype=float
        Values of the function of f(x, y, z) on the grid points
    x_vec : numpy.ndarray [:], dtype=float
            x values of grid
    y_vec : numpy.ndarray [:], dtype=float
            y values of grid
    z_vec : numpy.ndarray [:], dtype=float
            z values of grid
    x : float
        x point where the function is to be interpolated
    y : float
        y point where the function is to be interpolated
    z : float
        z point where the function is to be interpolated
    
    Returns
    -------
    f = float
        value of f(x, y, z) calculated using the trilinear interpolation
        method.
    '''
    
    # Fiding the indices of the grid points around the given point (x, y, z)
    # (The grid must be equaly spaced for this algorithm to work)
    ix_0 = int((len(x_vec)-1)/(x_vec[-1]-x_vec[0])*(x-x_vec[0]))
    ix_1 = int((len(x_vec)-1)/(x_vec[-1]-x_vec[0])*(x-x_vec[0])) + 1
    
    iy_0 = int((len(y_vec)-1)/(y_vec[-1]-y_vec[0])*(y-y_vec[0]))
    iy_1 = int((len(y_vec)-1)/(y_vec[-1]-y_vec[0])*(y-y_vec[0])) + 1
    
    iz_0 = int((len(z_vec)-1)/(z_vec[-1]-z_vec[0])*(z-z_vec[0]))
    iz_1 = int((len(z_vec)-1)/(z_vec[-1]-z_vec[0])*(z-z_vec[0])) + 1
    
    x0 = x_vec[ix_0]
    x1 = x_vec[ix_1]
    
    y0 = y_vec[iy_0]
    y1 = y_vec[iy_1]
    
    z0 = z_vec[iz_0]
    z1 = z_vec[iz_1]
    
    # A = np.array([[1., x0, y0, z0, x0*y0, x0*z0, y0*z0, x0*y0*z0],
    #               [1., x1, y0, z0, x1*y0, x1*z0, y0*z0, x1*y0*z0],
    #               [1., x0, y1, z0, x0*y1, x0*z0, y1*z0, x0*y1*z0],
    #               [1., x1, y1, z0, x1*y1, x1*z0, y1*z0, x1*y1*z0],
    #               [1., x0, y0, z1, x0*y0, x0*z1, y0*z1, x0*y0*z1],
    #               [1., x1, y0, z1, x1*y0, x1*z1, y0*z1, x1*y0*z1],
    #               [1., x0, y1, z1, x0*y1, x0*z1, y1*z1, x0*y1*z1],
    #               [1., x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1]])

    # c = np.array([f[ix_0, iy_0, iz_0],
    #               f[ix_1, iy_0, iz_0],
    #               f[ix_0, iy_1, iz_0],
    #               f[ix_1, iy_1, iz_0],
    #               f[ix_0, iy_0, iz_1],
    #               f[ix_1, iy_0, iz_1],
    #               f[ix_0, iy_1, iz_1],
    #               f[ix_1, iy_1, iz_1]])
    
    # a = linalg.solve(A, c)
    
    c = np.zeros((2, 2, 2))
    for i, i_x in enumerate((ix_0, ix_1)):
        for j, i_y in enumerate((iy_0, iy_1)):
            for k, i_z in enumerate((iz_0, iz_1)):
                c[i, j, k] = f[i_x, i_y, i_z]

    den = (x0-x1)*(y0-y1)*(z0-z1)
    a = np.zeros((8,))
    a[0] = 1./den*(- c[0,0,0]*x1*y1*z1 + c[0,0,1]*x1*y1*z0
                    + c[0,1,0]*x1*y0*z1 - c[0,1,1]*x1*y0*z0
                    + c[1,0,0]*x0*y1*z1 - c[1,0,1]*x0*y1*z0
                    - c[1,1,0]*x0*y0*z1 + c[1,1,1]*x0*y0*z0)
    
    a[1] = 1./den*(+ c[0,0,0]*y1*z1 - c[0,0,1]*y1*z0 
                    - c[0,1,0]*y0*z1 + c[0,1,1]*y0*z0
                    - c[1,0,0]*y1*z1 + c[1,0,1]*y1*z0
                    + c[1,1,0]*y0*z1 - c[1,1,1]*y0*z0)

    a[2] = 1./den*(+ c[0,0,0]*x1*z1 - c[0,0,1]*x1*z0
                    - c[0,1,0]*x1*z1 + c[0,1,1]*x1*z0
                    - c[1,0,0]*x0*z1 + c[1,0,1]*x0*z0
                    + c[1,1,0]*x0*z1 - c[1,1,1]*x0*z0)
    
    a[3] = 1./den*(+ c[0,0,0]*x1*y1 - c[0,0,1]*x1*y1
                    - c[0,1,0]*x1*y0 + c[0,1,1]*x1*y0
                    - c[1,0,0]*x0*y1 + c[1,0,1]*x0*y1
                    + c[1,1,0]*x0*y0 - c[1,1,1]*x0*y0)
    
    a[4] = 1./den*(- c[0,0,0]*z1 + c[0,0,1]*z0
                    + c[0,1,0]*z1 - c[0,1,1]*z0
                    + c[1,0,0]*z1 - c[1,0,1]*z0
                    - c[1,1,0]*z1 + c[1,1,1]*z0)
    
    a[5] = 1./den*(- c[0,0,0]*y1 + c[0,0,1]*y1
                    + c[0,1,0]*y0 - c[0,1,1]*y0
                    + c[1,0,0]*y1 - c[1,0,1]*y1
                    - c[1,1,0]*y0 + c[1,1,1]*y0)
    
    a[6] = 1./den*(- c[0,0,0]*x1 + c[0,0,1]*x1
                    + c[0,1,0]*x1 - c[0,1,1]*x1
                    + c[1,0,0]*x0 - c[1,0,1]*x0
                    - c[1,1,0]*x0 + c[1,1,1]*x0)
    
    a[7] = 1./den*(+ c[0,0,0] - c[0,0,1]
                    - c[0,1,0] + c[0,1,1]
                    - c[1,0,0] + c[1,0,1]
                    + c[1,1,0] - c[1,1,1])

    y = a[0] + a[1]*x + a[2]*y + a[3]*z + a[4]*x*y + a[5]*x*z + a[6]*y*z + a[7]*x*y*z
    return y

def curve_length(r):
    '''
    Calculate the length of a 2D or 3D discrete curve.
    
    L = sum(np.sqrt(dx**2+ dy**2+ dz**2))

    Parameters
    ----------
    r : numpy.ndarray[:, :], dtype='float'
        [m] Discrete curve
            axis 0: point index
            axis 1: x, y and z coordinates

    Returns
    -------
    l : float
        [m] curve length
    '''
        
    dl = np.linalg.norm(np.diff(r, axis=0), axis=1)
    L = np.sum(dl)
    return L

@jit
def fftx(x): # return y   # recursive function
    '''
    The function calculates the discrete Fourier transformation of x using the
    simplest recursive of the Cooley-Tukey radix 2 FFT algorithm. The algorithm
    used can be found at the link bellow:
    https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
    
    The @jit decorator is used to compile the function code Just In Time for
    its use with a better performance. However the code performance is
    significantly slower than the FFT algorithm found in numpy.fft.fft, which
    uses the LAPACK routine, when len(x) is not a power of 2.
    
    Parameters
    ----------
    x : numpy.ndarray[:], dtype = float or complex
        Signal to be processed
    
    Returns
    -------
    y : numpy.ndarray[:], dtype = complex
        DFT of x
    '''
    
    n = len(x)
    omega = np.exp(-2.*np.pi*1j/n)
    if (np.mod(n,2)==0):
        k2 = np.arange(0,n/2)
        ye = fftx(x[0:n-1:2])
        yo = omega**k2*fftx(x[1::2])
        y = np.concatenate([ye+yo,ye-yo])
    else:
        j = np.arange(0,n,1)
        j1 = j.reshape(len(j),1)
        j2 = j.reshape(1,len(j))
        F = omega**(j1@j2)
        y = F@x    
    return y

def arg_index(x, val):
    
    diff = x - val
    i = np.argmin(np.abs(diff))
      
    return i
