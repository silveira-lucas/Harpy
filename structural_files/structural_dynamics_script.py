from IPython.display import display
from os import system as syst
import sys
import sympy as sym
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
import time
import psutil 

from my_printer import MyPrinter
from joblib import Parallel, delayed
import multiprocessing

sym.init_printing()

t_start = time.time()

#%%

# Number of mode shapes per blade
n_m = 4

# Create or erase log file used by the parallelized functins
file_log = open('file_log.txt', 'wt')
file_log.close()

# Number of physical cores available for parallelisation
n_cores = psutil.cpu_count(logical = False)
# n_cores = multiprocessing.cpu_count()

# Parallel function verbose option
par_verbose = 51

#%% Personal functions

def rotate_matrix(angle,axis): # return A
    '''
    Generates a transformation matrix.
    
    Parameters
    ----------
    angle : sympy.Symbol or sympy.Function
            [rad] angle between the two reference frames along the 'axis'
    axis : str
           axis of roation. 'x', 'y' and 'z'.
    
    Returns
    -------
    A : sympy.Matrix [:, :]
        transformation matrix from reference frame 1 to reference frame 2.
    '''
    if (axis=='x'):
        A = sym.Matrix([[1, 0, 0],
                    [0, sym.cos(angle), sym.sin(angle)],
                    [0, -sym.sin(angle), sym.cos(angle)]])
    elif (axis=='y'):
        A = sym.Matrix([[sym.cos(angle), 0, -sym.sin(angle)],
                    [0, 1, 0],
                    [sym.sin(angle), 0, sym.cos(angle)]])
    elif (axis=='z'):
        A = sym.Matrix([[sym.cos(angle), sym.sin(angle), 0],
                    [-sym.sin(angle), sym.cos(angle), 0],
                    [0, 0, 1]])
    else:
        print('error in rotate_matrix function')
    
    return A

def red(M): # return A
    '''
    Reduce or simplify a matrix M.
    
    Parameters
    ----------
    M : sympy.matrix [:, :]
    
    Returns
    -------
    A : sympy.matrix [:, :]
        M simplified
    '''
    A = M.copy().as_mutable()
    A = sym.simplify(A.doit().expand(trig=True))
    return A

def coeficient_matrix_par(E, q, b=None, c=None): # return M
    '''
    Collects the terms o equatios E[:] that are linearly proportional to each 
    of the terms in the array q[:] and arrange the equation into a matrix M
    such that:
    
    E[:] = M[:, :] . q[:]
    
    The equation is ran in parallel over the rows of E[:]
    
    Parameters
    ----------
    E : sympy.matrix [:]
        system of equations E[j] = 0
    q : sympy.matrix [:]
        list of sympy symbols, functions or expressions
    
    Returns
    -------
    M : sympy.matris [:, :]
        matrix M satisfying equation E[:] = M[:, :] . q[:]
    '''
    
    # Function that will be executed in parallel
    def func_M(E, q, i, b=None, c=None):
        #--
        toto = sym.symbols('toto')
        if (b is None): b = [toto for i in range(len(q))]
        if (c is None): c = [toto for i in range(len(q))]
        #--
        M = sym.zeros(len(q),1)
        for j in range(q.shape[0]):
            t_1 = time.time()
            M[j] = E.coeff(b[j], n=0).coeff(c[j], n=0).coeff(q[j])
            t_2 = time.time();
            file_log = open('file_log.txt', 'a')
            print((i, j), end=' ', file=file_log)
            print(t_2 - t_1, file=file_log)
            file_log.close()
        #--
        return (i, M)

    # Executing the function in parallel
    totos = Parallel(n_jobs=n_cores, verbose=par_verbose)(delayed(func_M)(E[i], q, i, b, c) for i in range(len(q)) )

    # Collecting the values
    M = sym.zeros(len(q), len(q))
    for toto in totos:
        i = toto[0]
        for j in range(len(q)):
            M[i, j] = toto[1][j]
    del totos
    #--
    return M

def func_var_subs(var, dict_1, dict_2, i, j):
    '''
    Substitutes the expression in var using the dictionaries dict_1 and dict_2
    on this order. The counters i an j are used so this function can be called
    by parallelized functios. The ipython console cannot be called by 
    parallelised functions, so the log file is used to register the function
    progress.
    
    Parameters
    ----------
    var : sympy expression
    dict_1 : dictionary {key: value}
             key : sympy symbol, function or expression
             value : sympy symbol, function or expression
             the keys on dict_1 are substitued by corresponding values
    dict_2 : dictionary {key: value}
             key : sympy symbol, function or expression
             value : sympy symbol, function or expression
             the keys on dict_1 are substitued by corresponding values
    i : int
        counter used so the function can be called by parallelized functios
    j : int
        counter used so the function can be called by parallelized functios
    
    Returns
    -------
    (i, j var) : tuple
                 i : int
                     counter passed to the function as a paremeter so the
                     function can be called by parallelized functios
                 j : int
                     counter passed to the function as a paremeter so the
                     function can be called by parallelized functios
                 var : sympy expression
                       expression passed to the function where the keys in
                       dict_1 and dict_2 have been replace by their
                       correspondent values on this order, i.e. firt dict_1 and
                       then dict_2.
    '''
    
    t_1 = time.time()
    #--
    var = var.expand(trig=True).subs(dict_1).subs(dict_2)
    #--
    t_2 = time.time()
    file_log = open('file_log.txt', 'a')    
    print((i, j), end=' ', file=file_log)
    print(t_2 - t_1, file=file_log)
    file_log.close()
    return (i, j, var)

def assumptions(M, dict_1, dict_2={}):
    '''
    Substitutes the expressions in each term in the matrix M[i, j] using the
    dictionaries dict_1 and dict_2 on this order. The dictionaries contain
    assumptions used to simplify the matrix. The dictionaries can contain
    symbolic symbols, functions or expressions such as multiplications.
    
    The function func_var_subs is ran in parallel over the physical cores in to
    accelerate the process. The ipython console cannot be called by 
    parallelised functions, so the log file is used to register the function
    progress.
    
    Paramters
    ---------
    M : sympy.matrix [:, :]
        matrix of sympy expressions
    dict_1 : dictionary {key: value}
             key : sympy symbol, function or expression
             value : sympy symbol, function or expression
             the keys on dict_1 are substitued by corresponding values
    dict_2 : dictionary {key: value}
             key : sympy symbol, function or expression
             value : sympy symbol, function or expression
             the keys on dict_1 are substitued by corresponding values
    
    Returns
    -------
    M_res : sympy.matrix [:, :]
            matrix  passed to the function where the keys in dict_1 and dict_2
            have been replace by their correspondent values on this order, i.e.
            firt dict_1 and then dict_2.
    '''
    
    var = M.copy()
    totos = Parallel(n_jobs=n_cores, verbose=par_verbose)(delayed(func_var_subs)(var[i, j], dict_small, dict_small_squared, i, j) for i in range(M.shape[0]) for j in range(M.shape[1]) )
    M_res = sym.zeros(M.shape[0], M.shape[1])
    for toto in totos:
        i = toto[0]
        j = toto[1]
        M_res[i, j] = toto[2]
    del totos
    return M_res

def assumptions_madd(M, dict_1, dict_2):
    '''
    Substitutes the expressions in each term in the matrix M[i, j] using the
    dictionaries dict_1 and dict_2 on this order. The dictionaries contain
    assumptions used to simplify the matrix. The dictionaries can contain
    symbolic symbols, functions or expressions such as multiplications.
    
    The function func_var_subs is ran in parallel over the physical cores in to
    accelerate the process. The ipython console cannot be called by 
    parallelised functions, so the log file is used to register the function
    progress.
    
    Paramters
    ---------
    M : sympy.matrix [:, :]
        matrix of sympy expressions
        type(M[i, j]).__name__ must be Add for all i and j in M[:, :]
    dict_1 : dictionary {key: value}
             key : sympy symbol, function or expression
             value : sympy symbol, function or expression
             the keys on dict_1 are substitued by corresponding values
    dict_2 : dictionary {key: value}
             key : sympy symbol, function or expression
             value : sympy symbol, function or expression
             the keys on dict_1 are substitued by corresponding values
    
    Returns
    -------
    M_res : sympy.matrix [:, :]
            matrix  passed to the function where the keys in dict_1 and dict_2
            have been replace by their correspondent values on this order, i.e.
            firt dict_1 and then dict_2.
    '''
    
    possible = True
    for i_m in range(M.shape[0]):
        for j_m in range(M.shape[1]):
            if (type(M[i_m, j_m]).__name__ != 'Add'):
                possible = False
    if (not possible):
        print("I'm sorry Dave, I'm afraid I can't do that", file=sys.stderr)
    
    M_res = M.copy()
    for i_m in range(M.shape[0]):
        for j_m in range(M.shape[1]):
            arguments = list(M_res[i_m, j_m].args)
            totos = Parallel(n_jobs=n_cores, verbose=par_verbose)(delayed(func_var_subs)(arguments[i], dict_small, dict_small_squared, i, 0) for i in range(len(arguments)) )
            for toto in totos:
                i = toto[0]
                _ = toto[1]
                arguments[i] = toto[2]
            del totos
            M_res[i_m,j_m] = M_res[i_m,j_m].func(*arguments)
            
    return M_res

def func_var_simplify(var, i, j):
    '''
    This function calls the sympy methos doit, expand and simplify in order to
    reduce expression complexity without making assumptions. The counters i an
    j are used so this function can be called by parallelized functios. The
    ipython console cannot be called by parallelised functions, so the log file
    is used to register the function progress.
    
    Parameters
    ----------
    var : sympy expression
    i : int
        counter used so the function can be called by parallelized functios
    j : int
        counter used so the function can be called by parallelized functios
    
    Returns
    -------
    (i, j var) : tuple
                 i : int
                     counter passed to the function as a paremeter so the
                     function can be called by parallelized functios
                 j : int
                     counter passed to the function as a paremeter so the
                     function can be called by parallelized functios
                 var : sympy expression
                       expression simplified
    '''
    t_1 = time.time()
    #--
    var = var.doit().expand(trig=True).doit().simplify()
    #--
    t_2 = time.time()
    file_log = open('file_log.txt', 'a')    
    print((i, j), end=' ', file=file_log)
    print(t_2 - t_1, file=file_log)
    file_log.close()
    return (i, j, var)

def matrix_simplify(M):
    '''
    This function calls the sympy methos doit, expand and simplify in order to
    reduce the complexity of each term of the matrix M[:, :] without making 
    assumptions.
    
    The function func_var_subs is ran in parallel over the physical cores in to
    accelerate the process. The ipython console cannot be called by
    parallelised functions, so the log file is used to register the function
    progress.
    
    Paramters
    ---------
    M : sympy.matrix [:, :]
        matrix of sympy expressions
    
    Returns
    -------
    M_s : sympy.matrix [:, :]
          matrix  passed to the function simplified
    '''
    
    var = M.copy()
    totos = Parallel(n_jobs=n_cores, verbose=par_verbose)(delayed(func_var_simplify)(var[i, j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1]) )
    M_s = sym.zeros(M.shape[0], M.shape[1])
    for toto in totos:
        i = toto[0]
        j = toto[1]
        M_s[i, j] = toto[2]
    del totos
    return M_s

def func_subs_integral(var, i, j, dz, z, R):
    '''
    This function calls the integrate method over the expression in var. The
    terms linearly proportional to dz are selected, and the expression is
    integrated from z to R. The counters i an j are used so this function can
    be called by parallelized functios. The ipython console cannot be called by
    parallelised functions, so the log file is used to register the function
    progress.
    
    Parameters
    ----------
    var : sympy expression
          the expression must contain at least one term containing dz
    i : int
        counter used so the function can be called by parallelized functios
    j : int
        counter used so the function can be called by parallelized functios
    dz : sympy.Symbol
         infinitesimal integration variable
    z : sympy.Symbol
        integration variable
    R : sympy.Symbol
        upper integration limit
        
    Returns
    -------
    (i, j, var) : tuple
                 i : int
                     counter passed to the function as a paremeter so the
                     function can be called by parallelized functios
                 j : int
                     counter passed to the function as a paremeter so the
                     function can be called by parallelized functios
                 var : sympy expression
                       sympy expression integraed       
    '''
    
    t_1 = time.time()
    #--
    var = (var.coeff(dz).integrate((z,0,R), manual=True) + var.coeff(dz,n=0)).expand().doit()
    #--
    t_2 = time.time()
    with open('file_log.txt', 'a') as file_log:
        print('(%i, %i) %0.4f' %(i, j, t_2-t_1), file=file_log )
    return (i, j, var)

def func_integrate_par(expr, dz, z, R):
    '''
    This function calls the integrate method over the expression in var. The
    terms linearly proportional to dz are selected, and the expression is
    integrated from z to R. The expression must be constituted of additive
    terms, i.e. type(expr.core.add.Add) == sympy.core.add.Add must be True
    
    The function unc_subs_integral is ran in parallel over the physical cores
    in to accelerate the process. The ipython console cannot be called by
    parallelised functions, so the log file is used to register the function
    progress.
    
    Parameters
    ----------
    expr : sympy expression of the type sympy.core.add.Add
           the expression must contain at least one term containing dz
    dz : sympy.Symbol
         infinitesimal integration variable
    z : sympy.Symbol
        integration variable
    R : sympy.Symbol
        upper integration limit
        
    Returns
    -------
    expr_int : sympy expression
               sympy expression integraed       
    '''
    
    if ( type(expr).__name__ == 'Add' ):
        arguments = list(expr.args)
        totos = Parallel(n_jobs=n_cores, verbose=par_verbose)(delayed(func_subs_integral)(arguments[i], i, 0, dz, z, R) for i in range(len(arguments)) )
        for toto in totos:
            i = toto[0]
            # j = toto[1]
            arguments[i] = toto[2]
        del totos
        expr_int = expr.func(*arguments)
    else:
        expr_int = []
        sys.exit("expr type is not sympy.core.add.Add")
    return expr_int

def lhs_print(lhs_str, rhs):
    '''
    This function displays the sympy equations adding the left hand side
    defined in the input lhs_str.
    
    Parameters
    ----------
    lhs_str : string
              left hand side of the equation on latex notation
    rhs : sympy expression
    '''
    
    lhs = sym.Symbol(lhs_str)
    # lhs.name = lhs_str
    
    display(sym.relational.Eq(lhs, rhs, evaluate=False))


#%% Symbolic constants
'''
The symbolic constants must be explicitly declared so that python nows their
class methods and instances. Here we understand as constant a symblic variable
that is not explicitly function of time. However, they may vary from time step
to time step.
'''

z  = sym.symbols('z') # [m] blade section centre z coordinate (local frame)
dz = sym.symbols('dz')
R  = sym.symbols('R') # [m] rotor radius
m_n = sym.symbols('m_n') # [kg] nacelle mass
m_h = sym.symbols('m_h') # [kg] hub mass

I_x = sym.symbols('I_x') # [kg*m**2] nacelle moment of inertia around x axis (local frame)
I_z = sym.symbols('I_z') # [kg*m**2] nacelle moment of inertia around z axis (local frame)

h_t = sym.symbols('h_t') # [m] nacelle height
s_l = sym.symbols('s_l') # [m] shaft length
g = sym.symbols('g')     
eta = sym.symbols('eta') # [rad] blade initial azimuthal position
t = sym.symbols('t')     # [s] time
k_x = sym.symbols('k_x') # [N/m] tower top equivalent linear stiffness on x axis (local frame)
k_y = sym.symbols('k_y') # [N/m] tower top equivalent linear stiffness on y axis (local frame)

Gt_x = sym.symbols('Gt_x') # [N*m/rad] tower top equivalent angular stiffness around x axis (local frame)
Gt_z = sym.symbols('Gt_z') # [N*m/rad] tower top equivalent angular stiffness around y axis (local frame)
Gt_xy = sym.symbols('Gt_xy')
Gs_y = sym.symbols('Gs_y') # [N*m] shaft angular stiffness around y axis (local frame)

omega = sym.Matrix(sym.symbols('omega_0:%i'%(n_m))) # blade natural frequencies

tilt = sym.symbols('theta_tilt') # [rad] shaft tilt angle
cone = sym.symbols('theta_cone') # [rad] blades cone angle
pitch = sym.symbols('theta_pitch') # [rad] blades pitch angle

pi = sym.pi # [-] pi constant
g = sym.symbols('g') # [kg*m/s**2] gravity

'''
Sometimes is convenient to use a different nomenclature to print the latex on
the Ipython console and for code generation. The dict_names gather the list of
variables where the names are changed before generating the numrical code.
'''

dict_names = {omega[i]: 'omega[%i]' %i for i in range(n_m)}
dict_names = {**dict_names, tilt: 'tilt', cone: 'cone', pitch: 'pitch'}

#%% Funtions of z

'''
Symbolic variables that are an undefined function of the blade length z.
'''

# Blade section centre [m]
x = sym.Function('x')(z) # [m] blade section centre x coordinate (local frame)
y = sym.Function('y')(z) # [m] blade section centre y coordinate (local frame)

# Blade section mass per unit of lenght [kg/m]
m = sym.Function('m')(z)
m_a = sym.Function('m_a')(z)

# Blade mode shapes
phi_x = sym.zeros(n_m,1)  # matrix of mode shapes in the x direction [phi_0_x, phi_1_x, phi_2_x].T [m]
phi_y = sym.zeros(n_m,1)  # matrix of mode shapes in the x direction [phi_0_y, phi_1_y, phi_2_y].T [m]
for i in range(phi_x.shape[0]):
    phi_x[i] = sym.Function(('phi_x[%s,:]' %str(i)))(z)
    phi_y[i] = sym.Function(('phi_y[%s,:]' %str(i)))(z)

# Aerodynamic forces per unit length (generic blade)
f_x_b = sym.Function('f_b_x')(z) # force per unit length on x direction (for a generic blade) [N/m]
f_y_b = sym.Function('f_b_y')(z) # force per unit length on y direction (for a generic blade) [N/m]
f_z_b = sym.Function('f_b_z')(z) # force per unit length on z direction (for a generic blade) [N/m]
f_b = sym.Matrix([f_x_b, f_y_b, f_z_b]) # matrix with aerodynamic foreces [f_x_b, f_y_b, 0] (for a generic blade) [N/m]

# Aerodynamic forces per unit length (blades 0, 1 and 2)
f_x_0 = sym.Function('f_0_x')(z) # force per unit length on x direction (for blade 0) [N/m]
f_y_0 = sym.Function('f_0_y')(z) # force per unit length on y direction (for blade 0) [N/m]
f_z_0 = sym.Function('f_0_z')(z) # force per unit length on z direction (for blade 0) [N/m]
f_x_1 = sym.Function('f_1_x')(z) # force per unit length on x direction (for blade 1) [N/m]
f_y_1 = sym.Function('f_1_y')(z) # force per unit length on y direction (for blade 1) [N/m]
f_z_1 = sym.Function('f_1_z')(z) # force per unit length on z direction (for blade 1) [N/m]
f_x_2 = sym.Function('f_2_x')(z) # force per unit length on x direction (for blade 2) [N/m]
f_y_2 = sym.Function('f_2_y')(z) # force per unit length on y direction (for blade 2) [N/m]
f_z_2 = sym.Function('f_2_z')(z) # force per unit length on z direction (for blade 2) [N/m]

# Aerodynamic force vectors (blades 0, 1 and 2)
f_0 = sym.Matrix([f_x_0, f_y_0, f_z_0]) # matrix with foreces [f_x_b, f_y_b, 0] (for blade 0) [N/m]
f_1 = sym.Matrix([f_x_1, f_y_1, f_z_1]) # matrix with foreces [f_x_b, f_y_b, 0] (for blade 1) [N/m]
f_2 = sym.Matrix([f_x_2, f_y_2, f_z_2]) # matrix with foreces [f_x_b, f_y_b, 0] (for blade 2) [N/m]

#%% Functions of time

'''
Symbolic variables that are an undefined function of time t, their first and
second derivatives.
'''

# Time
t = sym.symbols('t') # time [s]

# Degrees of freedom
q = sym.Matrix(dynamicsymbols('q[0:%i]'%(3*n_m+5))) # matrix of degrees of freedom [q_0, q_1, q_2, ..., q_11].T
q_dot = q.diff(t)                        # (d/dt) [q_0, q_1, q_2, ..., q_11].T
q_ddot = q_dot.diff(t)                   # (d**2/dt**2) [q_0, q_1, q_2, ..., q_11].T

# Degrees of freedom (generic blade)
qb = sym.Matrix(dynamicsymbols('qb_0:%i'%(n_m))) # matrix of blade degrees of freedom (for a generic blade) [qb_0, qb_1, qb_2].T
qb_dot = qb.diff(t)                       # (d/dt) [qb_0, qb_1, qb_2].T
qb_ddot = qb_dot.diff(t)                  # (d**2/dt**2) [qb_0, qb_1, qb_2].T

# Shaft azimuth angle
theta = dynamicsymbols('theta') # azimuth position of blade 0 [rad]
theta_dot = theta.diff(t)       # (d/dt) theta [rad/s]
theta_ddot = theta_dot.diff(t)  # (d**2/dt**2) theta [rad/s**2]

# Rotor angular velocity
Omega = dynamicsymbols('Omega') # (d/dt) rotor angular velocity [rad/s]
Omega_dot = Omega.diff(t)       # (d**2/dt**2) Omega [rad/s**2]
Omega_ddot = Omega_dot.diff(t)

# Update the names dictionary
dict_names = {**dict_names, **{qb[j]: 'q[%i*i_b+%i]' %(n_m,j) for j in range(n_m)}}

#%% Dictionaries

# Generic blade to blade 0, 1 and 2 respectively
dict_b0 = { eta: pi,                   **dict(zip(qb[0:n_m],q[0:n_m])), **dict(zip(f_b[:], f_0[:])) }
dict_b1 = { eta: sym.Rational(5,3)*pi, **dict(zip(qb[0:n_m],q[n_m:2*n_m])), **dict(zip(f_b[:], f_1[:])) }
dict_b2 = { eta: sym.Rational(7,3)*pi, **dict(zip(qb[0:n_m],q[2*n_m:3*n_m])), **dict(zip(f_b[:], f_2[:])) }

# List of variables that vary with iterations
iter_list = list(f_0[:2] + f_1[:2] + f_2[:2] + q[:] +
                 q_dot[:] + q_ddot[:] + [theta, theta_dot, theta_ddot] + 
                 [Omega, Omega_dot, Omega_ddot])

# Small deflections and small deflections derivatives assumption
## sin(small) = small, cos(small) = 1.0
list_small = [*q[:], *q_dot[:]]
dict_small = {} # empty dictionary
for small in list_small:
    dict_small.update({sym.sin(small): small, sym.cos(small): 1, sym.tan(small): small})        

## small*small = 0.0
list_small_squared = [] # empty list
for small1 in list_small:
    for small2 in list_small:
        list_small_squared = list_small_squared + [small1 * small2]
set_small_squared = set(list_small_squared) # eliminate repetead items by using the set data structure
dict_small_squared = {small: 0 for small in set_small_squared}

# The integration of orthogonal mode shapes is zero
dict_mode_shapes_product = {}
for i in range(len(phi_x)):
    for j in range(len(phi_x)):
        if (i!=j):
            dict_mode_shapes_product.update({phi_x[i]*phi_x[j]: 0, phi_y[i]*phi_y[j]: 0})

#%% Assumptions

# tilt = 0
# cone = 0
# g = 0

# ut_x = q[3*n_m]
ut_y = q[3*n_m+0]
# theta_tz = q[3*n_m+2]
# theta_sy = q[3*n_m+3]

# theta_tx = 0
# q[-1] = 0

ut_x, theta_tx, theta_tz, theta_sy = [0 for i in range(4)]
for i in range(4):
    q[-(i+1)] = 0
    q_dot[-(i+1)] = 0
    q_ddot[-(i+1)] = 0


#%% Transformation tensors

# tower top rotation (due to tower deformation)
A_01 = red( rotate_matrix(theta_tx, 'x').T * rotate_matrix(theta_tz, 'z').T ).T

# Shaft tilt
A_12 = rotate_matrix(tilt, 'x')

# Shaft tip azimutal position (blade 0)
A_23 = rotate_matrix(theta + eta + theta_sy, 'y') # rotor azimutal position (blade 0)

# Rotor conning
A_34 = rotate_matrix(cone, 'x')  

# Combined transformation tensors
A_04 = red( A_01.T * A_12.T * A_23.T * A_34.T ).T

lhs_print('A_01', A_01)

transformation_matrices = {}
transformation_matrices['A_01'] = A_01.xreplace({theta_dot: Omega})
transformation_matrices['A_12'] = A_12.xreplace({theta_dot: Omega})
transformation_matrices['A_23'] = A_23.xreplace({theta_dot: Omega})
transformation_matrices['A_34'] = A_34.xreplace({theta_dot: Omega})

#%% Transformation tensor derivatives

Omega_01_1 = matrix_simplify( A_01 * A_01.T.diff(t) )
Omega_01_3 = matrix_simplify( A_23*A_12 * Omega_01_1 * A_12.T*A_23.T )
Omega_01_4 = matrix_simplify( A_34*A_23*A_12 * Omega_01_1 * A_12.T*A_23.T*A_34.T )

Omega_23_3 = matrix_simplify( A_23 * A_23.T.diff(t) )
Omega_23_4 = matrix_simplify( A_34 * Omega_23_3 * A_34.T )

print('Omega_01_1 ='); display(Omega_01_1); print('\n')
print('Omega_23_3 ='); display(Omega_23_3); print('\n')

rotation_matrices = {}
rotation_matrices['Omega_01_1'] = Omega_01_1.xreplace({theta_dot: Omega})
rotation_matrices['Omega_01_3'] = Omega_01_3.xreplace({theta_dot: Omega})
rotation_matrices['Omega_01_4'] = Omega_01_4.xreplace({theta_dot: Omega})
rotation_matrices['Omega_23_3'] = Omega_23_3.xreplace({theta_dot: Omega})
rotation_matrices['Omega_23_4'] = Omega_23_4.xreplace({theta_dot: Omega})

#%% Local position vectors

# Tower top position (frame 0)
r_t = sym.Matrix([ ut_x, ut_y, -h_t])

# Shaft end position (frame 3)
r_s = sym.Matrix([ 0, -s_l, 0])

# Blade section position (frame 5)
r_b = sym.Matrix([qb[0:n_m,0].T*phi_x, qb[0:n_m,0].T*phi_y, [z]]) + sym.Matrix([x, y, 0])

print('r_t = '); display(r_t); print('\n')
print('r_s = '); display(r_s); print('\n')
print('r_b = '); display(r_b); print('\n')

position_vectors = {}
position_vectors['r_t0'] = r_t
position_vectors['r_s3'] = r_s
position_vectors['v_t0'] = r_t.diff(t)

#%% Total position vectors

# Total position (frame 0)
r_0 = r_t + red(A_01.T*A_12.T*A_23.T) * (r_s + A_34.T * r_b)
r_0 = r_0.expand()

# Total position (frame 6)
r_4 = A_04*r_t + A_34*r_s + r_b

# Dictionary substitutions
r_0_b0 = r_0.xreplace(dict_b0)
r_0_b1 = r_0.xreplace(dict_b1)
r_0_b2 = r_0.xreplace(dict_b2)
r_4_b0 = r_4.xreplace(dict_b0)
r_4_b1 = r_4.xreplace(dict_b1)
r_4_b2 = r_4.xreplace(dict_b2)

#%% Absolute blade velocity vector in blade frame (frame 5)

v_4 = A_04*r_t.diff(t) + A_34*(Omega_01_3 + Omega_23_3)*r_s + (Omega_01_4 + Omega_23_4)*r_b + r_b.diff(t)             
v_4 = v_4.expand()

#%% Kinectic energy

# Velocity squared (generic blade)
vs = (v_4.T * v_4)[0].doit().expand()

# Eliminating the product of orthogonal mode-shapes
vs = vs.subs(dict_mode_shapes_product)

# Velocity squared (blades 0, 1 and 2)
vs_b0 = vs.xreplace(dict_b0)
vs_b1 = vs.xreplace(dict_b1)
vs_b2 = vs.xreplace(dict_b2)

# Tower kinetic energy
E_kin_0 = sym.Rational(1, 2)*(m_n+m_h)*( sym.diff(ut_x, t)**2 + sym.diff(ut_y, t)**2 ) + sym.Rational(1, 2)*(I_x*sym.diff(theta_tx, t)**2 + I_z*sym.diff(theta_tz, t)**2)

# Blades kinetic energy
E_kin_1 = sym.Rational(1, 2)*(m*(vs_b0+vs_b1+vs_b2) *dz)

# Total kinetic 
E_kin = E_kin_0 + E_kin_1
E_kin = E_kin.doit().expand()

# Integrating over the blade length
E_kin = func_integrate_par(E_kin, dz, z, R)

#%% Potential energy 

# Tower potential energy
E_pot_T = sym.Rational(1, 2)*( k_x*ut_x**2 + k_y*ut_y**2 + Gt_x*theta_tx**2 + Gt_z*theta_tz**2 + Gs_y*theta_sy**2 + Gt_xy*ut_y*theta_tx )

# Blade stiffness potential energy (generic blade)
E_pot_m = 0
for i in range(n_m):
    E_pot_m += sym.Rational(1, 2)*(omega[i]**2*m*((phi_x[i]**2+phi_y[i]**2)*dz)*qb[i]**2)

# Centrifugal stiffening of the blade (generic blade)
E_pot_c = sym.Rational(1, 2)* Omega**2 * sym.cos(cone)**2 * m_a * (sym.Matrix(r_b.diff(z)[:2]).T * sym.Matrix(r_b.diff(z)[:2]))[0] * dz

# Blades gravitational potential energy  (generic blade)
E_pot_g = (m * g * (-r_0[2]) * dz)
E_pot_b = E_pot_m + E_pot_g # + E_pot_c
del E_pot_m, E_pot_g

# Blades potential energy (blades 0, 1 and 2)
E_pot_b0 = E_pot_b.xreplace(dict_b0)
E_pot_b1 = E_pot_b.xreplace(dict_b1)
E_pot_b2 = E_pot_b.xreplace(dict_b2)

# Total potential energy
E_pot = E_pot_T + E_pot_b0 + E_pot_b1 + E_pot_b2
E_pot = E_pot.doit().expand()

# Integrating over the blade length
E_pot = func_integrate_par(E_pot, dz, z, R)

#%% Non-conservative forces

# Non-conservative work done by aerodynamic forces on the blade (generic blade)
W_b = ((f_b.T * r_4)[0] * dz).doit().expand()

# Non-conservative work done by aerodynamic forces on the blade (blades 0, 1 and 2)
W_b0 = W_b.xreplace(dict_b0)
W_b1 = W_b.xreplace(dict_b1)
W_b2 = W_b.xreplace(dict_b2)

# Total non-conservative work
W = (W_b0 + W_b1 + W_b2).doit().expand()

# Integrating over the blade length
W = func_integrate_par(W, dz, z, R)

#%% Gathering the similar integral terms on the E_kin, E_pot and W

# List of integrals in E_kin, E_pot and W
i_set = E_kin.atoms(sym.Integral) | E_pot.atoms(sym.Integral) | W.atoms(sym.Integral)
i_list = list(i_set)

# Arrange the list of itegrals into a dictionary
x_list = sym.symbols('xint[:%d]' %len(i_list))
dict_xi = dict(zip(x_list, i_list))
dict_ix = dict(zip(i_list, x_list))

# Replace the integrals on E_kin, E_pot and W
E_kin = E_kin.xreplace(dict_ix)
E_pot = E_pot.xreplace(dict_ix)
W = W.xreplace(dict_ix)

#%% Lagrange equations

# Find the length of q
for i_q in range(len(q)):
    if (q[i_q]!=0):
        len_q = i_q+1

q = q[:len_q,:]
q_dot = q_dot[:len_q,:]
q_ddot = q_ddot[:len_q,:]

E = sym.zeros(q.shape[0], q.shape[1])
def func_lagrange(E, E_kin, E_pot, q, t, i):
    q_dot = q.diff(t, 1)
    t_1 = time.time()
    #--
    E = sym.diff(sym.diff(E_kin, q_dot[i]), t) - sym.diff(E_kin, q[i]) + sym.diff(E_pot, q[i]) - sym.diff(W, q[i])
    E = E.doit().subs({theta_ddot: 0}).subs({theta_dot: Omega}).doit().expand(trig=True)
    #--
    t_2 = time.time()
    file_log = open('file_log.txt', 'a')
    print((i, ), end=' ', file=file_log)
    print(t_2 - t_1, file=file_log)
    file_log.close()
    return (i, E)

totos = Parallel(n_jobs=n_cores, verbose=51)(delayed(func_lagrange)(E[i], E_kin, E_pot, q, t, i) for i in reversed(range(len(q))) )

for toto in totos:
    E[toto[0]] = toto[1]
del totos

#%% Assmptions (small deflectons and small deflections derivatives)

# t_1 = time.time()
# Ec = assumptions_madd(E, dict_small, dict_small_squared)
# t_2 = time.time()
# print(t_2-t_1)

Ec = E.copy()

#%% Matrices

print('Matrices')

print('M_matrix')
M_mat = coeficient_matrix_par(Ec, q_ddot)

print('C_matrix')
C_mat = coeficient_matrix_par((Ec - M_mat*q_ddot).doit().expand(trig=True), q_dot)

print('K_matrix')
K_mat = coeficient_matrix_par((Ec - M_mat*q_ddot - C_mat*q_dot).doit().expand(trig=True), q)

print('R_vector')
R_vec = (Ec - M_mat*q_ddot - C_mat*q_dot - K_mat*q).doit().expand(trig=True)

#%% Simplifying

print('Simplifying')
M_mat = matrix_simplify(M_mat)
C_mat = matrix_simplify(C_mat)
K_mat = matrix_simplify(K_mat)
R_vec = matrix_simplify(R_vec)

#%% Substitute the integrals back for pretty printing

M_i = M_mat.xreplace(dict_xi)
C_i = C_mat.xreplace(dict_xi)
K_i = K_mat.xreplace(dict_xi)
R_i = R_vec.xreplace(dict_xi)

#%% Gathering the trigonometric functions

# List the common trigonometric trigonetric functions on M, C, K and R
trig_set = M_mat.atoms(sym.sin, sym.cos) | C_mat.atoms(sym.sin, sym.cos) | K_mat.atoms(sym.sin, sym.cos) | R_vec.atoms(sym.sin, sym.cos)
trig_list = list(trig_set)

# Arrange the list of trigonometric functions in a dictionary
y_list = sym.symbols('ytrig[:%d]' %len(trig_list))
dict_yt = dict(zip(y_list, trig_list))
dict_ty = dict(zip(trig_list, y_list))

# Replace the trigonometric functions on M, C, K and R
M_mat = M_mat.xreplace(dict_ty)
C_mat = C_mat.xreplace(dict_ty)
K_mat = K_mat.xreplace(dict_ty)
R_vec = R_vec.xreplace(dict_ty)

#%% Integrals that must be recalculated at each time step or iteration

is_constant = [True]*len(x_list)
for i in range(len(x_list)):
    for atom in i_list[i].atoms(sym.Symbol, sym.Function, sym.Derivative):
        for iter_variable in iter_list:
            if (atom == iter_variable):
                is_constant[i] = False
    pass
dict_constant = dict(zip(x_list, is_constant))

#%%

Ei = Ec.copy().xreplace(dict_xi)

#%%

for key, value in dict_names.items():
    key.name, dict_names[key] = value, key.name

#%%

# Generate a list of the names of all symbols and functions (except intrinsic functions like trigonometric functions)
set_1 = ( f_0.atoms(sym.Symbol, sym.Function) | f_1.atoms(sym.Symbol, sym.Function) | f_2.atoms(sym.Symbol, sym.Function) | {Omega} | Ei.atoms(sym.Symbol, sym.Function) | set(x_list) | set(y_list) | r_0.atoms(sym.Function, sym.Symbol) | v_4.atoms(sym.Function, sym.Symbol) )
set_2 = ( Ei.atoms(sym.sin, sym.cos) | r_0.atoms(sym.sin, sym.cos) | v_4.atoms(sym.cos, sym.sin) )
list_e = list(set_1 - set_2)
list_names = [item.name for item in list_e]  

# Change the symbols and functions names to include the 'self.' prefix
for item in list_e:
    item.name = 'self.' + item.name

# Generate the numerical code
MyPrinter.gen_file(M_mat , C_mat, K_mat, -R_vec, q, dict_xi, dict_yt, dict_constant, transformation_matrices, rotation_matrices, position_vectors)

# Change the symbols and functions names back to their original values
for item, item_name in zip(list_e, list_names):
    item.name = item_name

#%%

for key, value in dict_names.items():
    key.name, dict_names[key] = value, key.name

#%%

t_end = time.time()
print("The numpy.code 'wind_turbine_structual.py' was written, Dave")
print('The simulation took %0.4f [s], Dave' %(t_end-t_start))
