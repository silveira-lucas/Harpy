'''
"We human beings are still divided into nation states, but these states are
rapidly becoming a single global civilisation.", Jimmy Carter
'''

#%%

import numpy as np
from scipy.linalg import norm
from scipy.optimize import fmin
import xlrd
from scipy.interpolate import UnivariateSpline

import custom_functions.custom_functions as cf
from wind_box import WindBox
from wind_turbine_structural import WindTurbineStructural

#%%

class WindTurbine(WindTurbineStructural):
    '''
    This class creates objects which parameters describes the wind turbine
    properties. Its methods allows to read the properties external files and
    assign the values to the parameters.
    
    Attributes
    ----------
    NB : int 
         [-] number of blades
    P_rated : float
              [W] generator rated power
    u_cut_in : float
               [m/s] wind speed which the generator is turned on
    u_cut_out : float
                [m/s] wind speed which the generator is turned off
    h_t : float
            [m] rotor center hight
    s_l : float
                   [m] shaft length
    lmbda : float
            [-] tip-speed ratio
    Omega_min : float
                [rad/s] minimum generator angular speed
    Omega_max : float
                [rad/s] maximum generator angular speed
    yaw : float
          [rad] rotor yaw angle
    tilt : float
           [rad] shaft tilt angle
    cone : float
           [rad] rotor conning angle
    x : numpy.ndarray [:], type=float 
        [m] blade centerline x positions, including hub distance (on the blade 
        reference frame)
    y : numpy.ndarray [:], type=float 
        [m] blade centerline y positions, including hub distance (on the blade 
        reference frame)
    z : numpy.ndarray [:], type=float 
        [m] blade centerline z positions, including hub distance (on the blade 
        reference frame)
    twist : numpy.ndarray [:], type=float 
            [rad] aerodynamic twist angle around blade z axis (positive along z
            axis)
    chord : numpy.ndarray [:], type=float
            [m] chord value as funtion of z
    R : float
        [m] Rotor radius, i.e. max(z)
    airfoil : numpy.ndarray [:, :, :], type=float
              airfoil properties array [theta, property, section]
              property 0 : [rad] angle of attack
              property 1 : [-] lift coefficient
              property 2 : [-] drag coefficient
              property 3 : [-] moment coefficient
              property 4 : [-] Øye S. Dynamic stall f_s,st
              property 5 : [-] Øye S. Dynamic stall C_l,inv
              property 6 : [-] Øye S. Dynamic stall C_l,fs              
              ...
    TR : numpy.ndarray [:], dtype=float
         [%] blade thickness ratio as function of z, i.e. thickness/chord
    hub_R : float
            [m] hub radius
    r_b : numpy.ndarray [:, :, :], type = float
          [m] blade sections position vectors, i.e. [x, y, z]
    downwind : float
               [rad] azimuth position where the blade is deepest in the wake
    omega : float
            [rad/s] rotor angular speed
    theta : float
            [rad] rotor azimuth position
    pitch : float
            [rad] blade pitch angle
    label : str
            strig describing the wind turbine
    self.m : numpy.ndarray [:], type = float
             [kg/m] mass per unit length
    self.m_a : numpy.ndarray [:], type = float
               [kg*m] $\int_{z}^{z_{tip}}{m*z*dz}$
    self.eta : float
               [rad] blade initial azimuth position. The variable is allowed to
               be changed by routines iterating over the blades.
    m_n : float
          [kg] nacelle mass
    m_h : float
          [kg] hub mass
    I_x : float
          [kg*m**2] nacelle moment of inertia around it's x axis
    I_y : float
          [kg*m**2] nacelle moment of inertia around it's y axis
    I_z : float
          [kg*m**2] nacelle moment of inertia around it's y axis          
    k_x : float
          [N/m] k_x represents the tower top linear stiffness on the inertial x
          axis
    k_y : float
          [N/m] k_y represents the tower top linear stiffness on the inertial y
          axis
    Gt_x : float
           [N*m/rad] Gt_x represents the tower top angular stiffnes around the
           inertial x axis
    Gt_z : float
           [N*m/rad] Gt_z represents the tower top angular stiffnes around the
           inertial z axis
    Gt_xy ; float
            [kg*m/(s**2*rad)] Gt_xy represents the tower top coupled linear and
            angular stiffnes around the inertial z axis
    Gs_y : float
           [N*m/rad] Gs_y represents the shaft tip angular stiffnes around the
           its y axis
    omega_0 : numpy.np.array [:], dtype='float'
              [rad/s] blade natural angular frequencies
    phi_x : numpy.ndarray [:, :], float
              [-] blade first natural mode shape on its x axis. The mode shape
              is allowed to be projected on the pitch angle
              indices: [mode, i_z]
    phi_y : numpy.ndarray [:, :], float
              [-] blade first natural mode shape on its y axis. The mode shape
              is allowed to be projected on the pitch angle
              indices: [mode, i_z]
    f_0_x : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 0 on its x
            direction along the blade z direction.
    f_1_x : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 1 on its x
            direction along the blade z direction.
    f_2_x : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 2 on its x
            direction along the blade z direction.
    f_0_y : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 0 on its y
            direction along the blade z direction.
    f_1_y : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 1 on its y
            direction along the blade z direction.
    f_2_y : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 2 on its y
            direction along the blade z direction.
    f_0_z : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 0 on its z
            direction along the blade z direction.
    f_1_z : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 1 on its z
            direction along the blade z direction.
    f_2_z : numpy.ndarray [:], float
            [N/m] Aerodynamic force per unit lenght along the blade 2 on its z
            direction along the blade z direction.
    K_gen : float
            [N*m/(rad*s)**2] generator curve characteristic constant
    Cp_opt : float
             [-] optimal power coefficient
    Cp : float
         [-] optimal power coefficient             
    gen_eff : float
              [-] # generator efficiency. The generator efficiency is assumed
              constant, but it is allowed to be changed.
    n_g : float
          [-] gearbox ratio (>=1)
    I_rot : float
            [kg*m**2] rotor moment of inertia around the shaft tip y axis
    I_hub : float
            [kg*m**2] rotor hub moment of inertia around the shaft tip y axis
    g : float, default = 9.81
        [m/s**2] gravity
    ray_m : float, default = 0.0
            Rayleigh proportional damping factor for the mass matrix
    ray_k : float, default = 0.0
            Rayleigh proportional damping factor for the stiffness matrix
    '''
    
    def __init__(self):
        '''
        This init method only declares the variables as place as place holders,
        i.e. without declaring the type or allocating memory.The instance
        creator(s) methods will be implemented via classmethods for allowing
        more flexibility for different imput methods.
        '''
        
        # Initiate the structural class
        super().__init__()
        
        self.NB = [] 
        self.P_rated = [] 
        self.u_cut_in = [] 
        self.u_cut_out = [] 
        self.h_t = [] 
        self.s_l = [] 
        self.lmbda = [] 
        self.Omega_min = [] 
        self.Omega_max = []
        self.yaw = [] 
        self.tilt = [] 
        self.cone = [] 
        self.x = [] 
        self.y = [] 
        self.z = [] 
        self.twist = [] 
        self.chord = [] 
        self.R = [] 
        self.airfoil = [] 
        self.TR = [] 
        self.hub_R = [] 
        self.r_b = [] 
        self.downwind = [] 
        self.Omega = [] 
        self.theta = [] 
        self.pitch = [] 
        self.label = [] 
        
        self.m = []
        self.m_a = []
        self.eta = []
        
        self.m_n = []
        self.I_x = []
        self.I_y = []
        self.I_z = []

        self.k_x = []
        self.k_y = []
        self.Gt_x = []
        self.Gt_z = []
        self.Gt_xy = []
        self.Gs_y = []
        self.eta = []
        
        self.n_modes = None
        self.omega = []
        self.phi_x0 = []
        self.phi_y0 = []
        self.phi_x = []
        self.phi_y = []
        
        self.f_0_x = []
        self.f_1_x = []
        self.f_2_x = []
        self.f_0_y = []
        self.f_1_y = []
        self.f_2_y = []
        self.f_0_z = []
        self.f_1_z = []
        self.f_2_z = []
        
        self.K_gen = []
        self.Cp_opt = []
        self.gen_eff = []
        self.n_g = []
        self.I_rot = []
        self.I_hub = []
        
        self.ray_m = 0.0
        self.ray_k = 0.0

        self.g = 9.81
        
        self.pitch_min = 0.0 # [rad]
        self.pitch_max = 25.0 * (np.pi/180.0) # [rad]
        self.pitch_dot_max = 8.0 * (np.pi/180.0) # [rad/s]
        
        self.k_x = []
        self.k_y = []
        self.Gt_x = []
        self.Gt_z = []
        self.Gs_y = []
        
        self.is_stiff = False

    def read_blade_planform(self, blade_file, distribution='cos', sections=50):
        '''
        The method reads the blade planform file and initializes the related
        instance variables:
            
        self.x
        self.y
        self.z
        self.twist
        self.chord
        self.TR
        self.r_b
        
        It is assumed the blade planform file is writen in the form:
        
        blade_file columns : blade sections
        blade_file[:, 0] : half-chord x coordinate relative to hub center [m]
        blade_file[:, 1] : half-chord y coordinate relative to hub center [m]
        blade_file[:, 2] : half-chord z coordinate relative to hub center [m]
        blade_file[:, 3] : twist angle relative to hub center [deg]
        blade_file[:, 4] : chord length [m]
        blade_file[:, 4] : pitch axis after leading edge (x/c)
        blade_file[:, 5] : relative thickness
        
        Parameters
        ----------
        blade_file : str
                     blade planform file name
        distribution : str, optional, default: 'sin'
                       'sin' : the blade sections are distributed using a
                       sinuoisidal distribution, i.e. the sections are
                       concentrated at the blade tip.
                       'cos' : the blade sections are distributed using a
                       cosinuoisidal distribution, i.e. the sections are
                       concentrated at the blade root and tip
        sections : int, optional, defualt: 50 
                   number of sections
        '''
        
        # Read blade geometric properties
        df = np.loadtxt(blade_file) # blade geometric properties data-frame numpy.ndarray
        self.R = df[-1, 2]
        
        # The sections are divided using a specific distribution
        if (distribution=='sin'):
            theta = np.linspace(0., np.pi/2., sections)
            self.z = (self.R-self.hub_R)*np.sin(theta) + self.hub_R
        elif (distribution=='cos'):
            theta = np.linspace(0., np.pi, sections)
            self.z = (self.R-self.hub_R)*(1.0 - np.cos(theta))/2 + self.hub_R
        
        # Interpolate using the specific distribution
        self.x = np.interp(self.z, df[:, 2], df[:, 0])
        self.y = np.interp(self.z, df[:, 2], df[:, 1])
        self.twist = - np.interp(self.z, df[:, 2], df[:, 3]) * (np.pi/180.)
        self.chord = np.interp(self.z, df[:, 2], df[:, 4])
        self.TR = np.interp(self.z, df[:, 2], df[:, 6])
        
        # Differential of the blade pre-bend
        self.x_dot = np.interp(self.z, df[:, 2], np.gradient(df[:, 0], df[:, 2]))
        self.y_dot = np.interp(self.z, df[:, 2], np.gradient(df[:, 1], df[:, 2]))
        
        # Blade radius
        self.r_b = np.empty((len(self.z), self.NB), dtype=float)
        self.r_b[:, 0] = self.x
        self.r_b[:, 1] = self.y
        self.r_b[:, 2] = self.z

    def read_blade_structural(self, blade_structural_file, blade_modes, blade_frequencies):
        '''
        The method reads the blade structural file, reads the blade per unit
        lenght and interpolate it to the self.z[:] distribution.
        
        The method also reads the blade mode shapes file and interpolate them
        to the self.z[:] distribution
        
        The method also reads th blade natural angluar frequencies and
        initialise the corresponding insntance variables.
        
        Files formats:
        blade_frequencies : [:] in [rad/s], comment lines start with #
        blade_modes : [:, :]
                      column 0 : [m] blade section z coordinate
                      column 1 : [-] phi_0_x
                      column 2 : [-] phi_0_y
                      column 3 : [-] phi_0_twist
                      column 4 : [-] phi_1_x
                      column 5 : [-] phi_2_y
                      column 6 : [-] phi_3_twist
                      ⋮
        
        Parameters
        ----------
        blade_structural_file : str
                                file name
        blade_modes : str
                      file name
        blade_frequencies : str
                            file name
        '''
        
        # Blade geometric properties file
        df = np.loadtxt(blade_structural_file)
        self.m = np.interp(self.z, df[:, 2], df[:, 4])
        del df

        # $\int_{z}^{z_{tip}}{m(\zeta)*\zeta*d\zeta}$
        self.m_a = np.zeros(self.m.shape)
        for i_z in range(len(self.z)):
            self.m_a[i_z] = np.trapz(self.m[i_z:]*self.z[i_z:], self.z[i_z:])
                
        # Blade mode shapes file
        df = np.loadtxt(blade_modes)
        if (self.n_m is not None):
            n_modes = self.n_m
        else:
            n_modes = int((df.shape[1]-1)/3)
        
        self.phi_x0 = np.zeros((n_modes, len(self.z)))
        self.phi_y0 = np.zeros((n_modes, len(self.z)))
        for i_m in range(n_modes):
            self.phi_x0[i_m, :] = np.interp(self.z, df[:, 0], df[:, 3*i_m + 1*0 + 1])
            self.phi_y0[i_m, :] = np.interp(self.z, df[:, 0], df[:, 3*i_m + 1*1 + 1])
        
        # Derivatives of mode shapes
        self.phi_x0_dot = np.zeros((n_modes, len(self.z)))
        self.phi_y0_dot = np.zeros((n_modes, len(self.z)))
        self.phi_x0_ddot = np.zeros((n_modes, len(self.z)))
        self.phi_y0_ddot = np.zeros((n_modes, len(self.z)))
        
        for i_m in range(n_modes):            
            self.phi_x0_dot[i_m, :] = np.interp(self.z, df[:, 0], np.gradient(df[:, 3*i_m + 1*0 + 1], df[:, 0], edge_order=2))
            self.phi_y0_dot[i_m, :] = np.interp(self.z, df[:, 0], np.gradient(df[:, 3*i_m + 1*1 + 1], df[:, 0], edge_order=2))
            #
            self.phi_x0_ddot[i_m, :] = np.interp(self.z, df[:, 0], np.gradient(np.gradient(df[:, 3*i_m + 1*0 + 1], df[:, 0], edge_order=2), df[:, 0], edge_order=2))
            self.phi_y0_ddot[i_m, :] = np.interp(self.z, df[:, 0], np.gradient(np.gradient(df[:, 3*i_m + 1*1 + 1], df[:, 0], edge_order=2), df[:, 0], edge_order=2))        
        #
        del df
        
        # Blade natural angular frequencies files
        df = np.loadtxt(blade_frequencies)
        self.omega = df[0:n_modes]
    
    def read_tower_structural(self):
        '''
        Method still to be implemented
        '''
        
        self.m_n = []
        self.I_x = []
        self.I_y = []
        self.k_x = []
        self.k_y = []
        self.Gt_x = []
        self.Gt_z = []
        self.Gt_xy = []
        self.Gs_y = []
    
    def read_airfoil_files(self, ae_files, TR_data):
        '''
        The method reads the files containing the aerodinamic properties of the
        airfoils used to describe the blade. The aerodynamic properties are
        then initerpolated to the blade sections.
        
        Parameters
        ----------
        ae_files : numpy.ndarray, dtype='str'
                   array containing the names of the files containing the
                   airfoil properties
                   property 0 : [rad] angle of attack
                   property 1 : [-] lift coefficient
                   property 2 : [-] drag coefficient
                   property 3 : [-] moment coefficient
                   property 4 : [-] Øye S. Dynamic stall f_s,st
                   property 5 : [-] Øye S. Dynamic stall C_l,inv
                   property 6 : [-] Øye S. Dynamic stall C_l,fs              
        TR_data : numpy.ndarray, dtype='float'
                  [%] thickness ratio of the airfoils in ae_files
        '''
        
        # Reads the the first airfoil file and allocates memory for the airfoil_data variable
        df = np.loadtxt(ae_files[0])
        airfoil_data = np.zeros((df.shape[0], df.shape[1], len(TR_data)))
        
        # Reads the airfoils files
        for i_f in range(len(ae_files)):
            airfoil_data[:, :, i_f] = np.loadtxt(ae_files[i_f])
        
        # Allocates memory for the instance variable airfoil
        self.airfoil = np.zeros( (airfoil_data.shape[0], airfoil_data.shape[1], len(self.z)) )
        
        # Defines the airfoil properties interporlating linearly the airfoild_data properties
        # as a function of the thickess ratio
        for i_z in range(len(self.z)):
            for i_d1 in range(airfoil_data.shape[0]):
                for i_d2 in range(airfoil_data.shape[1]):
                    toto = np.array([airfoil_data[i_d1, i_d2 ,0], airfoil_data[i_d1, i_d2, 1], 
                            airfoil_data[i_d1, i_d2, 2], airfoil_data[i_d1, i_d2, 3],
                            airfoil_data[i_d1, i_d2, 4], airfoil_data[i_d1, i_d2, 5]])
                    self.airfoil[i_d1,i_d2,i_z] = np.interp(self.TR[i_z], TR_data, toto)
        
        for i_z in range(len(self.z)):
            self.airfoil[:, 0, i_z] = (np.pi/180.) * self.airfoil[:, 0, i_z] # angles in radians
    
    def downwind_azimuth(self): # return theta_max
        '''
        The method calculates the azimuth angle where a blade points downwind
        using the scipy.optmize.fmin function.
        '''
        
        A_01 = cf.rotate_tensor(self.yaw, 'z')
        A_12 = cf.rotate_tensor(self.tilt, 'x')
        
        # The function returns the -y position given an azimuth angle
        def downwind_y(theta):
            A_23 = cf.rotate_tensor(theta, 'y')
            r = (A_01.T @ A_12.T @ A_23.T) @ np.array([0., 0., 10.])
            return -r[1]
        
        # Calculate (-y) for a set of values between 0 and 2*pi rad and finds the
        # minimum among this set
        theta = np.linspace(0., 2.*np.pi, num=100)
        y = [downwind_y(theta[i]) for i in range(len(theta))]
        theta_max = theta[np.argmin(y)]
        
        # Finds the minimum of (-y) using the minimum find previously
        self.downwind = fmin(downwind_y, x0=theta_max, disp=False)[0]
    
    def generator_moment(self, Omega, rho=1.225): # return M_gen
        '''
        The method returns the generator moment applied on the shaft given the
        instant rotor angular velocity self.Omega.
        
        Parameters
        ----------
        Omega : float
                [rad/s] shaft speed at the generator side
        rho : float, optional, default = 1.225
              [kg/m**3] free-wind density.
        
        Returns
        -------
        M_gen : float
                [N*m] Moment applied by the generator on the shaft.
        '''
        
        # Generator characteristic curve constant
        A_rot = np.pi * self.R**2 # [m**2] rotor swept area
        self.K_gen = rho * A_rot * self.R**3 * self.Cp_opt / (2. * self.n_g * self.lmbda**3)

        # Generator momentent applied the the shaft []
        if (Omega < self.Omega_min):
            M_gen = 0.
        elif (Omega < self.Omega_max + 0.001):
            M_gen = self.K_gen * Omega**2
        else:
            M_gen = self.K_gen * self.Omega_max**2
        return M_gen
    
    def mode_shapes(self):
        '''
        The method projects the mode shapes to account for the blade pitch
        angle.
        '''
        
        c_pitch = np.cos(self.pitch)
        s_pitch = np.sin(self.pitch)
        
        n_modes = self.phi_x0.shape[0]
        self.phi_x = np.zeros(self.phi_x0.shape)
        self.phi_y = np.zeros(self.phi_x0.shape)
        self.phi_x_dot = np.zeros(self.phi_x0_dot.shape)
        self.phi_y_dot = np.zeros(self.phi_x0_dot.shape)
        self.phi_x_ddot = np.zeros(self.phi_x0_dot.shape)
        self.phi_y_ddot = np.zeros(self.phi_x0_dot.shape)
        for i_m in range(n_modes):
            self.phi_x[i_m, :] = self.phi_x0[i_m, :] * c_pitch + self.phi_y0[i_m, :] * s_pitch
            self.phi_y[i_m, :] = - self.phi_x0[i_m, :] * s_pitch + self.phi_y0[i_m, :] * c_pitch
            # 
            self.phi_x_dot[i_m, :] = self.phi_x0_dot[i_m, :] * c_pitch + self.phi_y0_dot[i_m, :] * s_pitch
            self.phi_y_dot[i_m, :] = - self.phi_x0_dot[i_m, :] * s_pitch + self.phi_y0_dot[i_m, :] * c_pitch
            # 
            self.phi_x_ddot[i_m, :] = self.phi_x0_ddot[i_m, :] * c_pitch + self.phi_y0_ddot[i_m, :] * s_pitch
            self.phi_y_ddot[i_m, :] = - self.phi_x0_ddot[i_m, :] * s_pitch + self.phi_y0_ddot[i_m, :] * c_pitch            
    
    @classmethod
    def construct(cls, wt_file=None, n_modes=None):
        '''
        The class method is an alternative constructor method. It reads the
        xls sheets and creates and WindTurbine instance based on the
        information contained in the xls files and files it points to.
        
        Parameters
        ----------
        cls : class
              WindTurbine
        wt_file : str, optional, default='./turbine_data/WT_general_properties.xlsx'
                  The wt_file sheets are read in the format of dictionaries,
                  i.e. sheet and rows order do not matter, only their
                  corresponding names.
        n_modes : int, optional
                  number of modes shapes per blade

        Retunrs
        -------
        obj : WindTurbine instance
              WindTurbine instance created based on the information contained
              in the xls wt_file and files it points to.
        '''
        
        # Create class object
        obj = cls()
        obj.n_m = n_modes
        
        # Check for input file
        if (wt_file is None):
            wt_file = './turbine_data/WT_general_properties.xlsx'
        
        # Overall properties
        doc = xlrd.open_workbook(wt_file, on_demand=True).sheet_by_name('Overall')
        keys = doc.col_values(0)
        values = doc.col_values(1)
        overall = dict(zip(keys, values))
        del doc, keys, values
        
        # Drive train properties
        doc = xlrd.open_workbook(wt_file, on_demand=True).sheet_by_name('Drive Train')
        keys = doc.col_values(0)
        values = doc.col_values(1)
        drive_train = dict(zip(keys, values))
        del doc, keys, values

        # Blade planform properties
        doc = xlrd.open_workbook(wt_file, on_demand=True).sheet_by_name('Blades')
        keys = doc.col_values(0)
        values = doc.col_values(1)
        blades = dict(zip(keys, values))
        del doc, keys, values                
        
        # Airfoils properties
        doc = xlrd.open_workbook(wt_file, on_demand=True).sheet_by_name('Airfoildata')
        keys = doc.col_values(0)
        values = doc.col_values(1)
        airfoil_data = dict(zip(keys, values))
        del doc, keys, values                
        
        obj.gen_eff = drive_train['Electrical Generator Efficiency'] * 10.0**(-2) #[-]
        obj.n_g = drive_train['Gearbox Ratio'] # [-]
        G = drive_train['Shaft shear modulus of elasticity'] 
        Ip = drive_train['Shaft polar moment of inertia']

        obj.NB = int(overall['Number of blades']) # [-]
        obj.P_rated = overall['Rated power']/(obj.gen_eff) * 10.0**(6) # [W]
        obj.u_cut_in = overall['Cut in wind speed'] # [m/s]
        obj.u_cut_out = overall['Cut out wind speed'] # [m/s]
        obj.h_t = overall['Hub Height'] # [m]
        obj.s_l = overall['Hub Overhang'] # [m]
        obj.lmbda = overall['Tip speed ratio'] # [-]
        obj.Omega_min = overall['Minimum Rotor Speed'] * (2.*np.pi/60.) # [rad/s]
        obj.Omega_max = overall['Maximum Rotor Speed'] * (2.*np.pi/60.) # [rad/s]
        obj.tilt = overall['Shaft Tilt Angle'] * (np.pi/180.) # [rad]
        obj.cone = - overall['Rotor Precone Angle'] * (np.pi/180.) # [rad]
        obj.hub_R = overall['Hub Diameter']/2.0 # [m]
        obj.yaw = overall['Yaw angle'] * (np.pi/180.) # [rad]
        obj.m_n = overall['Nacelle mass'] # [kg]
        obj.m_h = overall['Hub mass'] # [kg]
        obj.I_hub = overall['Hub Inertia About Shaft Axis'] # [kg*m**2]
        obj.I_x = overall['Nacelle Inertia About Yaw Axis'] # [kg*m**2]
        obj.I_y = overall['Nacelle Inertia About Lateral Axis'] # [kg*m**2]
        obj.I_z = overall['Nacelle Inertia About Yaw Axis'] # [kg*m**2]
        obj.Cp_opt = overall['Optimal Mechanical Aerodynamic Rotor Efficiency, Cp']
        obj.k_x = overall['Tower linear equivalent stiffness'] # [N/m]
        obj.k_y = overall['Tower linear equivalent stiffness'] # [N/m]
        obj.Gt_x = overall['Tower angular equivalent stiffness'] # [N*m/rad]
        obj.Gt_z = overall['Tower torsional equivalent stiffness'] # [N*m/rad]
        obj.Gs_y = G*Ip/obj.s_l # [N*m/rad]
        
        blade_planform_file = blades['Blade planform properties coarse']
        blade_structural_file = blades['Blade structural propeties']
        blade_modes_file = blades['Blade mode shapes']
        blade_frequencies_file = blades['Blade mode frequencies']
        obj.read_blade_planform(blade_planform_file)
        obj.read_blade_structural(blade_structural_file, blade_modes_file, blade_frequencies_file)
        
        # Inital shaft azimuth angle is zero
        obj.theta = 0.0 # [rad]
        
        # Inital angular rotor angular velocity is zero
        obj.Omega = 0.0
        
        # Inital forces on the blade are zero
        obj.f_0_x = 0.0 * obj.z
        obj.f_0_y = 0.0 * obj.z
        obj.f_0_z = 0.0 * obj.z
        obj.f_1_x = 0.0 * obj.z
        obj.f_1_y = 0.0 * obj.z
        obj.f_1_z = 0.0 * obj.z
        obj.f_2_x = 0.0 * obj.z
        obj.f_2_y = 0.0 * obj.z
        obj.f_2_z = 0.0 * obj.z
        
        # Initial pitch is zero
        obj.pitch = 0.0 # [rad]
        obj.mode_shapes()
        
        ae_files = np.array(list(airfoil_data.keys()))
        TR_data = np.array(list(airfoil_data.values()))
        obj.read_airfoil_files(ae_files, TR_data)
        obj.downwind_azimuth()

        # Rotor moment of inertia
        obj.I_rot = obj.I_hub + 3 * np.trapz(obj.m*obj.z**2, obj.z)
        
        # Calculating the generator constant
        _ = obj.generator_moment(0.)
        
        return obj

#%%

        
