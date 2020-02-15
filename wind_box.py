import numpy as np
from os import system as sys
import platform
import os.path
import h5py
from scipy.interpolate import RegularGridInterpolator
import xlrd
import custom_functions.custom_functions as cf

#from wind_turbine import WindTurbine

#%%

class WindSimu(object):
    '''
    '''
    
    def __init__(self):
        '''
        This init method only declares the variables as place as place holders,
        i.e. without declaring the type or allocating memory.The instance
        creator(s) methods will be implemented via classmethods for allowing
        more flexibility for different imput methods.
        '''
        
        self.n_dim = []
        self.n_vel = []
        self.n_0 = []
        self.n_1 = []
        self.n_2 = []
        self.l_0 = []
        self.l_1 = []
        self.l_2 = []
        self.terrain = []
        self.Mann_alphaepsilon = []
        self.Mann_L = []
        self.Mann_gamma = []
        self.seed = []
        self.input_file = []
        
    def write_input(self):
        '''
        This instance method writes the input file necessary for the Mann's
        WindSimu executable.
        '''
        
        input_file = self.input_file
        
        # windsimu output files names
        file_0 = input_file[:-4] + '0' + '.bin'
        file_1 = input_file[:-4] + '1' + '.bin'
        file_2 = input_file[:-4] + '2' + '.bin'
        
        # write wind input file
        with open(self.input_file, 'w') as f:
            print(self.n_dim, file=f)
            print(self.n_vel, file=f)
            print(self.n_0, file=f)
            print(self.n_1, file=f)
            print(self.n_2, file=f)
            print(self.l_0, file=f)
            print(self.l_1, file=f)
            print(self.l_2, file=f)
            print(self.terrain, file=f)
            print(self.Mann_alphaepsilon, file=f)
            print(self.Mann_L, file=f)
            print(self.Mann_gamma, file=f)
            print(self.seed, file=f)
            print(file_0, file=f)
            print(file_1, file=f)
            print(file_2, file=f)

    def simulate(self):
        '''
        This instance method calls the Mann's WindSimu executable. It must be
        mentioned that this method only works for now in UnixBSD and Linux
        operating systems and depends on wine.
        '''

        input_file = self.input_file
        
        operating_system = platform.system()
        
        if (operating_system == 'Linux'):
            sys('wine ./wind/windsimu.exe' + input_file)
        elif (operating_system == 'Darwin'):
            sys('wine ./wind/windsimu.exe ' + input_file)
        elif (operating_system == 'Windows'):
            sys('.\wind\windsimu.exe ' + input_file)
        
        # sys('wine wind/windsimu.exe ' + input_file)
    
    def read_bin(self):
        '''
        The instance method reads the binary file outputed by Mann's executable
        and outputs velocity components as numpy arrays        
        
        Returns
        -------
        u_0[:, :, :] : numpy.ndarray, dtype = 'float'
                       wind velocity flutuation in the x direction at the
                       [x, y, x] position
        u_1[:, :, :] : numpy.ndarray, dtype = 'float'
                       wind velocity flutuation in the y direction at the
                       [x, y, x] position
        u_2[:, :, :] : numpy.ndarray, dtype = 'float'
                       wind velocity flutuation in the z direction at the
                       [x, y, x] position
        '''
        
        # Input file name
        input_file = self.input_file
        
        # Binaray files names
        file_0 = input_file[:-4] + '0' + '.bin'
        file_1 = input_file[:-4] + '1' + '.bin'
        file_2 = input_file[:-4] + '2' + '.bin'

        # read windsimu raw files                
        with open(file_0, 'rb') as f:
            u0_raw = np.fromfile(f, np.float32)
        with open(file_1, 'rb') as f:
            u1_raw = np.fromfile(f, np.float32)
        with open(file_2, 'rb') as f:
            u2_raw = np.fromfile(f, np.float32)
        
        u_0 = np.zeros((self.n_0, self.n_1, self.n_2))
        u_1 = np.zeros((self.n_0, self.n_1, self.n_2))
        u_2 = np.zeros((self.n_0, self.n_1, self.n_2))
        
        itael = 0
        for i in range(self.n_0):
            for j in range(self.n_1):
                for k in range(self.n_2):
                    u_0[i, j, k] = u0_raw[itael]
                    u_1[i, j, k] = u1_raw[itael]
                    u_2[i, j, k] = u2_raw[itael]
                    itael += 1
        return u_0, u_1, u_2

#%%

class Mann_Turb_Box_64(object):
    
    def __init__(self):
        self.prefix = []
        self.Mann_L = []
        self.Mann_alphaepsilon = []
        self.Mann_gamma = []
        self.n_0 = []
        self.n_1 = []
        self.n_2 = []
        self.delta_0 = []
        self.delta_1 = []
        self.delta_2 = []
        self.high_frequency_compensation = []
    
    def simulate(self):
        sys('wine ./wind/mann_turb_x64.exe %s %0.6f %0.6f %0.6f %i %i %i %i %0.6f %0.6f %0.6f %i' %(self.prefix, self.Mann_alphaepsilon, self.Mann_L, self.Mann_gamma, self.seed, self.n_0, self.n_1, self.n_2, self.delta_0, self.delta_1, self.delta_2, self.high_frequency_compensation))
    
    def read_bin(self):
        '''
        

        Returns
        -------
        u_0 : TYPE
            DESCRIPTION.
        u_1 : TYPE
            DESCRIPTION.
        u_2 : TYPE
            DESCRIPTION.

        '''
        # Binaray files names
        file_0 = self.prefix + '_u' + '.bin'
        file_1 = self.prefix + '_v' + '.bin'
        file_2 = self.prefix + '_w' + '.bin'

        # read windsimu raw files                
        with open(file_0, 'rb') as f:
            u0_raw = np.fromfile(f, np.float32)
        with open(file_1, 'rb') as f:
            u1_raw = np.fromfile(f, np.float32)
        with open(file_2, 'rb') as f:
            u2_raw = np.fromfile(f, np.float32)
        
        u_0 = np.zeros((self.n_0, self.n_1, self.n_2), dtype=float)
        u_1 = np.zeros((self.n_0, self.n_1, self.n_2), dtype=float)
        u_2 = np.zeros((self.n_0, self.n_1, self.n_2), dtype=float)
        
        itael = 0
        for i in range(self.n_0):
            for j in range(self.n_1):
                for k in range(self.n_2):
                    u_0[i, j, k] = u0_raw[itael]
                    u_1[i, j, k] = u1_raw[itael]
                    u_2[i, j, k] = u2_raw[itael]
                    itael += 1
        return u_0, u_1, u_2
    
    pass

#%%

class WindBox(object):
    '''
    Instance variables
    ------------------
        u_0[:, :, :] : ndarray, dtype = float
                       wind velocity flutuation in the x direction at the
                       [x, y, x] position in [m/s]
        u_1[:, :, :] : ndarray, dtype = float
                       wind velocity flutuation in the y direction at the
                       [x, y, x] position in [m/s]
        u_2[:, :, :] : ndarray, dtype = float
                       wind velocity flutuation in the z direction at the
                       [x, y, x] position in [m/s]
        u_mean : float
                 mean wind speed in the x direction in [m/s]
        x_0[:] : ndarray, dtype = float
                 windbox points coordinates along the x direction in [m]
        x_1[:] : ndarray, dtype = float
                 windbox points coordinates along the y direction in [m]
        x_2[:] : ndarray, dtype = float
                 windbox points coordinates along the z direction in [m]
        rho : float
              air density in [kg/m**3]
        windsimu : WindSimu class instance
    '''
    
    def __init__(self):
        self.turbulence_generator = []
        self.ws = WindSimu()
        self.mann = Mann_Turb_Box_64()
        #
        self.l_0 = []
        self.l_1 = []
        self.l_2 = []
        #
        self.n_0 = []
        self.n_1 = []
        self.n_2 = []
        #
        self.u_0 = []
        self.u_1 = []
        self.u_2 = []
        self.x_0 = []
        self.x_1 = []
        self.x_2 = []
        self.u0_f = []
        self.u1_f = []
        self.u2_f = []
        #
        self.rho = []
        self.u_mean = []
        self.shear_format = []
        self.z_r = []
        self.z_0 = 0.02
        self.alpha = []
        #
        self.d_0 = []
        #
        self.t_ramp = []
    
    def get(self):
        '''
        '''
        
        l_hub = self.z_r
        
        if (self.turbulence_generator == 'windsimu_x32'):
            self.u_0, self.u_1, self.u_2 = self.ws.read_bin()
        elif (self.turbulence_generator == 'mann_turb_x64'):
            self.u_0, self.u_1, self.u_2 = self.mann.read_bin()
        
        self.x_0 = np.linspace(-self.d_0/2., self.l_0-self.d_0/2., self.n_0)
        self.x_1 = np.linspace(-self.l_1/2., self.l_1/2., self.n_1)
        self.x_2 = np.linspace(l_hub-self.l_2/2., l_hub+self.l_2/2., self.n_2)
        
        self.u0_f = RegularGridInterpolator((self.x_0, self.x_1, self.x_2), self.u_0)
        self.u1_f = RegularGridInterpolator((self.x_0, self.x_1, self.x_2), self.u_1)
        self.u2_f = RegularGridInterpolator((self.x_0, self.x_1, self.x_2), self.u_2)
        
    def export_hdf5(self, hdf5_file):
        '''
        '''

        with h5py.File(hdf5_file, 'w') as f:
            f.create_dataset('turbulence_generator', data=self.turbulence_generator)
            #
            f.create_dataset('n_0', data=self.n_0)
            f.create_dataset('n_1', data=self.n_1)
            f.create_dataset('n_2', data=self.n_2)
            f.create_dataset('l_0', data=self.l_0)
            f.create_dataset('l_1', data=self.l_1)
            f.create_dataset('l_2', data=self.l_2)
            #
            f.create_dataset('u_0', data=self.u_0)
            f.create_dataset('u_1', data=self.u_1)
            f.create_dataset('u_2', data=self.u_2)
            f.create_dataset('x_0', data=self.x_0)
            f.create_dataset('x_1', data=self.x_1)
            f.create_dataset('x_2', data=self.x_2)
            #
            if (self.turbulence_generator == 'mann_turb_x64'):
                f.create_dataset('Mann_L', data=self.mann.Mann_L)
                f.create_dataset('Mann_alphaepsilon', data=self.mann.Mann_alphaepsilon)
                f.create_dataset('Mann_gamma', data=self.mann.Mann_gamma)
                f.create_dataset('high_frequency_compensation', data=self.mann.high_frequency_compensation)
                f.create_dataset('seed', data=self.mann.seed)
                #
            elif (self.turbulence_generator == 'windsimu_x32'):
                f.create_dataset('n_dim', data=self.ws.n_dim)
                f.create_dataset('n_vel', data=self.ws.n_vel)
                f.create_dataset('terrain', data=self.ws.terrain)
                f.create_dataset('Mann_alphaepsilon', data=self.ws.Mann_alphaepsilon)
                f.create_dataset('Mann_L', data=self.ws.Mann_L)
                f.create_dataset('Mann_gamma', data=self.ws.Mann_gamma)
                f.create_dataset('seed', data=self.ws.seed)
    
    def import_hdf5(self, hdf5_file):
        '''
        '''
        #
        with h5py.File(hdf5_file, 'r') as f:
            self.u_0 = f['u_0'][:]
            self.u_1 = f['u_1'][:]
            self.u_2 = f['u_2'][:]
            self.x_0 = f['x_0'][:]
            self.x_1 = f['x_1'][:]
            self.x_2 = f['x_2'][:]

        self.u0_f = RegularGridInterpolator((self.x_0, self.x_1, self.x_2), self.u_0)
        self.u1_f = RegularGridInterpolator((self.x_0, self.x_1, self.x_2), self.u_1)
        self.u2_f = RegularGridInterpolator((self.x_0, self.x_1, self.x_2), self.u_2)
    
    def func_wind_turb(self, r_i, t):
        '''
        '''
        
        # Transfomation tensor from the inertial to the wind-box reference of frame
        A_iw = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        
        r_w = A_iw @ r_i + np.array([self.u_mean*t, 0., 0.])
        
        z = (A_iw @ r_i)[2]
        if (self.shear_format == 'power_law'):
            u_x = self.u_mean * (z/self.z_r)**self.alpha
        elif (self.shear_format == 'logarithmic'):
            u_x = self.u_mean * (np.log(z/self.z_0)/np.log(self.z_r/self.z_0))
        elif (self.shear_format == 'constant'):
            u_x = self.u_mean
        
        u_w = np.zeros((3,))
        u_w[0] = self.u0_f(r_w) - u_x
        u_w[1] = self.u1_f(r_w)
        u_w[2] = self.u2_f(r_w)
        
        u_i =  A_iw.T @ u_w
        
        if (t < self.t_ramp):
            u_i = ((0.99/self.t_ramp)*t + 0.01) * u_i
        
        return u_i
    
    def func_wind_no_turb(self, r_i, t):
        '''
        '''        
        # Transfomation tensor from the inertial to the wind-box reference of frame        
        A_iw = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        
        z = (A_iw @ r_i)[2]
        if (self.shear_format == 'power_law'):
            u_x = self.u_mean * (z/self.z_r)**self.alpha
        elif (self.shear_format == 'logarithmic'):
            u_x = self.u_mean * (np.log(z/self.z_0)/np.log(self.z_r/self.z_0))
        elif (self.shear_format == 'constant'):
            u_x = self.u_mean
        
        u_w = np.zeros((3,))
        u_w[0] = 0. - u_x
        u_w[1] = 0.
        u_w[2] = 0.
        
        u_i =  A_iw.T @ u_w
        
        if (t < self.t_ramp):
            u_i = ((0.99/self.t_ramp)*t + 0.01) * u_i
        
        return u_i
    
    @classmethod
    def construct(cls, wt, wt_file, t_max, delta_t, u_mean, rho=1.225, turbulence_generator=None, shear_format='constant', alpha=0.2, t_ramp=0., seed=None):
        '''
        '''
        
        # Create the object
        obj = cls()
        obj.rho = rho
        obj.u_mean = u_mean
        obj.shear_format = shear_format
        obj.z_r = wt.h_t + wt.s_l * np.sin(wt.tilt)
        obj.alpha = alpha
        obj.t_ramp = t_ramp

        if (turbulence_generator == 'mann_turb_x64'):
            obj.turbulence_generator = turbulence_generator
        elif (turbulence_generator == 'windsimu_x32'):
            obj.turbulence_generator = turbulence_generator
        else:
            obj.turbulence_generator = None
            print('Turbulence will be ignored')
        
        # If turbulent, create the turbulence
        if (obj.turbulence_generator is not None):
            #
            # Wind function is the turbulent wind function
            obj.func_wind = obj.func_wind_turb
            
            # Blade length
            l_blabe = cf.curve_length(np.concatenate((np.array([[0., 0., 0.]]), wt.r_b), axis=0))
            
            # Smallest circle which contatins the wind turbine
            L = np.sqrt(l_blabe**2 + wt.s_l**2)
            obj.d_0 = 2*L
            
            # Horizontal and vertical distances necessary
            obj.l_1 = obj.d_0*1.1
            obj.l_2 = obj.d_0*1.1

            # Longitudinal distance necessary
            n_0 = (obj.u_mean * t_max + obj.d_0)/(obj.u_mean * delta_t)
            
            # obj.n_0 must be a power of 2
            i=0
            while (n_0>2**i):
                i += 1
            obj.n_0 = 2**i
            obj.l_0 = obj.u_mean * delta_t * obj.n_0
        
        else:
            #
            # Wind function is the no turbulent one
            obj.func_wind = obj.func_wind_no_turb

        if (obj.turbulence_generator == 'mann_turb_x64'):
            
            # Read the wt_file
            doc = xlrd.open_workbook(wt_file, on_demand=True).sheet_by_name('mann_turb_x64')
            keys = doc.col_values(0)
            values = doc.col_values(1)
            mann_turb_dict = dict(zip(keys, values))
            del doc, keys, values

            obj.mann.Mann_L = np.float(mann_turb_dict['Mann length scale'])
            obj.mann.Mann_alphaepsilon = np.float(mann_turb_dict['Mann (alpha * epsilon)**(2/3)'])
            obj.mann.Mann_gamma = np.float(mann_turb_dict['Mann gamma'])
            obj.mann.n_1 = np.int(mann_turb_dict['Number of grid points in horizontal direction'])
            obj.mann.n_2 = np.int(mann_turb_dict['Number of gridpoints in vertical direction'])
            obj.mann.high_frequency_compensation = np.int(mann_turb_dict['High frequency compensation'])
            if (seed is None):
                obj.mann.seed = np.int(mann_turb_dict['Seed'])
            else:
                obj.mann.seed = seed

            obj.mann.prefix = np.str(mann_turb_dict['Prefix']) + '_s' + str(np.abs(obj.mann.seed)) + '_'
            
            obj.mann.n_0 = obj.n_0
            obj.mann.l_0 = obj.l_0
            obj.mann.l_1 = obj.l_1
            obj.mann.l_2 = obj.l_2
            obj.mann.delta_0 = obj.mann.l_0 / obj.mann.n_0
            obj.mann.delta_1 = obj.mann.l_1 / obj.mann.n_1
            obj.mann.delta_2 = obj.mann.l_2 / obj.mann.n_2
            
            obj.n_1 = obj.mann.n_1
            obj.n_2 = obj.mann.n_2
            
        elif (turbulence_generator == 'windsimu_x32'):

            # Read the wt_file
            doc = xlrd.open_workbook(wt_file, on_demand=True).sheet_by_name('windsimu_x32')
            keys = doc.col_values(0)
            values = doc.col_values(1)
            wind_box_dict = dict(zip(keys, values))
            del doc, keys, values

            # Define the windsimu attributes
            obj.ws.n_dim = np.int(wind_box_dict['Number of spatial dimensions'])
            obj.ws.n_vel = np.int(wind_box_dict['Number of velocity components to be simulated'])
            obj.ws.n_1 = np.int(wind_box_dict['Number of grid points in horizontal direction'])
            obj.ws.n_2 = np.int(wind_box_dict['Number of gridpoints in vertical direction'])
            obj.ws.terrain = np.str(wind_box_dict['Turbulence description'])
            obj.ws.Mann_alphaepsilon = np.float(wind_box_dict['Mann (alpha * epsilon)**(2/3)'])
            obj.ws.Mann_L = np.float(wind_box_dict['Mann length scale'])
            obj.ws.Mann_gamma = np.float(wind_box_dict['Mann gamma'])
            if (seed is None):
                obj.ws.seed = np.int(wind_box_dict['Seed'])
            else:
                obj.ws.seed = seed

            obj.ws.input_file = np.str(wind_box_dict['Input file']) + '_s' + str(np.abs(obj.ws.seed)) + '_' + '.inp'

            #
            obj.ws.n_0 = obj.n_0
            obj.ws.l_0 = obj.l_0
            obj.ws.l_1 = obj.l_1
            obj.ws.l_2 = obj.l_2
            #
            obj.n_1 = obj.ws.n_1
            obj.n_2 = obj.ws.n_2
        
        
        
        if (turbulence_generator == 'windsimu_x32'):
            
            hdf5_file = obj.ws.input_file[:-4] + '.hdf5'

            condition = True
            if (os.path.exists(hdf5_file)):
                with h5py.File(hdf5_file, 'r') as f:
                    if (f['turbulence_generator'][()] != obj.turbulence_generator): condition = False
                    if (f['n_0'][()] != obj.n_0): condition = False
                    if (f['n_1'][()] != obj.n_1): condition = False
                    if (f['n_2'][()] != obj.n_2): condition = False
                    if (f['l_0'][()] != obj.l_0): condition = False
                    if (f['l_1'][()] != obj.l_1): condition = False
                    if (f['l_2'][()] != obj.l_2): condition = False
                    if (f['n_dim'][()] != obj.ws.n_dim): condition = False
                    if (f['n_vel'][()] != obj.ws.n_vel): condition = False
                    if (f['terrain'][()] != obj.ws.terrain): condition = False
                    if (f['Mann_alphaepsilon'][()] != obj.ws.Mann_alphaepsilon): condition = False
                    if (f['Mann_L'][()] != obj.ws.Mann_L): condition = False
                    if (f['Mann_gamma'][()] != obj.ws.Mann_gamma): condition = False
                    if (f['seed'][()] != obj.ws.seed): condition = False
            else:
                condition = False

            print(condition)
            if (condition):
                print('Importing wimdsimu box from hdf5 file')
                obj.import_hdf5(hdf5_file=hdf5_file)
            else:
                print('Write windsimu input file')
                obj.ws.write_input()
                print('Simulating windsimu box')
                obj.ws.simulate()
                print('Reading windsimu binary files')
                obj.get()
                print('Exporting windbox to hdf5 file')
                obj.export_hdf5(hdf5_file=hdf5_file)
            
        elif (obj.turbulence_generator == 'mann_turb_x64'):

            hdf5_file = obj.mann.prefix + '.hdf5'

            condition = True
            if (os.path.exists(hdf5_file)):
                with h5py.File(hdf5_file, 'r') as f:
                    if (f['turbulence_generator'][()] != obj.turbulence_generator): condition = False
                    if (f['n_0'][()] != obj.n_0): condition = False
                    if (f['n_1'][()] != obj.n_1): condition = False
                    if (f['n_2'][()] != obj.n_2): condition = False
                    if (f['l_0'][()] != obj.l_0): condition = False
                    if (f['l_1'][()] != obj.l_1): condition = False
                    if (f['l_2'][()] != obj.l_2): condition = False
                    if (f['Mann_L'][()] != obj.mann.Mann_L): condition = False
                    if (f['Mann_alphaepsilon'][()] != obj.mann.Mann_alphaepsilon): condition = False
                    if (f['Mann_gamma'][()] != obj.mann.Mann_gamma): condition = False
                    if (f['high_frequency_compensation'][()] != obj.mann.high_frequency_compensation): condition = False
                    if (f['seed'][()] != obj.mann.seed): condition = False
            else:
                condition = False
            
            print(condition)
            if (condition):
                print('Importing mann_turb_x64 box from hdf5 file')
                obj.import_hdf5(hdf5_file=hdf5_file)
            else:
                print('Simulating mann_turb_x64 box')
                obj.mann.simulate()
                print('Reading mann_turb_x64 binary files')
                obj.get()
                print('Exporting windbox to hdf5 file')
                obj.export_hdf5(hdf5_file=hdf5_file)
        
        return obj