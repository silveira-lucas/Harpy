import numpy as np

#%%

class WindTurbineController(object):
    '''
    This class implements two very very simple pitch and generator torque
    controllers using an proportional integral algorithms.
    
    Atributes
    ---------
    t_vec : numpy.ndarray [:], dtype=float
            [s] simulation time array
    KI : float
         [rad*s/rad] Collective Blade Pitch Integral Gain
    KP : flaot
         [rad/rad] Collective Blade Derivative Gain
    KK_1 : float
           [rad] Collective Blade Pitch Gain schedueling KK1
    KK_2 : float
           [rad**2] Collective Blade Pitch Gain schedueling KK2    
    Omega_ref : float
                [rad/s] Approximately equal to the rated shaft angular speed
    KP_G : flaot
           [N*m/rad] Generator Torque Derivative Gain
    KI_G flaot
         [N*m/rad] Generator Torque Integral Gain
    Omega_filt : numpy.ndarray [:], dtype=float
                 [rad/s] shaft angular velocity signal filtered entering the
                 controller
    pitch : numpy.ndarray [:], dytpe=float
            [rad] Collective Blade Pitch
    pitch_prop : float
                 [rad] Collective Blade Pitch proportional term
    pitch_int : numpy.ndarray [:], dtype=float integral term
                [rad] Collective Blade Pitch
    Qg_int : numpy.ndarray [:], dtype=float
             [N*m] Generator Torque integral term
    Qg : numpy.ndarray [:], dtype=float
         [N*m] Generator Torque proportional term
    '''

    def __init__(self, wt, t_vec):
        '''
        Default initialiser method for the controler. For now the controller
        attributes are hardcoded in this method, this is a bad practice and
        should be changed as soon as possible.
        '''
        
        delta_t = np.diff(t_vec)
        self.t_vec = np.concatenate( (t_vec, [t_vec[-1]+delta_t[-1]]))
        
        # Pitch controller constants
        self.KP = 1.06713 # [rad/rad]
        self.KI = 0.242445 # [rad*s/rad]
        self.KK_1 = 11.4 * (np.pi/180.) # [rad]
        self.KK_2 = 402.9 * (np.pi/180.)**2 # [rad**2]
        self.Omega_rated = (wt.P_rated/wt.K_gen)**(1./3.) # [rad/s]
        self.Omega_ref = self.Omega_rated + 0.01 # [rad/s]
        
        # Torque controller constants
        self.KP_G = 6.83456e7 # [N*m*s/rad]
        self.KI_G = 1.53367e7 # [N*m/rad]
        self.Qg_max = 15.6*10**6 # [N*m]
        
        # Running variables
        self.Omega_filt = np.zeros(self.t_vec.shape)
        self.pitch = np.zeros(self.t_vec.shape)
        self.pitch[0] = wt.pitch
        self.pitch_prop = []
        self.pitch_int = np.zeros(self.t_vec.shape)
        
        self.Qg_int = np.zeros(self.t_vec.shape)
        self.Qg_prop = []
        self.Qg = np.zeros(self.t_vec.shape)

    def pitch_control(self, wt, t, i_t):
        '''
        This method implements a very very very naive proportional + integral
        control for the pitch.
        
        Parameters
        ----------
        wt : WindTurbine object
        t : float
            [s] time from the begning of the simulation
        i_t : int
              time step counter
        '''
        
        # Check if it is the first time step        
        if (i_t == 0):
            delta_t = self.t_vec[1] - self.t_vec[i_t]
        else:
            delta_t = t - self.t_vec[i_t-1]
        
        # Filter the signal entering the controller
        alpha_filt = np.exp(-delta_t*np.pi/2.)
        self.Omega_filt[i_t+1] = (1.-alpha_filt)*wt.Omega + alpha_filt*self.Omega_filt[i_t]
        
        # Second order polynomial curve fitting
        GK = 1./(1. + wt.pitch/self.KK_1 + wt.pitch**2/self.KK_2 )
        
        # Proportional term
        self.pitch_prop = GK * self.KP * (self.Omega_filt[i_t+1] - self.Omega_ref)
        
        # Integral term
        self.pitch_int[i_t+1] = self.pitch_int[i_t] + GK*self.KI*(self.Omega_filt[i_t+1] - self.Omega_ref) * delta_t
        
        # Check if pitch_int is under the pitch limits
        if (self.pitch_int[i_t+1] < wt.pitch_min):
            self.pitch_int[i_t+1] = wt.pitch_min
        elif (self.pitch_int[i_t+1] > wt.pitch_max):
            self.pitch_int[i_t+1] = wt.pitch_max
        
        # Sum of the proportional and integral terms
        self.pitch[i_t + 1] = self.pitch_prop + self.pitch_int[i_t+1]
        
        # Check if the rate of change in pitch is greater than the limit
        if ((self.pitch[i_t+1] - self.pitch[i_t])/delta_t > wt.pitch_dot_max):
            self.pitch[i_t+1] = self.pitch[i_t] + wt.pitch_dot_max * delta_t
        elif ((self.pitch[i_t+1] - self.pitch[i_t])/delta_t < - wt.pitch_dot_max):
            self.pitch[i_t+1] = self.pitch[i_t] - wt.pitch_dot_max * delta_t
        
        # Check if the new pitch is under the limits
        if (self.pitch[i_t+1] < wt.pitch_min):
            self.pitch[i_t+1] = wt.pitch_min
        elif (self.pitch[i_t+1] > wt.pitch_max):
            self.pitch[i_t+1] = wt.pitch_max
        
        return
    
    def generator_control(self, wt, wind, t, i_t):
        '''
        This method controls the generator torque in order to stabalise the
        torque around the set point, defined by mean wind speed, using an
        proportional integral KI algorithm.
        
        Parameters
        ----------
        wt : WindTurbine object
        wind : WindBox object
        t : float
            [s] time since the begnning of the simulation
        i_t : int
              time steps counter
        '''
        
        # Check if it is the first time step        
        if (i_t == 0):
            delta_t = self.t_vec[1] - self.t_vec[i_t]
        else:
            delta_t = t - self.t_vec[i_t-1]

        # Filter the signal entering the controller? (*** Ask Martin)
#        alpha_filt = np.exp(-delta_t*np.pi/2.)
        alpha_filt = 0.
        self.Omega_filt[i_t+1] = (1.-alpha_filt)*wt.Omega + alpha_filt*self.Omega_filt[i_t]
        
        Omega_sp = (wt.lmbda * wind.u_mean)/wt.R
        # Check if Omega_sp is under the limits
        if (Omega_sp > wt.Omega_max):
            Omega_sp = wt.Omega_max
        
        # Proportional term
        self.Qg_prop = self.KP_G * (self.Omega_filt[i_t+1] - Omega_sp)

        # Integral term
        self.Qg_int[i_t+1] = self.Qg_int[i_t] + self.KI_G*(self.Omega_filt[i_t+1] - Omega_sp) * delta_t
        
        # Check if Qg_int is under the limits
        if (self.Qg_int[i_t+1] < 0.):
            self.Qg_int[i_t+1] = 0.
        elif (self.Qg_int[i_t+1] > wt.K_gen*wt.Omega_max**2):
            self.Qg_int[i_t+1] = wt.K_gen*wt.Omega_max**2
        
        # Generator moment
        self.Qg[i_t+1] = wt.generator_moment(Omega=Omega_sp) + self.Qg_prop + self.Qg_int[i_t+1]
        
        # Check if Qg is under the limits
        if (self.Qg[i_t+1] < 0.):
            self.Qg[i_t+1] = 0.
        elif (self.Qg[i_t+1] > wt.K_gen*wt.Omega_max**2):
            self.Qg[i_t+1] = wt.K_gen*wt.Omega_max**2
        
        return self.Qg[i_t+1]