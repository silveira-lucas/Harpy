import numpy as np

import custom_functions.custom_functions as cf

#%%

class DynamicStall(object):
    '''
    This class implements the methods necessary to calculate the blade section
    lift coeficient taking into account the dynamic_stall. It follows the
    method proposed by Stig Øye [Øye S., Dynamic Stall simulated as time lag of
    separeation, Technical University of Denmark]

    Indices:    
    i_t : time iteration counter
    i_b : blade number counter
    i_z : blde section counter
    
    Attributes
    ----------
    fs : numpy.ndarray[:, :, :], dtype=float
         Stig Øye dynamic stall model degree of attached flow.
         indices: [i_t, i_b, i_z]
    '''
    
    def __init__(self, wt, t_vec):
        '''
        Default class constructor. Initialises the attribute fs according to
        the WindTurbine objec wt and time array t_vec.

        Parameters
        ----------
        wt : WindTurbine object.
        t_vec : numpy.ndarray[:], dtype=float
                [s] simulation time array.
        '''
        
        self.fs = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
    
    def dynamic_stall(self, wt, alpha, urel, i_t, i_b, i_z, delta_t):
        '''
        This method calculates the blade section lift coeficient taking into
        account the dynamic_stall. It follows the method proposed by Stig Øye.
        [Øye S., Dynamic Stall simulated as time lag of separeation, Technical
        University of Denmark]

        Parameters
        ----------
        wt : WindTurbine object.
        alpha : float
                [rad] blade section angle of attack.
        urel : float.
               [m/s] wind speed relative to the blade section.
        i_t : int
              [-] time iteration counter.
        i_b : int
              [-] blade number couter.
        i_z : TYPE
              [-] blade section number counter.
        delta_t : float
                  [s] time difference between t[i_t] - t[i_t-1].

        Returns
        -------
        Cl : float
             [-] blade section airfoil lift coeficient.
        '''
        
        # Static or equilibrium component of fs
        fs_st = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 4, i_z])
        
        # Inviscid lift coefficient
        Cl_inv = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 5, i_z])
        
        # Fully separated lift coeffcient
        Cl_fs  = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 6, i_z])
        
        # Time coefficient
        tau = 4. * wt.chord[i_z] / urel
        
        # Numerical integration to obtain the instantaneous value of fs
        self.fs[i_t, i_b, i_z] = fs_st + (self.fs[i_t-1, i_b, i_z]-fs_st) * np.exp(-delta_t/tau)
        
        # Instantaneous value of the lift coefficient
        Cl = self.fs[i_t, i_b, i_z] * Cl_inv + (1-self.fs[i_t, i_b, i_z])*Cl_fs
        
        return Cl

#%%

class DynamicInflow(object):
    '''
    This class implements the methods necessary to calcualte the dynamic inflow
    (not corrected by the skew angle).
    
    Attributes
    ----------
    w_0 : numpy.ndarray[:, :, :], dtype=float
          Induced velocity on the rotor plane reference frame.
    w_qs : numpy.ndarray[:, :, :], dtype=float
           Quasi-steady induced wind velocity.
    w_int : numpy.ndarray[:, :, :], dtype=float
            Intermediary value of the induced velocity.
    '''
    
    def __init__(self, wt, t_vec):
        '''
        Default class constructor. Initialises the attributes according to the
        WindTurbine objec wt and time array t_vec.

        Parameters
        ----------
        wt : WindTurbine object.
        t_vec : numpy.ndarray[:], dtype=float
                [s] simulation time array.
        '''
        
        self.w_0 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.w_qs = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.w_int = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )

    def dynamic_inflow(self, wt, wind, f_3, a, uprime, f_P, z_3, Z_3, u_3, i_t, i_b, i_z, delta_t):
        '''
        This method calcualtes the dynamic inflow (not corrected by the skew
        angle) following the method proposed by Stig Øye [Hansen O. L. Hansen
        Aerodynamics of wind turbines].

        Parameters
        ----------
        wt :  WindTurbine object
        wind : WindBox object
            DESCRIPTION.
        f_3 : numpy.ndarray[:], dtype=float
              [-] Aerodynamic forces per unit of lenght at the blade section on
              rotor plane reference frame.
        a : float
            [-] Axial induction factor
        uprime : float
                 [m/s] |u + f_g n(n⋅w)|
        f_P : float
              Prandtl's tip loss corection factor.
        z_3 : float
              Blade section z coordinate projected on the rotor plane.
        Z_3 : float
              Blade tip z coordinate projected on the rotor plane.
        u_3 : numpy.ndarray[:], dtyp=float
              [m/s] Wind velocity at the blade section location described in
              the rotor plane reference frame.
        i_t : int
              Time iteration counter
        i_b : int
              Blade number counter
        i_z : int
              Blade section counter
        delta_t : float
                  Time difference between the next time iteration and the
                  current one, i.e. self.t_vec[i_t+1]-t
        '''
        
        # Quasi-steady induced wind velocity at the tangential and normal
        # directions respectively. The outward induced velocity is zero.
        self.w_qs[i_t+1, i_b, i_z, 0] = - (wt.NB*f_3[0])/(4.*np.pi*wind.rho*z_3*f_P*uprime)
        self.w_qs[i_t+1, i_b, i_z, 1] = - (wt.NB*f_3[1])/(4.*np.pi*wind.rho*z_3*f_P*uprime)
        self.w_qs[i_t+1, i_b, i_z, 2] = 0.
        
        # Correction if the axial induction factor is too high
        if (a>=0.5):
            a = 0.5
        
        # Time constants
        tau_1 = 1.1*Z_3 / ((1.-1.3*a)*np.linalg.norm(u_3))
        tau_2 = (0.39-0.26*(z_3/Z_3)**2)*tau_1
        
        # Numerical integration to obtain w_0
        H = self.w_qs[i_t+1, i_b, i_z, :] + 0.6*tau_1*(self.w_qs[i_t+1, i_b, i_z, :] - self.w_qs[i_t, i_b, i_z, :])/delta_t
        self.w_int[i_t+1, i_b, i_z, :] = H + (self.w_int[i_t, i_b, i_z, :]-H)*np.exp(-delta_t/tau_1)
        self.w_0[i_t+1, i_b, i_z, :] = self.w_int[i_t+1, i_b, i_z, :] + (self.w_0[i_t, i_b, i_z, :]-self.w_int[i_t+1, i_b, i_z, :])*np.exp(-delta_t/tau_2)

#%%

class UnsteadyBem(object):
    '''
    In this code, the Unsteady Blade Element Momentum is yet another object.
    The choice of separating the wind turbine and the BEM model was made in
    order to maintain in the wind turbine wt object only the attributes that
    inherently characterise it like the blade radius, pitch angle, etc.
    
    Indices:
    i_t : time iteration counter
    i_b : blade number counter
    i_z : blde section counter
    i_d : direction {0: x, 1: y, 2: z}
    i_df : degree of freedom
    
    Attributes
    ----------
    
    ds : DynamicStall object
    dw : DynamicInflow object
    t_vec : numpy.ndarray[:], dtype=float
            [s] simulation time array
    phi : numpy.ndarray[:, :, :], dtype=float
          [rad] blade section inflow angle
          indices: [i_t, i_b, i_z]
    alpha : numpy.ndarray[:, :, :], dtype=float
            [rad] blade section angle of attack
            indices: [i_t, i_b, i_z]
    f_aero : numpy.ndarray[:, :, :, :], dtype=float
             [N/m] blade section aerodynamic forces per meter (in the blade 
             root reference frame)
             indices: [i_t, i_b, i_z, i_d]
    r_b4 : numpy.ndarray[:, :, :, :], dtype=float
           [m] blade section position in the moving blade root reference frame
           indices: [i_t, i_b, i_z, i_d]
    v_4 : numpy.ndarray[:, :, :, :], dtype=float
          [m/s] blade section absolute velocity in the moving blade root
          reference frame
          indices: [i_t, i_b, i_z, i_d]
    w : numpy.ndarray[:, :, :, :], dtype=float
        [m/s] iduced velocity in the rotor shaft reference of frame
        indices: [i_t, i_b, i_z, i_d]
    a : numpy.ndarray[:, :], dtype=float
        [-] axial induction factor
        indices: [i_b, i_z]
    chi : numpy.ndarray[:], dtype=float
          [rad] skew angle
          indices : [i_t]
    u_prime_2 : numpy.ndarray[:, :, :], dtype=float
                [m/s] (u + f_g n(n⋅w)), described in the nacelle reference
                frame and without the wind turbulent fluctuations components
                indices: [i_b, i_z, i_d]    
    theta : numpy.ndarray[:], dtype=float
            [rad] shaft azimuth angle as a funtion of time
    Omega : numpy.ndarray[:], dtype=float
            [rad/s] shaft angular speed as a function of time
    q : numpy.ndarray[:, :], dtype=float
        generalised degrees of freedom
        indices: [i_t, i_df]
    q_dot : numpy.ndarray[:, :], dtype=float
            generalised degrees of freedom first derivative on time
            indices: [i_t, i_df]
    q_ddot : numpy.ndarray[:, :], dtype=float
             generalised degrees of freedom second derivative on time
             indices: [i_t, i_df]             
    '''
    
    def __init__(self, wt, t_vec):
        '''
        This method allocates memory for the object attributes and initialises
        theta and Omega.

        Parameters
        ----------
        wt : WindTurbine object
        t_vec : numpy.ndarray[:], dtype=float
                simulation time array
        '''
        
        self.ds = DynamicStall(wt, t_vec)
        self.dw = DynamicInflow(wt, t_vec)
        
        self.t_vec = t_vec
        self.phi = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        self.alpha = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        self.r_b4 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.w = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.a = np.zeros( (wt.NB, len(wt.z)) )
        self.chi = np.zeros( (len(t_vec),) )
        self.u_prime_2 = np.zeros( (wt.NB, len(wt.z), 3) )
        self.fa = np.zeros( (wt.NB, len(wt.z), 3) )
        self.theta = np.zeros( t_vec.shape )
        self.Omega = np.zeros( t_vec.shape )
        self.q = np.zeros( (len(t_vec), len(wt.q)) )
        self.q_dot = np.zeros( (len(t_vec), len(wt.q)) )
        self.q_ddot = np.zeros( (len(t_vec), len(wt.q)) )
        
        self.l = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        self.d = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        self.f_aero = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        
        # initial conditions
        self.theta[0] = wt.theta
        self.Omega[0] = wt.Omega
    
    def skew_angle(self, wt):
        '''
        This method calculates the average skew angle at r/R = 0.7

        Parameters
        ----------
        wt : WindTurbine object.

        Returns
        -------
        chi : float
              [rad] Skew angle.
        '''
        
        u_07 = np.zeros((wt.NB, 3))
        u_a = np.zeros((3,))
        
        # Induced velocity at r/R=0.7 at each blade
        for i_b in range(wt.NB):
            for i_d in range(3):
                u_07[i_b, i_d] = np.interp(0.7, wt.z/wt.R, self.u_prime_2[i_b, :, i_d])
        
        # Averaged induced velocity at r/R=0.7
        u_a = np.sum(u_07[:, :], axis=0)/wt.NB
        uan = np.linalg.norm(u_a)
        
        # Skew angle, chi = arccos(n.u'/|u'|)
        chi = np.arccos(u_a[1]/uan)
        return chi
    
    def unsteady_bem(self, wt, wind, t, i_t, q, q_dot, dynamic_stall_on=True, dynamic_inflow_on=True):

        # Print time        
        if (t==self.t_vec[i_t]):
            print('i_t: %i, time: %0.3f, chi: %0.3f'%(i_t, t, self.chi[i_t]*(180./np.pi)) )
        
        delta_t = t - self.t_vec[i_t]
        
        wt.q = q
        wt.q_dot = q_dot
        wt.Omega = self.Omega[i_t]

        # If the simulation is between time-steps due to the (Runge-Kutta)
        # integrator, the azimuth angle needs to account for this half
        # time-step position
        wt.theta = self.theta[i_t] + wt.Omega*delta_t
        
        A_y0 = cf.rotate_tensor(wt.yaw, 'z')
        
        for i_b in range(wt.NB):
            
            # Initial azimuth positon of the blade i_b
            wt.eta = np.pi + (2. * i_b * np.pi)/wt.NB

            # Reference matrices
            wt.reference_matrices()
            #
            wt.A_02 = (wt.A_01.T @ wt.A_12.T).T
            wt.A_03 = (wt.A_01.T @ wt.A_12.T @ wt.A_23.T ).T
            wt.A_04 = (wt.A_01.T @ wt.A_12.T @ wt.A_23.T @ wt.A_34.T ).T
            wt.A_24 = (wt.A_23.T @ wt.A_34.T).T
            
            # blade deflection (its reference of frame)
            ur = np.zeros( (len(wt.z), 3) )
            ur[:, 0] = q[(wt.n_m*i_b)+0:(wt.n_m*i_b)+wt.n_m] @ wt.phi_x[:, :]
            ur[:, 1] = q[(wt.n_m*i_b)+0:(wt.n_m*i_b)+wt.n_m] @ wt.phi_y[:, :]
            ur[:, 2] = 0.

            # blade deflection velocity (its reference of frame)
            ur_dot = np.zeros( (len(wt.z), 3) )
            ur_dot[:, 0] = q_dot[(wt.n_m*i_b)+0:(wt.n_m*i_b)+wt.n_m] @ wt.phi_x[:, :]
            ur_dot[:, 1] = q_dot[(wt.n_m*i_b)+0:(wt.n_m*i_b)+wt.n_m] @ wt.phi_y[:, :]
            ur_dot[:, 2] = 0.
            
            # Position vectors
            wt.position_vectors()
            
            # Tower position vector
            r_t0 = wt.r_t0
            
            # Shaft position vector
            r_s3 = wt.r_s3
            
            # Blade positon vector
            self.r_b4[i_t, i_b, :, :] = wt.r_b + ur
            
            # max z coordinate on the undeflected rotor plane
            Z_3 = (wt.A_34.T @ wt.r_b[-1, :])[2]
            
            for i_z in reversed(range(len(wt.z)-1)):
    
                # Total position vector
                r_0 = r_t0 + (wt.A_01.T @ wt.A_12.T @ wt.A_23.T) @ (r_s3 + wt.A_34.T @ self.r_b4[i_t, i_b, i_z, :])
                
                # Tower-top velocity
                v_t0 = wt.v_t0
                
                # Total velocity
                v_4 = (wt.A_04 @ v_t0) + wt.A_34 @ (wt.Omega_01_3 + wt.Omega_23_3) @ r_s3 + (wt.Omega_01_4 + wt.Omega_23_4) @ self.r_b4[i_t, i_b, i_z, :] + ur_dot[i_z, :]
                            
                # Wind velocity
                u_0 = A_y0 @ wind.func_wind(A_y0.T @ r_0, t)
                u_0_no_turb = A_y0 @ wind.func_wind_no_turb(A_y0.T @ r_0, t)
                u_2 = wt.A_02 @ u_0_no_turb
                u_3 = wt.A_03 @ u_0
                
                # Wind velocity relative to the blade
                u_rel = wt.A_04 @ u_0 + wt.A_34 @ self.w[i_t, i_b, i_z, :] - v_4
                urel = np.linalg.norm(u_rel[:2])
                
                # Inflow angle
                phi = np.arctan2(u_rel[1], -u_rel[0]) # [rad]
                alpha = phi - (wt.twist[i_z] - wt.pitch) # [rad]
                
                if (dynamic_stall_on and i_t!=0):
                    Cl = self.ds.dynamic_stall(wt, alpha, urel, i_t, i_b, i_z, t-self.t_vec[i_t-1])
                else:
                    Cl = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 1, i_z])
                
                # Airfoil drag coefficient
                Cd = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 2, i_z])
                
                # Lift and drag coefficient per unit of length
                l = 1./2. * wind.rho * urel**2 * wt.chord[i_z] * Cl # [N/m]
                d = 1./2. * wind.rho * urel**2 * wt.chord[i_z] * Cd # [N/m]
                
                # Aerodynamic forces on the blade frame
                self.fa[i_b, i_z, 0] = l*np.sin(phi) - d*np.cos(phi)
                self.fa[i_b, i_z, 1] = l*np.cos(phi) + d*np.sin(phi)
                self.fa[i_b, i_z, 2] = 0.
                
                # Saving values for post-processing
                if (delta_t==0):
                    self.l[i_t, i_b, i_z] = l
                    self.d[i_t, i_b, i_z] = d
                    self.phi[i_t, i_b, i_z] = phi
                    self.alpha[i_t, i_b, i_z] = alpha
                    self.f_aero[i_t, i_b, i_z, :] = self.fa[i_b, i_z, :]
                
                # Prandtl's tip loss correction
                z_3 = (wt.A_34.T @ wt.r_b[i_z, :])[2] # z coordinate on the undeflected rotor plane
                if (wind.u_mean!=0):
                    f = (wt.NB/2.) * (Z_3-z_3)/(z_3*np.abs(np.sin(phi)))
                    f_P = 2./np.pi * np.arccos(np.exp(-f))
                
                # Axial induction factor
                if (wind.u_mean!=0):
                    self.a[i_b, i_z] = np.abs(self.w[i_t, i_b, i_z, 1]/wind.u_mean)
                else:
                    self.a[i_b, i_z] = 0.
                
                # Glauert correction factor
                if (self.a[i_b, i_z]<=1./3.):
                    F_G = 1.
                else:
                    F_G = (1./4.)*(5.-3.*self.a[i_b, i_z])
                
                # u_prime
                u_prime_3 = u_3.copy() + np.array([0., F_G * self.w[i_t, i_b, i_z, 1], 0.])
                uprime = np.linalg.norm(u_prime_3)
                
                f_3 = wt.A_34.T @ self.fa[i_b, i_z, :]
                if (t!=self.t_vec[-1] and t==self.t_vec[i_t] and wind.u_mean!=0):
                    if (dynamic_inflow_on):
                        self.dw.dynamic_inflow(wt, wind, f_3, self.a[i_b, i_z], uprime, f_P, z_3, Z_3, u_3, i_t, i_b, i_z, self.t_vec[i_t+1]-t)
                    else:
                        self.dw.w_0[i_t+1, i_b, i_z, 0] = - (wt.NB*f_3[0])/(4.*np.pi*wind.rho*z_3*f_P*uprime)
                        self.dw.w_0[i_t+1, i_b, i_z, 1] = - (wt.NB*f_3[1])/(4.*np.pi*wind.rho*z_3*f_P*uprime)
                        self.dw.w_0[i_t+1, i_b, i_z, 2] = 0.
                    
                    # Yaw/tilt correction factor
                    YT_cf = 1. + (z_3/Z_3*np.tan(self.chi[i_t]/2.)*np.cos(self.theta[i_t] + wt.eta - wt.downwind))
                    
                    # Induced velocity
                    self.w[i_t+1, i_b, i_z, :] = YT_cf * self.dw.w_0[i_t+1, i_b, i_z, :]

                # To calculate skew angle
                self.u_prime_2[i_b, i_z, :] = u_2.copy() + np.array([0., F_G * self.w[i_t, i_b, i_z, 1], 0.])
                
            # end for i_z
        # end for i_b
        
        # Calculate the skew angle        
        if (t!=self.t_vec[-1] and t==self.t_vec[i_t] and wind.u_mean!=0):
            self.chi[i_t+1] = self.skew_angle(wt)
        
        # Forces on blade 0
        wt.f_0_x = self.fa[0, :, 0]
        wt.f_0_y = self.fa[0, :, 1]
        wt.f_0_z = self.fa[0, :, 2]
        
        # Forces on blade 1
        wt.f_1_x = self.fa[1, :, 0]
        wt.f_1_y = self.fa[1, :, 1]
        wt.f_1_z = self.fa[1, :, 2]
        
        # Forces on blade 2
        wt.f_2_x = self.fa[2, :, 0]
        wt.f_2_y = self.fa[2, :, 1]
        wt.f_2_z = self.fa[2, :, 2]
        
        return