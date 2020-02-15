import numpy as np

import custom_functions.custom_functions as cf

#%%

class UnsteadyBem(object):
    '''
    '''
    
    def __init__(self, wt, t_vec):
        
        self.t_vec = t_vec
        
        self.l = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        self.d = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        
        self.phi = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        self.alpha = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        
        self.r_0 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.r_b4 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.v_4 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        
        self.w = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.w_0 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.w_qs = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.w_int = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        
        self.a = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        
        self.chi = np.zeros( (len(t_vec),) )
        self.YT_cf = np.zeros( (len(t_vec), wt.NB) )
        
        self.u_prime_3 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.u_prime_2 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        
        self.px = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        self.py = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        
        self.f_aero = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.f_gravity = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        self.f_inertia = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        
        self.fs = np.zeros( (len(t_vec), wt.NB, len(wt.z)) )
        
        self.theta = np.zeros( t_vec.shape )
        self.Omega = np.zeros( t_vec.shape )
        self.Power_rot = np.zeros( t_vec.shape )
        
        self.M_aero = np.zeros( (len(t_vec), wt.NB, 3) )
        self.M_blade = np.zeros( (len(t_vec), wt.NB, 3) )
        self.M_gravity = np.zeros( (len(t_vec), wt.NB, 3) )
        self.M_inertia = np.zeros( (len(t_vec), wt.NB, 3) )
        
        self.M_rot = np.zeros( (len(t_vec), 3) )
        
        self.A_01 = np.zeros( (len(t_vec), wt.NB, 3, 3) )
        self.A_23 = np.zeros( (len(t_vec), wt.NB, 3, 3) )
        
        self.q = np.zeros( (len(t_vec), len(wt.q)) )
        self.q_dot = np.zeros( (len(t_vec), len(wt.q)) )
        self.q_ddot = np.zeros( (len(t_vec), len(wt.q)) )
        
        self.probe_t1 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        # self.probe_t2 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        # self.probe_t3 = np.zeros( (len(t_vec), wt.NB, len(wt.z), 3) )
        
        # initial conditions
        self.theta[0] = wt.theta
        self.Omega[0] = wt.Omega
    
    def dynamic_stall(self, wt, alpha, urel, i_t, i_b, i_z, delta_t):
        '''
        This method calculates the blade section lift coeficient taking into
        account the dynamic_stall. It follows the method proposed by Stig Øye.
        []
        
        Parameters
        ----------
        wt : WindTurbine object.
        alpha : float
                [rad] blade section angle of attack
        urel : float
               [m/s] wind speed relative to the blade section.
        i_t : int
              [-] time iteration counter.
        i_b : int
              [-] blade number couter.
        i_z : int
              [-] blade section number counter.
        delta_t : float
                  [s] time difference between t[i_t] - t[i_t-1].

        Returns
        -------
        Cl : float
             [-] blade section airfoil lift coeficient.
        '''
        
        fs_st = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 4, i_z])
        Cl_inv = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 5, i_z])
        Cl_fs  = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 6, i_z])
        tau = 4. * wt.chord[i_z] / urel
        self.fs[i_t, i_b, i_z] = fs_st + (self.fs[i_t-1, i_b, i_z]-fs_st) * np.exp(-delta_t/tau)
        Cl = self.fs[i_t, i_b, i_z] * Cl_inv + (1-self.fs[i_t, i_b, i_z])*Cl_fs
    
        return Cl

    def dynamic_inflow(self, wt, wind, phi, uprime, f_P, z_3, Z_3, u_3, i_t, i_b, i_z, delta_t):
        '''
        '''
    
        self.w_qs[i_t+1, i_b, i_z, 0] = - (wt.NB*self.l[i_t, i_b, i_z]*np.sin(phi))/(4.*np.pi*wind.rho*z_3*f_P*uprime)
        self.w_qs[i_t+1, i_b, i_z, 1] = - (wt.NB*self.l[i_t, i_b, i_z]*np.cos(phi))/(4.*np.pi*wind.rho*z_3*f_P*uprime)
        self.w_qs[i_t+1, i_b, i_z, 2] = 0.
        
        if (self.a[i_t, i_b, i_z]>=0.5):
            self.a[i_t, i_b, i_z] = 0.5
        
        tau_1 = 1.1*Z_3 / ((1.-1.3*self.a[i_t, i_b, i_z])*np.linalg.norm(u_3))
        tau_2 = (0.39-0.26*(z_3/Z_3)**2)*tau_1
        
        H = self.w_qs[i_t+1, i_b, i_z, :] + 0.6*tau_1*(self.w_qs[i_t+1, i_b, i_z, :] - self.w_qs[i_t, i_b, i_z, :])/delta_t
        self.w_int[i_t+1, i_b, i_z, :] = H + (self.w_int[i_t, i_b, i_z, :]-H)*np.exp(-delta_t/tau_1)
        self.w_0[i_t+1, i_b, i_z, :] = self.w_int[i_t+1, i_b, i_z, :] + (self.w_0[i_t, i_b, i_z, :]-self.w_int[i_t+1, i_b, i_z, :])*np.exp(-delta_t/tau_2)

    def unsteady_bem(self, wt, wind, t, i_t, q, q_dot, dynamic_stall_on=True, dynamic_inflow_on=True):
        '''
        '''
        
        if (t==self.t_vec[i_t]):
            # Print time
            print('i_t: %i, time: %0.3f, chi: %0.3f'%(i_t, t, self.chi[i_t]*(180./np.pi)) )
        
        wt.q = q
        wt.q_dot = q_dot
        wt.theta = self.theta[i_t]
        wt.Omega = self.Omega[i_t]
        
        A_y0 = cf.rotate_tensor(wt.yaw, 'z')
        
        for i_b in range(wt.NB):
            
            # Initial azimuth positon of the blade i_b
            wt.eta = np.pi + (2. * i_b * np.pi)/wt.NB
            
            # blade deflection (its reference of frame)
            ur = np.zeros( (len(wt.z), 3) )
            ur[:, 0] = q[3*i_b+0] * wt.phi_0_x[:] + q[3*i_b+1] * wt.phi_1_x[:] + q[3*i_b+2] * wt.phi_2_x[:]
            ur[:, 1] = q[3*i_b+0] * wt.phi_0_y[:] + q[3*i_b+1] * wt.phi_1_y[:] + q[3*i_b+2] * wt.phi_2_y[:]
            ur[:, 2] = 0.
    
            # blade deflection velocity (its reference of frame)
            ur_dot = np.zeros( (len(wt.z), 3) )
            ur_dot[:, 0] = q_dot[3*i_b+0] * wt.phi_0_x[:] + q_dot[3*i_b+1] * wt.phi_1_x[:] + q_dot[3*i_b+2] * wt.phi_2_x[:]
            ur_dot[:, 1] = q_dot[3*i_b+0] * wt.phi_0_y[:] + q_dot[3*i_b+1] * wt.phi_1_y[:] + q_dot[3*i_b+2] * wt.phi_2_y[:]
            ur_dot[:, 2] = 0.
            
            # Reference matrices
            wt.reference_matrices()
            self.A_01[i_t, i_b, :, :] = wt.A_12
            self.A_23[i_t, i_b, :, :] = wt.A_23
            #
            wt.A_02 = (wt.A_01.T @ wt.A_12.T).T
            wt.A_03 = (wt.A_01.T @ wt.A_12.T @ wt.A_23.T ).T
            wt.A_04 = (wt.A_01.T @ wt.A_12.T @ wt.A_23.T @ wt.A_34.T ).T
            wt.A_24 = (wt.A_23.T @ wt.A_34.T).T
            
            # Position vectors
            wt.position_vectors()
            
            # Tower position vector
            # r_t0 = np.array([0., 0., -wt.h_t])
            r_t0 = wt.r_t0
            
            # Shaft position vector
            # r_s3 = np.array([0., -wt.s_l, 0.])
            r_s3 = wt.r_s3
            
            # Blade positon vector
            self.r_b4[i_t, i_b, :, :] = wt.r_b[:, :] + ur[:, :]
            
            # Rotor plane normal vector
            n_0 = np.eye(3) @ wt.A_12.T @ np.array([0., 1., 0.])
            # u_mean = n_0 @ ( A_y0 @ np.array([0., wind.u_mean, 0.]) )
            # u_mean = n_0 @ (A_y0 @ wind.func_wind_no_turb(A_y0.T @ np.array([0., 0., -wt.h_t]), t))
            u_mean = wind.u_mean
            
            for i_z in reversed(range(len(wt.z)-1)):
    
                # Total positio vector
                self.r_0[i_t, i_b, i_z, :] = r_t0 + (wt.A_01.T @ wt.A_12.T @ wt.A_23.T) @ (r_s3 + wt.A_34.T @ self.r_b4[i_t, i_b, i_z, :])
                
                # Tower-top velocity
                v_t0 = wt.v_t0
                
                # Total velocity
                self.v_4[i_t, i_b, i_z, :] = (wt.A_04 @ v_t0) + wt.A_34 @ (wt.Omega_01_3 + wt.Omega_23_3) @ r_s3 + (wt.Omega_01_4 + wt.Omega_23_4) @ self.r_b4[i_t, i_b, i_z, :] + ur_dot[i_z, :]
                            
                # Wind velocity
                u_0 = A_y0 @ wind.func_wind(A_y0.T @ self.r_0[i_t, i_b, i_z, :].copy(), t)
                u_0_no_turb = A_y0 @ wind.func_wind_no_turb(A_y0.T @ self.r_0[i_t, i_b, i_z, :].copy(), t)
                # u_0 = A_y0 @ np.array([0, wind.u_mean, 0.])
    #            u_0 = A_yaw.T @ np.array([0, wind.u_mean * (-r_0[i_t, i_b, i_z, 2]/wt.h_t)**0.2, 0.])
                u_2 = wt.A_02 @ u_0_no_turb
                u_3 = wt.A_03 @ u_0
                u_4 = wt.A_04 @ u_0
                
                # Wind velocity relative to the blade
                u_rel = u_4 + wt.A_34 @ self.w[i_t, i_b, i_z, :] - self.v_4[i_t, i_b, i_z, :]
                urel = np.linalg.norm(u_rel[:1])
                
                # Inflow angle
                phi = np.arctan2(u_rel[1], -u_rel[0]) # [rad]
                alpha = phi - (wt.twist[i_z] - wt.pitch) # [rad]
                self.phi[i_t, i_b, i_z] = phi
                self.alpha[i_t, i_b, i_z] = alpha
                
                # Airfoil lift coefficient
                if (dynamic_stall_on and i_t!=0):
                    Cl = self.dynamic_stall(wt, alpha, urel, i_t, i_b, i_z, t-self.t_vec[i_t-1])
                else:
                    Cl = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 1, i_z])
                
                # Airfoil drag coefficient
                Cd = np.interp(alpha, wt.airfoil[:, 0, i_z], wt.airfoil[:, 2, i_z])
                
                # Lift and drag coefficient per unit of length
                self.l[i_t, i_b, i_z] = 1./2. * wind.rho * urel**2 * wt.chord[i_z] * Cl # [N/m]
                self.d[i_t, i_b, i_z] = 1./2. * wind.rho * urel**2 * wt.chord[i_z] * Cd # [N/m]
                
                # Aerodynamic forces on the blade frame
                px = self.l[i_t, i_b, i_z]*np.sin(phi) - self.d[i_t, i_b, i_z]*np.cos(phi)
                py = self.l[i_t, i_b, i_z]*np.cos(phi) + self.d[i_t, i_b, i_z]*np.sin(phi)
                self.f_aero[i_t, i_b, i_z, :] = np.array([px, py, 0.])
                
                # Prandtl's tip loss correction
                z_3 = (wt.A_34.T @ wt.r_b[i_z, :])[2] # z coordinate on the undeflected rotor plane
                Z_3 = (wt.A_34.T @ wt.r_b[-1, :])[2] # max z coordinate on the undeflected rotor plane
                #
                if (u_mean!=0):
                    f = (wt.NB/2.) * (Z_3-z_3)/(z_3*np.abs(np.sin(phi)))
                    f_P = 2./np.pi * np.arccos(np.exp(-f))
                
                # Axial induction factor
                if (u_mean!=0):
                    self.a[i_t, i_b, i_z] = np.abs(self.w[i_t, i_b, i_z, 1]/u_mean)
                else:
                    self.a[i_t, i_b, i_z] = 0.
                
                # Glauert correction factor
                if (self.a[i_t, i_b, i_z]<=1./3.):
                    F_G = 1.
                else:
                    F_G = (1./4.)*(5.-3.*self.a[i_t, i_b, i_z])
                
                # u_prime
                self.u_prime_3[i_t, i_b, i_z, :] = u_3.copy() + np.array([0., F_G * self.w[i_t, i_b, i_z, 1], 0.])
                self.u_prime_2[i_t, i_b, i_z, :] = u_2.copy() + np.array([0., F_G * self.w[i_t, i_b, i_z, 1], 0.])
    
                uprime = np.linalg.norm(self.u_prime_3[i_t, i_b, i_z, :2])
                
                if (t!=self.t_vec[-1] and t==self.t_vec[i_t] and u_mean!=0):
                    if (dynamic_inflow_on):
                        self.dynamic_inflow(wt, wind, phi, uprime, f_P, z_3, Z_3, u_3, i_t, i_b, i_z, self.t_vec[i_t+1]-t)
                    else:
                        self.w_0[i_t+1, i_b, i_z, 0] = - (wt.NB*self.l[i_t, i_b, i_z]*np.sin(phi))/(4.*np.pi*wind.rho*z_3*f_P*uprime)
                        self.w_0[i_t+1, i_b, i_z, 1] = - (wt.NB*self.l[i_t, i_b, i_z]*np.cos(phi))/(4.*np.pi*wind.rho*z_3*f_P*uprime)
                        self.w_0[i_t+1, i_b, i_z, 2] = 0.
                    
                    # Yaw/tilt correction factor
                    self.YT_cf[i_t, i_b] = 1. + (z_3/Z_3*np.tan(self.chi[i_t]/2.)*np.cos(self.theta[i_t] + wt.eta - wt.downwind))
                    
                    # Induced velocity
                    self.w[i_t+1, i_b, i_z, :] = self.w_0[i_t+1, i_b, i_z, :] * self.YT_cf[i_t, i_b]
            
            # end for i_z
            
            # Gravity forces on the blade
            for i_z in reversed(range(len(wt.z))):
                # Gravity forces on the blade ()
                fg_0 = np.array([0, 0, wt.m[i_z]*wt.g])
                self.f_gravity[i_t, i_b, i_z, :] = wt.A_04 @ fg_0
            
            # Blade root moments
            
            self.M_aero[i_t, i_b, 0] = np.trapz( - self.f_aero[i_t, i_b, :, 1]*self.r_b4[i_t, i_b, :, 2] + self.f_aero[i_t, i_b, :, 2]*self.r_b4[i_t, i_b, :, 1], self.r_b4[i_t, i_b, :, 2])
            self.M_aero[i_t, i_b, 1] = np.trapz(   self.f_aero[i_t, i_b, :, 0]*self.r_b4[i_t, i_b, :, 2] - self.f_aero[i_t, i_b, :, 2]*self.r_b4[i_t, i_b, :, 0], self.r_b4[i_t, i_b, :, 2])
            self.M_aero[i_t, i_b, 2] = np.trapz( - self.f_aero[i_t, i_b, :, 0]*self.r_b4[i_t, i_b, :, 1] + self.f_aero[i_t, i_b, :, 1]*self.r_b4[i_t, i_b, :, 0], self.r_b4[i_t, i_b, :, 2])
            self.M_aero[i_t, i_b, :] = wt.A_23.T @ wt.A_34.T @ self.M_aero[i_t, i_b, :]
            
            self.M_gravity[i_t, i_b, 0] = np.trapz( - self.f_gravity[i_t, i_b, :, 1]*self.r_b4[i_t, i_b, :, 2] + self.f_gravity[i_t, i_b, :, 2]*self.r_b4[i_t, i_b, :, 1], self.r_b4[i_t, i_b, :, 2])
            self.M_gravity[i_t, i_b, 1] = np.trapz(   self.f_gravity[i_t, i_b, :, 0]*self.r_b4[i_t, i_b, :, 2] - self.f_gravity[i_t, i_b, :, 2]*self.r_b4[i_t, i_b, :, 0], self.r_b4[i_t, i_b, :, 2])
            self.M_gravity[i_t, i_b, 2] = np.trapz( - self.f_gravity[i_t, i_b, :, 0]*self.r_b4[i_t, i_b, :, 1] + self.f_gravity[i_t, i_b, :, 1]*self.r_b4[i_t, i_b, :, 0], self.r_b4[i_t, i_b, :, 2])
            self.M_gravity[i_t, i_b, :] = wt.A_23.T @ wt.A_34.T @ self.M_gravity[i_t, i_b, :]
                        
        # end for i_b
        
        # Wake skew angle
    #    if (t!=t_vec[-1] and t==t_vec[i_t] and wind.u_mean!=0):
    #        a_07 = np.zeros((wt.NB,))
    #        for i_b in range(wt.NB):
    #            a_07[i_b] = np.interp(0.7, wt.z/wt.R, a[i_t, i_b, :])
    #        a_averaged = np.sum(a_07[:])/wt.NB
    #        chi[i_t+1] = (0.6*a_averaged + 1.)*wt.yaw
        
        if (t!=self.t_vec[-1] and t==self.t_vec[i_t] and u_mean!=0):
            u_07 = np.zeros((wt.NB, 3))
            u_a = np.zeros((3,))
            for i_b in range(wt.NB):
                for i_d in range(3):
                    u_07[i_b, i_d] = np.interp(0.7, wt.z/wt.R, self.u_prime_2[i_t, i_b, :, i_d])
#                u_07[i_b, :] = wt.A_23.T @ u_07[i_b, :]
            u_a = np.sum(u_07[:, :], axis=0)/wt.NB
            uan = np.linalg.norm(u_a)
            self.chi[i_t+1] = np.arccos(u_a[1]/uan)
        
        # Forces on blade 0
        wt.f_0_x = self.f_aero[i_t, 0, :, 0]
        wt.f_0_y = self.f_aero[i_t, 0, :, 1]
        wt.f_0_z = self.f_aero[i_t, 0, :, 2]
        
        # Forces on blade 1
        wt.f_1_x = self.f_aero[i_t, 1, :, 0]
        wt.f_1_y = self.f_aero[i_t, 1, :, 1]
        wt.f_1_z = self.f_aero[i_t, 1, :, 2]
        
        # Forces on blade 2
        wt.f_2_x = self.f_aero[i_t, 2, :, 0]
        wt.f_2_y = self.f_aero[i_t, 2, :, 1]
        wt.f_2_z = self.f_aero[i_t, 2, :, 2]
        
        # Deformation
        wt.reinitilialise()
        if (wt.is_stiff):
            q_ddot = np.zeros(q.shape)
        else:
            M = wt.mass_matrix()
            C = wt.gyro_matrix()
            K = wt.stiffness_matrix()
            F = wt.force_vector()            
            q_ddot = np.linalg.solve(M, (F - C@q_dot - K@q))

        wt.q_ddot = q_ddot
        
        for i_b in range(wt.NB):
            ur_ddot = np.zeros( (len(wt.z), 3) )
            ur_ddot[:, 0] = q_ddot[3*i_b+0] * wt.phi_0_x[:] + q_ddot[3*i_b+1] * wt.phi_1_x[:] + q_ddot[3*i_b+2] * wt.phi_2_x[:]
            ur_ddot[:, 1] = q_ddot[3*i_b+0] * wt.phi_0_y[:] + q_ddot[3*i_b+1] * wt.phi_1_y[:] + q_ddot[3*i_b+2] * wt.phi_2_y[:]
            ur_ddot[:, 2] = 0.
            
            for i_d in range(3):
                self.f_inertia[i_t, i_b, :, i_d] = wt.m*ur_ddot[:, i_d]
            
            self.M_inertia[i_t, i_b, 0] = np.trapz( - self.f_inertia[i_t, i_b, :, 1]*self.r_b4[i_t, i_b, :, 2] + self.f_inertia[i_t, i_b, :, 2]*self.r_b4[i_t, i_b, :, 1], self.r_b4[i_t, i_b, :, 2])
            self.M_inertia[i_t, i_b, 1] = np.trapz(   self.f_inertia[i_t, i_b, :, 0]*self.r_b4[i_t, i_b, :, 2] - self.f_inertia[i_t, i_b, :, 2]*self.r_b4[i_t, i_b, :, 0], self.r_b4[i_t, i_b, :, 2])
            self.M_inertia[i_t, i_b, 2] = np.trapz( - self.f_inertia[i_t, i_b, :, 0]*self.r_b4[i_t, i_b, :, 1] + self.f_inertia[i_t, i_b, :, 1]*self.r_b4[i_t, i_b, :, 0], self.r_b4[i_t, i_b, :, 2])
            self.M_inertia[i_t, i_b, :] = - self.A_23[i_t, i_b, :, :].T @ wt.A_34.T @ self.M_inertia[i_t, i_b, :]
            
            # self.M_inertia[i_t, i_b, 0] = np.trapz( - wt.m*ur_ddot[:, 1]*self.r_b4[i_t, i_b, :, 2] + wt.m*ur_ddot[:, 2]*self.r_b4[i_t, i_b, :, 1], self.r_b4[i_t, i_b, :, 2])
            # self.M_inertia[i_t, i_b, 1] = np.trapz(   wt.m*ur_ddot[:, 0]*self.r_b4[i_t, i_b, :, 2] - wt.m*ur_ddot[:, 2]*self.r_b4[i_t, i_b, :, 0], self.r_b4[i_t, i_b, :, 2])
            # self.M_inertia[i_t, i_b, 2] = np.trapz( - wt.m*ur_ddot[:, 0]*self.r_b4[i_t, i_b, :, 1] + wt.m*ur_ddot[:, 1]*self.r_b4[i_t, i_b, :, 0], self.r_b4[i_t, i_b, :, 2])
            # self.M_inertia[i_t, i_b, :] = - self.A_23[i_t, i_b, :, :].T @ wt.A_34.T @ self.M_inertia[i_t, i_b, :]
            
        self.M_blade[i_t, :, :] = self.M_aero[i_t, :, :] + self.M_gravity[i_t, :, :] +self. M_inertia[i_t, :, :]
        
        self.M_rot[i_t, :] = self.M_blade[i_t, 0, :] + self.M_blade[i_t, 1, :] + self.M_blade[i_t, 2, :]
        self.Power_rot[i_t] = self.M_rot[i_t, 1] * self.Omega[i_t]
        
        # Shaft azimuth angle
        if (t!=self.t_vec[-1] and t==self.t_vec[i_t]):
            delta_t = self.t_vec[i_t + 1] - t
            self.theta[i_t+1] = self.theta[i_t] + wt.Omega * delta_t
            
            # no variation in Omega yet
            self.Omega[i_t+1] = self.Omega[i_t]
    
        return q_ddot