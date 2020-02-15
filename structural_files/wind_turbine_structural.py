import numpy 

#%% 

class WindTurbineStructural(object):
    '''
    Created automatically by Harpy symbolic module
    '''

    def __init__(self):
        '''
        This init method only declares the variables as place as place 
        holders, i.e. without declaring the type or allocating memory.The 
        instance creator(s) methods will be implemented via classmethods for 
        allowing more flexibility for different imput methods.
        '''
        self.xint = numpy.zeros((59,), dtype=float)
        self.ytrig = numpy.zeros((6,), dtype=float)
        self.q = numpy.zeros((11,), dtype=float)
        self.q_dot = numpy.zeros((11,), dtype=float)
        self.q_ddot = numpy.zeros((11,), dtype=float)
        
    def reference_matrices(self):
        '''
        This method calculates the transformation tensors between reference 
        frames and the angular velocity tensors associated with the movig 
        reference frames.
        '''
        self.A_01 = numpy.zeros((3, 3), dtype=float)
        self.A_01[0, 0] = 1
        self.A_01[1, 1] = 1
        self.A_01[2, 2] = 1
        #
        self.A_12 = numpy.zeros((3, 3), dtype=float)
        self.A_12[0, 0] = 1
        self.A_12[1, 1] = 1
        self.A_12[2, 2] = 1
        #
        self.A_23 = numpy.zeros((3, 3), dtype=float)
        self.A_23[0, 0] = numpy.cos(self.eta + self.theta)
        self.A_23[0, 2] = -numpy.sin(self.eta + self.theta)
        self.A_23[1, 1] = 1
        self.A_23[2, 0] = numpy.sin(self.eta + self.theta)
        self.A_23[2, 2] = numpy.cos(self.eta + self.theta)
        #
        self.A_34 = numpy.zeros((3, 3), dtype=float)
        self.A_34[0, 0] = 1
        self.A_34[1, 1] = 1
        self.A_34[2, 2] = 1
        #
        self.Omega_01_1 = numpy.zeros((3, 3), dtype=float)
        #
        self.Omega_01_3 = numpy.zeros((3, 3), dtype=float)
        #
        self.Omega_01_4 = numpy.zeros((3, 3), dtype=float)
        #
        self.Omega_23_3 = numpy.zeros((3, 3), dtype=float)
        self.Omega_23_3[0, 2] = self.Omega
        self.Omega_23_3[2, 0] = -self.Omega
        #
        self.Omega_23_4 = numpy.zeros((3, 3), dtype=float)
        self.Omega_23_4[0, 2] = self.Omega
        self.Omega_23_4[2, 0] = -self.Omega
        #
        #

    def position_vectors(self):
        '''
        Theis method prints the local position vectors associated with the 
        nacelle, shaft tip and blade section.
        '''
        self.r_t0 = numpy.zeros((3,), dtype=float)
        self.r_t0[0] = self.q[9]
        self.r_t0[1] = self.q[10]
        self.r_t0[2] = -self.h_t
        #
        self.r_s3 = numpy.zeros((3,), dtype=float)
        self.r_s3[1] = -self.s_l
        #
        self.v_t0 = numpy.zeros((3,), dtype=float)
        self.v_t0[0] = self.q_dot[9]
        self.v_t0[1] = self.q_dot[10]
        #
        #

    def initilialise(self):
        '''
        This method calculates the integrals and trigonometric funcitons 
        included on M, C, K and F, both those which vary with time-step and 
        those which do not. The same integrals appear multiple times on the 
        matrices and force vector. Calculating the integrals and trigonometric 
        functions only once makes the code a bit more efficient.
        '''
        self.xint[0] = numpy.trapz(self.f_0_y*self.phi_2_y,self.z)
        self.xint[1] = numpy.trapz(self.z*self.f_2_z,self.z)
        self.xint[2] = numpy.trapz(self.f_0_y*self.phi_1_y,self.z)
        self.xint[3] = numpy.trapz(self.f_1_z,self.z)
        self.xint[4] = numpy.trapz(self.f_0_z,self.z)
        self.xint[5] = numpy.trapz(self.f_2_z,self.z)
        self.xint[6] = numpy.trapz(self.z*self.m*self.phi_1_x,self.z)
        self.xint[7] = numpy.trapz(self.m*self.phi_2_y**2,self.z)
        self.xint[8] = numpy.trapz(self.m*self.phi_2_x**2,self.z)
        self.xint[9] = numpy.trapz(self.m*self.phi_2_x,self.z)
        self.xint[10] = numpy.trapz(self.f_2_x*self.phi_1_x,self.z)
        self.xint[11] = numpy.trapz(self.f_1_x*self.phi_1_x,self.z)
        self.xint[12] = numpy.trapz(self.f_0_x*self.phi_0_x,self.z)
        self.xint[13] = numpy.trapz(self.z*self.f_0_z,self.z)
        self.xint[14] = numpy.trapz(self.f_0_x*self.phi_1_x,self.z)
        self.xint[15] = numpy.trapz(self.m,self.z)
        self.xint[16] = numpy.trapz(self.m*self.phi_1_x,self.z)
        self.xint[17] = numpy.trapz(self.f_0_y,self.z)
        self.xint[18] = numpy.trapz(self.z**2*self.m,self.z)
        self.xint[19] = numpy.trapz(self.m*self.x,self.z)
        self.xint[20] = numpy.trapz(self.f_1_y,self.z)
        self.xint[21] = numpy.trapz(self.m*self.x**2,self.z)
        self.xint[22] = numpy.trapz(self.m*self.phi_1_y,self.z)
        self.xint[23] = numpy.trapz(self.f_0_y*self.phi_0_y,self.z)
        self.xint[24] = numpy.trapz(self.f_2_y,self.z)
        self.xint[25] = numpy.trapz(self.f_0_y*self.y,self.z)
        self.xint[26] = numpy.trapz(self.m*self.phi_0_x**2,self.z)
        self.xint[27] = numpy.trapz(self.f_1_x*self.x,self.z)
        self.xint[28] = numpy.trapz(self.f_2_x*self.x,self.z)
        self.xint[29] = numpy.trapz(self.f_1_x*self.phi_2_x,self.z)
        self.xint[30] = numpy.trapz(self.m*self.phi_0_x*self.x,self.z)
        self.xint[31] = numpy.trapz(self.f_2_y*self.phi_0_y,self.z)
        self.xint[32] = numpy.trapz(self.f_2_y*self.y,self.z)
        self.xint[33] = numpy.trapz(self.z*self.f_1_z,self.z)
        self.xint[34] = numpy.trapz(self.f_0_x,self.z)
        self.xint[35] = numpy.trapz(self.f_2_x*self.phi_2_x,self.z)
        self.xint[36] = numpy.trapz(self.f_1_y*self.phi_0_y,self.z)
        self.xint[37] = numpy.trapz(self.f_1_y*self.y,self.z)
        self.xint[38] = numpy.trapz(self.m*self.phi_0_x,self.z)
        self.xint[39] = numpy.trapz(self.m*self.phi_0_y**2,self.z)
        self.xint[40] = numpy.trapz(self.m*self.phi_2_y,self.z)
        self.xint[41] = numpy.trapz(self.z*self.m*self.phi_2_x,self.z)
        self.xint[42] = numpy.trapz(self.f_0_x*self.phi_2_x,self.z)
        self.xint[43] = numpy.trapz(self.m*self.phi_0_y,self.z)
        self.xint[44] = numpy.trapz(self.f_2_x,self.z)
        self.xint[45] = numpy.trapz(self.f_1_x,self.z)
        self.xint[46] = numpy.trapz(self.m*self.phi_1_x*self.x,self.z)
        self.xint[47] = numpy.trapz(self.f_2_y*self.phi_1_y,self.z)
        self.xint[48] = numpy.trapz(self.m*self.phi_1_x**2,self.z)
        self.xint[49] = numpy.trapz(self.z*self.m,self.z)
        self.xint[50] = numpy.trapz(self.f_1_x*self.phi_0_x,self.z)
        self.xint[51] = numpy.trapz(self.f_2_x*self.phi_0_x,self.z)
        self.xint[52] = numpy.trapz(self.m*self.phi_1_y**2,self.z)
        self.xint[53] = numpy.trapz(self.f_0_x*self.x,self.z)
        self.xint[54] = numpy.trapz(self.z*self.m*self.phi_0_x,self.z)
        self.xint[55] = numpy.trapz(self.m*self.phi_2_x*self.x,self.z)
        self.xint[56] = numpy.trapz(self.f_1_y*self.phi_1_y,self.z)
        self.xint[57] = numpy.trapz(self.f_1_y*self.phi_2_y,self.z)
        self.xint[58] = numpy.trapz(self.f_2_y*self.phi_2_y,self.z)
        self.ytrig[0] = numpy.cos(self.theta + (1/3)*numpy.pi)
        self.ytrig[1] = numpy.sin(self.theta + (1/3)*numpy.pi)
        self.ytrig[2] = numpy.cos(self.theta)
        self.ytrig[3] = numpy.sin(self.theta + (1/6)*numpy.pi)
        self.ytrig[4] = numpy.cos(self.theta + (1/6)*numpy.pi)
        self.ytrig[5] = numpy.sin(self.theta)
        
    def reinitilialise(self):
        '''
        This method recalculates the integrals and trigonometric funcitons 
        included on M, C, K and F which vary with time-step. The same 
        integrals appear multiple times on the matrices and force vector. 
        Calculating the integrals and trigonometric functions only once makes 
        the code a bit more efficient.
        '''
        self.xint[0] = numpy.trapz(self.f_0_y*self.phi_2_y,self.z)
        self.xint[2] = numpy.trapz(self.f_0_y*self.phi_1_y,self.z)
        self.xint[10] = numpy.trapz(self.f_2_x*self.phi_1_x,self.z)
        self.xint[11] = numpy.trapz(self.f_1_x*self.phi_1_x,self.z)
        self.xint[12] = numpy.trapz(self.f_0_x*self.phi_0_x,self.z)
        self.xint[14] = numpy.trapz(self.f_0_x*self.phi_1_x,self.z)
        self.xint[17] = numpy.trapz(self.f_0_y,self.z)
        self.xint[20] = numpy.trapz(self.f_1_y,self.z)
        self.xint[23] = numpy.trapz(self.f_0_y*self.phi_0_y,self.z)
        self.xint[24] = numpy.trapz(self.f_2_y,self.z)
        self.xint[25] = numpy.trapz(self.f_0_y*self.y,self.z)
        self.xint[27] = numpy.trapz(self.f_1_x*self.x,self.z)
        self.xint[28] = numpy.trapz(self.f_2_x*self.x,self.z)
        self.xint[29] = numpy.trapz(self.f_1_x*self.phi_2_x,self.z)
        self.xint[31] = numpy.trapz(self.f_2_y*self.phi_0_y,self.z)
        self.xint[32] = numpy.trapz(self.f_2_y*self.y,self.z)
        self.xint[34] = numpy.trapz(self.f_0_x,self.z)
        self.xint[35] = numpy.trapz(self.f_2_x*self.phi_2_x,self.z)
        self.xint[36] = numpy.trapz(self.f_1_y*self.phi_0_y,self.z)
        self.xint[37] = numpy.trapz(self.f_1_y*self.y,self.z)
        self.xint[42] = numpy.trapz(self.f_0_x*self.phi_2_x,self.z)
        self.xint[44] = numpy.trapz(self.f_2_x,self.z)
        self.xint[45] = numpy.trapz(self.f_1_x,self.z)
        self.xint[47] = numpy.trapz(self.f_2_y*self.phi_1_y,self.z)
        self.xint[50] = numpy.trapz(self.f_1_x*self.phi_0_x,self.z)
        self.xint[51] = numpy.trapz(self.f_2_x*self.phi_0_x,self.z)
        self.xint[53] = numpy.trapz(self.f_0_x*self.x,self.z)
        self.xint[56] = numpy.trapz(self.f_1_y*self.phi_1_y,self.z)
        self.xint[57] = numpy.trapz(self.f_1_y*self.phi_2_y,self.z)
        self.xint[58] = numpy.trapz(self.f_2_y*self.phi_2_y,self.z)
        self.ytrig[0] = numpy.cos(self.theta + (1/3)*numpy.pi)
        self.ytrig[1] = numpy.sin(self.theta + (1/3)*numpy.pi)
        self.ytrig[2] = numpy.cos(self.theta)
        self.ytrig[3] = numpy.sin(self.theta + (1/6)*numpy.pi)
        self.ytrig[4] = numpy.cos(self.theta + (1/6)*numpy.pi)
        self.ytrig[5] = numpy.sin(self.theta)
        
    def mass_matrix(self):
        '''
        This method calculates the mass matrix.

        Returns
        -------
        M : numpy.array [:, :], dtype=float
            Mass matrix
        '''
        M = numpy.zeros((11, 11), dtype=float)
        M[0,0] = self.xint[26] + self.xint[39]
        M[0,9] = -self.ytrig[2]*self.xint[38]
        M[0,10] = self.xint[43]
        M[1,1] = self.xint[48] + self.xint[52]
        M[1,9] = -self.ytrig[2]*self.xint[16]
        M[1,10] = self.xint[22]
        M[2,2] = self.xint[7] + self.xint[8]
        M[2,9] = -self.ytrig[2]*self.xint[9]
        M[2,10] = self.xint[40]
        M[3,3] = self.xint[26] + self.xint[39]
        M[3,9] = self.ytrig[3]*self.xint[38]
        M[3,10] = self.xint[43]
        M[4,4] = self.xint[48] + self.xint[52]
        M[4,9] = self.ytrig[3]*self.xint[16]
        M[4,10] = self.xint[22]
        M[5,5] = self.xint[7] + self.xint[8]
        M[5,9] = self.ytrig[3]*self.xint[9]
        M[5,10] = self.xint[40]
        M[6,6] = self.xint[26] + self.xint[39]
        M[6,9] = self.ytrig[0]*self.xint[38]
        M[6,10] = self.xint[43]
        M[7,7] = self.xint[48] + self.xint[52]
        M[7,9] = self.ytrig[0]*self.xint[16]
        M[7,10] = self.xint[22]
        M[8,8] = self.xint[7] + self.xint[8]
        M[8,9] = self.ytrig[0]*self.xint[9]
        M[8,10] = self.xint[40]
        M[9,0] = -self.ytrig[2]*self.xint[38]
        M[9,1] = -self.ytrig[2]*self.xint[16]
        M[9,2] = -self.ytrig[2]*self.xint[9]
        M[9,3] = self.ytrig[3]*self.xint[38]
        M[9,4] = self.ytrig[3]*self.xint[16]
        M[9,5] = self.ytrig[3]*self.xint[9]
        M[9,6] = self.ytrig[0]*self.xint[38]
        M[9,7] = self.ytrig[0]*self.xint[16]
        M[9,8] = self.ytrig[0]*self.xint[9]
        M[9,9] = self.m_h + self.m_n + 3*self.xint[15]
        M[10,0] = self.xint[43]
        M[10,1] = self.xint[22]
        M[10,2] = self.xint[40]
        M[10,3] = self.xint[43]
        M[10,4] = self.xint[22]
        M[10,5] = self.xint[40]
        M[10,6] = self.xint[43]
        M[10,7] = self.xint[22]
        M[10,8] = self.xint[40]
        M[10,10] = self.m_h + self.m_n + 3*self.xint[15]
        return M
    
    def gyro_matrix(self):
        '''
        This method calculates matrix proportional to the first derivative of 
        the generalised coordinates.

        Returns
        -------
        F : numpy.array [:], dtype=float
            Forcing vector
        '''
        G = numpy.zeros((11, 11), dtype=float)
        G[9,0] = 2*self.ytrig[5]*self.xint[38]*self.Omega
        G[9,1] = 2*self.ytrig[5]*self.xint[16]*self.Omega
        G[9,2] = 2*self.ytrig[5]*self.xint[9]*self.Omega
        G[9,3] = 2*self.ytrig[4]*self.xint[38]*self.Omega
        G[9,4] = 2*self.ytrig[4]*self.xint[16]*self.Omega
        G[9,5] = 2*self.ytrig[4]*self.xint[9]*self.Omega
        G[9,6] = -2*self.ytrig[1]*self.xint[38]*self.Omega
        G[9,7] = -2*self.ytrig[1]*self.xint[16]*self.Omega
        G[9,8] = -2*self.ytrig[1]*self.xint[9]*self.Omega
        return G
    
    def stiffness_matrix(self):
        '''
        This method calculates the stiffness matrix.

        Returns
        -------
        K : numpy.array [:, :], dtype=float
            Stiffness matrix
        '''
        K = numpy.zeros((11, 11), dtype=float)
        K[0,0] = self.omega[0]**2*self.xint[26] + self.omega[0]**2*self.xint[39] - self.xint[26]*self.Omega**2
        K[1,1] = self.omega[1]**2*self.xint[48] + self.omega[1]**2*self.xint[52] - self.xint[48]*self.Omega**2
        K[2,2] = self.omega[2]**2*self.xint[7] + self.omega[2]**2*self.xint[8] - self.xint[8]*self.Omega**2
        K[3,3] = self.omega[0]**2*self.xint[26] + self.omega[0]**2*self.xint[39] - self.xint[26]*self.Omega**2
        K[4,4] = self.omega[1]**2*self.xint[48] + self.omega[1]**2*self.xint[52] - self.xint[48]*self.Omega**2
        K[5,5] = self.omega[2]**2*self.xint[7] + self.omega[2]**2*self.xint[8] - self.xint[8]*self.Omega**2
        K[6,6] = self.omega[0]**2*self.xint[26] + self.omega[0]**2*self.xint[39] - self.xint[26]*self.Omega**2
        K[7,7] = self.omega[1]**2*self.xint[48] + self.omega[1]**2*self.xint[52] - self.xint[48]*self.Omega**2
        K[8,8] = self.omega[2]**2*self.xint[7] + self.omega[2]**2*self.xint[8] - self.xint[8]*self.Omega**2
        K[9,0] = self.ytrig[2]*self.xint[38]*self.Omega**2
        K[9,1] = self.ytrig[2]*self.xint[16]*self.Omega**2
        K[9,2] = self.ytrig[2]*self.xint[9]*self.Omega**2
        K[9,3] = -self.ytrig[3]*self.xint[38]*self.Omega**2
        K[9,4] = -self.ytrig[3]*self.xint[16]*self.Omega**2
        K[9,5] = -self.ytrig[3]*self.xint[9]*self.Omega**2
        K[9,6] = -self.ytrig[0]*self.xint[38]*self.Omega**2
        K[9,7] = -self.ytrig[0]*self.xint[16]*self.Omega**2
        K[9,8] = -self.ytrig[0]*self.xint[9]*self.Omega**2
        K[9,9] = self.k_x
        K[10,10] = self.k_y
        return K
    
    def force_vector(self):
        '''
        This method calculates the generalised forcing vector.

        Returns
        -------
        F : numpy.array [:], dtype=float
            Forcing vector
        '''
        F = numpy.zeros((11,), dtype=float)
        F[0] = self.xint[12] + self.xint[23] + self.xint[30]*self.Omega**2
        F[1] = self.xint[14] + self.xint[2] + self.xint[46]*self.Omega**2
        F[2] = self.xint[0] + self.xint[42] + self.xint[55]*self.Omega**2
        F[3] = self.xint[30]*self.Omega**2 + self.xint[36] + self.xint[50]
        F[4] = self.xint[11] + self.xint[46]*self.Omega**2 + self.xint[56]
        F[5] = self.xint[29] + self.xint[55]*self.Omega**2 + self.xint[57]
        F[6] = self.xint[30]*self.Omega**2 + self.xint[31] + self.xint[51]
        F[7] = self.xint[10] + self.xint[46]*self.Omega**2 + self.xint[47]
        F[8] = self.xint[35] + self.xint[55]*self.Omega**2 + self.xint[58]
        F[9] = self.ytrig[0]*self.xint[44] + self.ytrig[1]*self.xint[5] - self.ytrig[2]*self.xint[34] + self.ytrig[3]*self.xint[45] - self.ytrig[4]*self.xint[3] - self.ytrig[5]*self.xint[4]
        F[10] = self.xint[17] + self.xint[20] + self.xint[24]
        return F
