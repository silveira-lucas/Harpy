from sympy.printing.pycode import NumPyPrinter

#%%

class MyPrinter(NumPyPrinter):
    
    def _print_Function(self, expr):
        '''
        This method overrides the original _print_Funciton from Sympy
        NumpyPrinter. With this method only the function attribue name is
        printed, the function variables are not printed. This is convenient
        when in the numerical code the function is actually an array and not
        a numerical function.

        Parameters
        ----------
        expr : sympy.Function
               sympy undefined function

        Returns
        -------
        expr.name : str
                    sympy.Function attribute name
        '''
        return '%s' % (expr.name )
    
    def _print_Derivative(self, expr):
        '''
        This method overrides the original _print_Derivative from Sympy 
        NumpyPrinter. This method adds the prefix '_dot', '_ddot', '_dddot', 
        ..., to the function name to be printed. This is convenient when in 
        the numerical code the function derivative is stored inside a 
        numpy.ndarray.
        
        Parameters
        ----------
        expr : sympy.function.Derivative
               n**{th} derivative of an undefined sympy function
        
        Returns
        -------
        a : str
            derivative array name
        '''
        
        # Get the derivative order
        dim = 'ot'
        for i in range(expr.derivative_count):
            dim = 'd' + dim
        
        # Get the derivative argument name
        a = expr.args[0].name

        # Check if the derivative name contains '[]'
        j = None
        for i, char in enumerate(a):
            if (char == '['):
                j = i
        if (j is not None):
            a = a[:j] + '_'+ dim + a[j:]
        else:
            a = a[:] + '_' + dim

        return a
    
    def _print_Integral(self, expr):
        '''
        This method translates the symbolic integral into a numerical
        trapezoidal integral, i.e. the numpy.trapz function.

        Parameters
        ----------
        expr : sympy expression

        Returns
        -------
        expr_num : str
                   numpy.trapz integral of the sympy expression
        '''
        
        part_1 = 'numpy.trapz'
        part_2 = MyPrinter().doprint(expr.args[0])
        part_3 = expr.args[1][0]
        
        expr_num = '%s(%s,%s)' % (part_1, part_2, part_3)
        return expr_num
    
    @staticmethod
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

    
    @staticmethod
    def gen_file(Mx, Cx, Kx, Fx, q, dict_xi, dict_yt, dict_constant, transformation_matrices, rotation_matrices, position_vectors):
        '''
        This function generates the wind turbine structural dynamics class. 
        The class contains the methods to calculate the instantaneous local 
        position vectors, transformation matrices, angular velocity matrices 
        and the linearised differential system of equations of motion, i.e. 
        the system mass matrix M, the matrix proportional to the general 
        coordinates first derivatives and the stiffness matrix.        
        Parameters
        ----------
        Mx : sympy.matrix [:, :]
             Mass matrix. The integrals and trigonometric functions are written
             in the matrix using the keys contained in dict_xi and dict_yt.
        Cx : sympy.matrix [:, :]
             Matrix proportional to the general coordinates derivatives. The
             integrals and trigonometric functions are written in the matrix
             using the keys contained in dict_xi and dict_yt.
        Kx : sympy.matrix [:, :]
             Stiffness matrix. The integrals and trigonometric functions are
             written in the matrix using the keys contained in dict_xi and
             dict_yt.
        Fx : sympy.matrix [:, 1]
             Forcing vector. The integrals and trigonometric functions are
             written in the matrix using the keys contained in dict_xi and
             dict_yt.
        q : sympy.matrix [:, 1]
            List of sympy.symbols of the generalised degrees of freedom.
        dict_xi : dict
                  Dictioary of the integrals contained in the equations of
                  motion and their correspondent keys.
        dict_yt : dict
                  Dictioary of the trigonometric functions contained in the
                  equations of motion and their correspondent keys.
        dict_constant : dict
                        Associative array containing the information if the
                        integral contains terms which vary with time-step or
                        not. The integrals which do not vary with time-step are
                        calculated only once. Integrasl which do vary with time
                        are recalculated every time step before calculating M,
                        C, K and F.
        transformation_matrices : dict{'str': 'sympy.matrix'}
                                  Transformation matrices from one reference
                                  frame to another.
        rotation_matrices : dict{'str': 'sympy.matrix'}
                            Angular velocity tensors.
        position_vectors : dict{'str': 'sympy.matrix'}
                           Position vectors

        '''
        
        # Loading the doprint function into code_gen
        code_gen = MyPrinter().doprint

        #
        matrices = {**transformation_matrices, **rotation_matrices}

        # Open code file
        with open('wind_turbine_structural.py', 'w') as f:
            
            # Importing the libraries
            print('import numpy \n', file=f)
            print('#%% \n', file=f)
            print('class WindTurbineStructural(object):', file=f)
            
            # Polite message :)
            ident = '    '
            print(ident+"'''", file=f)
            print(ident+'Created automatically by Harpy symbolic module', file=f)
            print(ident+"'''"+'\n', file=f)
            
            # Print the __init__ method
            print(ident+ 'def __init__(self):', file=f)
            message = """This init method only declares the variables as place as place holders, i.e. without declaring
                         the type or allocating memory.The instance creator(s) methods will be implemented via classmethods
                         for allowing more flexibility for different imput methods."""
            
            print(ident*2+"'''", file=f)
            print(MyPrinter.print_comment(message, ident=ident*2), file=f)
            print(ident*2+"'''", file=f)
            print(ident*2+'self.xint = numpy.zeros((%i,), dtype=float)' %len(dict_xi), file=f)
            print(ident*2+'self.ytrig = numpy.zeros((%i,), dtype=float)' %len(dict_yt), file=f)
            print(ident*2+'self.q = numpy.zeros((%i,), dtype=float)' %len(q), file=f)
            print(ident*2+'self.q_dot = numpy.zeros((%i,), dtype=float)' %len(q), file=f)
            print(ident*2+'self.q_ddot = numpy.zeros((%i,), dtype=float)' %len(q), file=f)
            print(ident*2, file=f)
            
#            print(ident+'def reference_matrices(self, wt):', file=f)
#            print(2*ident+'A_23 = numpy.zeros((3, 3), dtype=float)', file=f)
#            for i in range(A_23.shape[0]):
#                for j in range(A_23.shape[1]):
#                    if (not code_gen(A_23[i,j])=='0'):
#                        print(ident*2+'A_23[%i,%i] = %s' %(i, j, code_gen(A_23[i,j])), file=f)
#            print(ident*2, file=f)

            # Print reference matrices
            print(ident+'def reference_matrices(self):', file=f)
            message = """This method calculates the transformation tensors between reference frames and the angular velocity 
                         tensors associated with the movig reference frames."""
            message = MyPrinter.print_comment(message, ident*2)
            print(ident*2+"'''", file=f)            
            print(message, file=f)
            print(ident*2+"'''", file=f)
            for key, value in matrices.items():
                print(ident*2+'self.%s = numpy.zeros((%i, %i), dtype=float)' %(key, value.shape[0], value.shape[1]), file=f)
                for i in range(value.shape[0]):
                    for j in range(value.shape[1]):
                        if (code_gen(value[i, j]) != '0'):
                            print(ident*2+'self.%s[%i, %i] = %s' %(key, i, j, code_gen(value[i, j])), file=f)
                print(ident*2+'#', file=f)
            print(ident*2+'#', file=f)
            print('', file=f)
            
            # Print the position vectors
            print(ident+'def position_vectors(self):', file=f)
            message = """Theis method prints the local position vectors associated with the nacelle, shaft tip and
                      blade section."""
            message = MyPrinter.print_comment(message, ident*2)
            print(ident*2+"'''", file =f)
            print(message, file=f)
            print(ident*2+"'''", file =f)            
            for key, value in position_vectors.items():
                print(ident*2+'self.%s = numpy.zeros((%i,), dtype=float)' %(key, value.shape[0]), file=f)
                for i in range(value.shape[0]):
                    if (code_gen(value[i]) != '0'):
                        print(ident*2+'self.%s[%i] = %s' %(key, i, code_gen(value[i])), file=f)
                print(ident*2+'#', file=f)
            print(ident*2+'#', file=f)
            print('', file=f)
            
            # Print the initialise method
            print(ident+ 'def initilialise(self):', file=f)
            message = """This method calculates the integrals and trigonometric funcitons included on M, C, K and F,
                         both those which vary with time-step and those which do not.
                         The same integrals appear multiple times on the matrices and force vector. Calculating the
                         integrals and trigonometric functions only once makes the code a bit more efficient."""
            message = MyPrinter.print_comment(message, ident*2)
            print(ident*2+"'''", file =f)
            print(message, file=f)
            print(ident*2+"'''", file =f)                        
            for key, value in dict_xi.items():
                print(ident*2+'%s = %s' % (key.name, code_gen(value)), file=f)
            for key, value in dict_yt.items():
                print(ident*2+'%s = %s' % (key.name, code_gen(value)), file=f)
            print(ident*2, file=f)
            
            # Print the reinitialise method
            print(ident+ 'def reinitilialise(self):', file=f)
            message = """This method recalculates the integrals and trigonometric funcitons included on M, C, K and F
                         which vary with time-step.
                         The same integrals appear multiple times on the matrices and force vector. Calculating the
                         integrals and trigonometric functions only once makes the code a bit more efficient."""
            message = MyPrinter.print_comment(message, ident*2)
            print(ident*2+"'''", file =f)
            print(message, file=f)
            print(ident*2+"'''", file =f)                        
            for key, value in dict_xi.items():
                if (not dict_constant[key]):            
                    print(ident*2+'%s = %s' % (str(key), code_gen(value)), file=f)
            for key, value in dict_yt.items():
                print(ident*2+'%s = %s' % (str(key), code_gen(value)), file=f)
            print(ident*2, file=f)
            
            # Print the mass_matrix method
            print(ident+ 'def mass_matrix(self):', file=f)    
            message = """This method calculates the mass matrix."""
            message = MyPrinter.print_comment(message, ident*2) + '\n'
            print(ident*2+"'''", file =f)
            print(message, file=f)
            print(ident*2+'Returns', file=f)
            print(ident*2+'-------', file=f)
            print(ident*2+'M : numpy.array [:, :], dtype=float', file=f)
            print(ident*2+'    Mass matrix', file=f)            
            print(ident*2+"'''", file =f)
            print(ident*2+'M = numpy.zeros(%s, dtype=float)' %(str(Mx.shape)), file=f)
            for i in range(Mx.shape[0]):
                for j in range(Mx.shape[1]):
                    if (not code_gen(Mx[i,j])=='0'):
                        print(ident*2+'M[%i,%i] = %s' %(i, j, code_gen(Mx[i,j])), file=f)
            print(ident*2+'return M', file=f)
            print(ident, file=f)
            
            # Print the gyro_matrix method
            print(ident+ 'def gyro_matrix(self):', file=f)
            message = """This method calculates matrix proportional to the first derivative of the generalised coordinates."""
            message = MyPrinter.print_comment(message, ident*2) + '\n'
            print(ident*2+"'''", file =f)
            print(message, file=f)
            print(ident*2+'Returns', file=f)
            print(ident*2+'-------', file=f)
            print(ident*2+'F : numpy.array [:], dtype=float', file=f)
            print(ident*2+'    Forcing vector', file=f)            
            print(ident*2+"'''", file =f)

            print(ident*2+'G = numpy.zeros(%s, dtype=float)' %(str(Cx.shape)), file=f)
            for i in range(Cx.shape[0]):
                for j in range(Cx.shape[1]):
                    if (not code_gen(Cx[i,j])=='0'):
                        print(ident*2+'G[%s,%s] = %s' %(str(i), str(j), code_gen(Cx[i,j])), file=f)
            print(ident*2+'return G', file=f)
            print(ident, file=f)
            
            # Print the stiffness_matrix method
            print(ident+ 'def stiffness_matrix(self):', file=f)
            message = """This method calculates the stiffness matrix."""
            message = MyPrinter.print_comment(message, ident*2) + '\n'
            print(ident*2+"'''", file =f)
            print(message, file=f)
            print(ident*2+'Returns', file=f)
            print(ident*2+'-------', file=f)
            print(ident*2+'K : numpy.array [:, :], dtype=float', file=f)
            print(ident*2+'    Stiffness matrix', file=f)            
            print(ident*2+"'''", file =f)
            print(ident*2+'K = numpy.zeros(%s, dtype=float)' %(str(Kx.shape)), file=f)
            for i in range(Kx.shape[0]):
                for j in range(Kx.shape[1]):
                    if (not code_gen(Kx[i,j])=='0'):
                        print(ident*2+'K[%s,%s] = %s' %(str(i), str(j), code_gen(Kx[i,j])), file=f)
            print(ident*2+'return K', file=f)
            print(ident, file=f)
            
            # Print the force_vector method
            print(ident+ 'def force_vector(self):', file=f)
            message = """This method calculates the generalised forcing vector."""
            message = MyPrinter.print_comment(message, ident*2) + '\n'
            print(ident*2+"'''", file =f)
            print(message, file=f)
            print(ident*2+'Returns', file=f)
            print(ident*2+'-------', file=f)
            print(ident*2+'F : numpy.array [:], dtype=float', file=f)
            print(ident*2+'    Forcing vector', file=f)            
            print(ident*2+"'''", file =f)
            print(ident*2+'F = numpy.zeros((%i,), dtype=float)' %(Fx.shape[0]), file=f)
            for i in range(Fx.shape[0]):
                if (not code_gen(Fx[i])=='0'):
                    print(ident*2+'F[%s] = %s' %(str(i), code_gen(Fx[i])), file=f)
            print(ident*2+'return F', file=f)
            
            