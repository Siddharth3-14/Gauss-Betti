import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.fftpack


# class GaussianRandomField:
#     """GaussianRandomField

#     The class for making Gaussian random field with specified spectral index and size of grid.
        
#     Attributes:
#         Nzise (int): size of the grid.
#         n (int): Spectral index of the power law used to generate the Gaussian Random Field.
#         k_ind (array): Grid in the fourier space.
#         PowerSpectrum (array): The power spectrum grid made using the spectral index used to make the Gaussian Random Field.
#         corr_s (array): Correlation matrix in the fourier space.
#         corr_f (array): Correlation matrix in the spatial space.

#     """
#     def __init__(self, Nsize, n):
#         self.Nsize = Nsize
#         self.n = n
#         self.fourier_space_ind()
#         self.PowerSpectrum_grid_generator()
#         self.gen_correlation()


#     def fourier_space_ind(self):
#         """fourier_space_ind

#         Generates the fourier space grid.
    
#         """
#         k_ind = np.mgrid[:self.Nsize, :self.Nsize] - int((self.Nsize + 1)/2)
#         k_ind = scipy.fftpack.fftshift(k_ind)
#         self.k_ind = k_ind

#     def PowerSpectrum_grid_generator(self):
#         """PowerSpectrum_grid_generator

#         Generates the powerspectrum grid.
    
#         """
#         # k_c = 0.785
#         # A_n = 1/((k_c)**self.n)
#         A_n = 1
#         k_idx = self.k_ind
#         k = np.sqrt(k_idx[0]**2 + k_idx[1]**2 + 1e-10)
#         PowerSpectrum = (A_n*(abs(k)**(self.n)))
#         PowerSpectrum[0, 0] = 0
#         self.PowerSpectrum = PowerSpectrum  # PowerSpectrum

#     def gen_correlation(self):
#         """gen_correlation

#         Generates the correlation matrices in fourier and spatial spcae.
    
#         """
#         length = self.PowerSpectrum.shape[0]
#         diagonal = np.reshape(self.PowerSpectrum, (length*length,))
#         corr_f = np.diag(diagonal)  # covariance in fourier space
#         corr_s = np.fft.ifft2(corr_f).real  # covariance in spatial space
#         self.corr_f = corr_f
#         self.corr_s = corr_s

#     def Gen_GRF(self, type = 'grid'):
#         """GenerateBettiP

#         Generates the Gaussian Random field with the specified paramters.
    
#         Args:
#             type (str): Takes either 'grid' or 'array' in string format
    
#         Returns:
#         Numpy array: Gaussian Random field
#         """
#         WhiteNoise = np.random.normal(size=(self.Nsize, self.Nsize)) + 1j*(np.random.normal(size=(self.Nsize, self.Nsize)))
#         Gaussian_field = np.fft.ifftn(WhiteNoise * np.sqrt(self.PowerSpectrum)).real
#         Gaussian_field = Gaussian_field - np.mean(Gaussian_field)
#         Gaussian_field = Gaussian_field/(np.std(Gaussian_field))
#         if type == 'grid':
#             return Gaussian_field
#         elif type == 'array':
#             array = np.reshape(Gaussian_field, (self.Nsize*self.Nsize,))
#             return array
#         else:
#             print('wrong input')
#             return None



class GaussianRandomField:
    """GaussianRandomField

    The class for making Gaussian random field with specified spectral index and size of grid.
        
    Attributes:
        Nzise (int): size of the grid.
        n (int): Spectral index of the power law used to generate the Gaussian Random Field.
        k_ind (array): Grid in the fourier space.
        PowerSpectrum (array): The power spectrum grid made using the spectral index used to make the Gaussian Random Field.
        corr_s (array): Correlation matrix in the fourier space.
        corr_f (array): Correlation matrix in the spatial space.

    """
    def __init__(self,Nsize, n):
        self.Nsize = Nsize
        self.n = n
        self.gen_correlation()

    def fourier_space_ind(self):
        """fourier_space_ind

        Generates the fourier space grid.
    
        """
        k_ind = np.mgrid[:self.Nsize, :self.Nsize] - int( (self.Nsize + 1)/2 )
        k_ind = scipy.fftpack.fftshift(k_ind)
        return( k_ind )

    def gen_correlation(self):
        k_idx = self.fourier_space_ind()

            # Defines the amplitude as a power law 1/|k|^(alpha/2)
        amplitude = np.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, self.n/2 )
        amplitude[0,0] = 0
        diagonal = np.reshape(amplitude, (self.Nsize*self.Nsize,))
        corr_f = np.diag(diagonal)  # covariance in fourier space
        corr_s = np.fft.ifft2(corr_f).real  # covariance in spatial space
        self.corr_f = corr_f
        self.corr_s = corr_s
        self.amplitude = amplitude

    def Gen_GRF(self,type = 'grid'):
        """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
            
            Input args:
                alpha (double, default = 3.0): 
                    The power of the power-law momentum distribution
                size (integer, default = 128):
                    The size of the square output Gaussian Random Fields
                flag_normalize (boolean, default = True):
                    Normalizes the Gaussian Field:
                        - to have an average of 0.0
                        - to have a standard deviation of 1.0
            Returns:
                gfield (numpy array of shape (size, size)):
                    The random gaussian random field
                    
            Example:
            import matplotlib
            import matplotlib.pyplot as plt
            example = gaussian_random_field()
            plt.imshow(example)
            """
            
            # Defines momentum indices
        
            # Draws a complex gaussian random noise with normal
            # (circular) distribution
        noise = np.random.normal(size = (self.Nsize, self.Nsize)) + 1j * np.random.normal(size = (self.Nsize, self.Nsize))
        
            # To real space
        gfield = np.fft.ifft2(noise * self.amplitude).real
        
            # Sets the standard deviation to one
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
            
        if type == 'grid':
            return gfield
        elif type == 'array':
            array = np.reshape(gfield, (self.Nsize*self.Nsize,))
            return array
        else:
            print('wrong input')
            return None
