import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.fftpack


class GaussianRandomField:
    def __init__(self, Nsize, n):
        self.Nsize = Nsize
        self.n = n
        self.fourier_space_ind()
        self.PowerSpectrum_grid_generator()
        self.gen_correlation()
        self.Gen_GRF()

    def fourier_space_ind(self):
        k_ind = np.mgrid[:self.Nsize, :self.Nsize] - int((self.Nsize + 1)/2)
        k_ind = scipy.fftpack.fftshift(k_ind)
        self.k_ind = k_ind

    def PowerSpectrum_grid_generator(self):
        # k_c = 0.785
        # A_n = 1/((k_c)**self.n)
        A_n = 1
        k_idx = self.k_ind
        k = np.sqrt(k_idx[0]**2 + k_idx[1]**2 + 1e-10)
        PowerSpectrum = (A_n*(abs(k)**(self.n)))
        PowerSpectrum[0, 0] = 0
        self.PowerSpectrum = PowerSpectrum  # PowerSpectrum

    def gen_correlation(self):
        length = self.PowerSpectrum.shape[0]
        diagonal = np.reshape(self.PowerSpectrum, (length*length,))
        corr_f = np.diag(diagonal)  # covariance in fourier space
        corr_s = np.fft.ifft2(corr_f).real  # covariance in spatial space
        self.corr_f = corr_f
        self.corr_s = corr_s

    def Gen_GRF(self, type = 'grid'):
        WhiteNoise = np.random.normal(size=(self.Nsize, self.Nsize)) + 1j*(np.random.normal(size=(self.Nsize, self.Nsize)))
        Gaussian_field = np.fft.ifftn(WhiteNoise * np.sqrt(self.PowerSpectrum)).real
        Gaussian_field = Gaussian_field - np.mean(Gaussian_field)
        Gaussian_field = Gaussian_field/(np.std(Gaussian_field))
        if type == 'grid':
            return Gaussian_field
        elif type == 'array':
            array = np.reshape(Gaussian_field, (self.Nsize*self.Nsize,))
            return array
        else:
            print('wrong input')
            return None


