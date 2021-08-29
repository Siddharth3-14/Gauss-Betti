from numpy import matmul
import mainFunc
from gaussClass import GaussianRandomField
import numpy as np
import matplotlib.pyplot as plt

nsize = 20
niter = 100
average = 50
power0 = 1.5
power1 = 1

mainFunc.testLikelihoodROC(nsize,power0,power1,niter,average)
# mainFunc.testBettiGenusROC(nsize,power0,power1,average,niter)
# mainFunc.AllROC(nsize,power0,power1,average,niter)




# def print_diag(mat):
#     for i in range(mat.shape[0]):
#         print(mat[i,i])

# print_diag(Gauss1.corr_f)
# print('\n')
# print_diag(Gauss2.corr_f)
# print('\n')
# print_diag(Gauss3.corr_f)


# Gauss1 = GaussianRandomField(nsize,0)
# Gauss2 = GaussianRandomField(nsize,-0.5)
# Gauss3 = GaussianRandomField(nsize,-1)
# Gauss4 = GaussianRandomField(nsize,-1.5)
# Gauss5 = GaussianRandomField(nsize,-2)
# Gauss6 = GaussianRandomField(nsize,-2.5)

# print('power = 0')
# mainFunc.testCorrelationMatrix(Gauss1)
# # eigen1,_ = np.linalg.eig(Gauss1.corr_s)
# # print(eigen1)
# print('power = -0.5')
# mainFunc.testCorrelationMatrix(Gauss2)
# # eigen2,_ = np.linalg.eig(Gauss2.corr_s)
# # print(eigen2)
# print('power = -1')
# mainFunc.testCorrelationMatrix(Gauss3)
# # eigen3,_ = np.linalg.eig(Gauss3.corr_s)
# # print(eigen3)
# print('power = -1.5')
# mainFunc.testCorrelationMatrix(Gauss4)
# # eigen4,_ = np.linalg.eig(Gauss4.corr_s)
# # print(eigen4)
# print('power = -2')
# mainFunc.testCorrelationMatrix(Gauss5)
# # eigen5,_ = np.linalg.eig(Gauss5.corr_s)
# # print(eigen5)
# print('power = -2.5')
# mainFunc.testCorrelationMatrix(Gauss6)
# # eigen6,_ = np.linalg.eig(Gauss6.corr_s)
# # print(eigen6)


