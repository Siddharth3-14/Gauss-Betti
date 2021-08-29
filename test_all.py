from matplotlib.figure import Figure
import mainFunc
import matplotlib.pyplot as plt
import numpy as np
from gaussClass import GaussianRandomField
import utilities


print('Starting . . .')
#test case 1
nsize = 20
niter = 400
average = 30
    
# test case 1
print('Starting test case 1')
power0 = 1
power1 = 1.5
mainFunc.AllROC(nsize,power0,power1,average,niter)
print('Finished test case 1')

print('\n')

# #test case 2
# print('Starting test case 2')
# power0 = -1
# power1 = 1
# mainFunc.AllROC(nsize,power0,power1,average,niter)
# print('Finished test case 2')

# print('\n')

# #test case 3
# print('Starting test case 3')
# power0 = -0.5
# power1 = 1
# mainFunc.AllROC(nsize,power0,power1,average,niter)
# print('Finished test case 3')

# print('\n')

# print('Starting test case 4')
# power0 = 0
# power1 = 1
# mainFunc.AllROC(nsize,power0,power1,average,niter)
# print('Finished test case 4')

# print('Starting test case 5')
# power0 = 0.5
# power1 = 1
# mainFunc.AllROC(nsize,power0,power1,average,niter)
# print('Finished test case 5')

# print('\n')

# print('Starting test case 5')
# power0 = 1
# power1 = 1
# mainFunc.AllROC(nsize,power0,power1,average,niter)
# print('Finished test case 5')

# print('\n')

# print('Starting test case 5')
# power0 = 1.5
# power1 = 1
# mainFunc.AllROC(nsize,power0,power1,average,niter)
# print('Finished test case 5')

# print('\n')

# print('. . . Finished')