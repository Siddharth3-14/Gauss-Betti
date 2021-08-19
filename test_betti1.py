from matplotlib.figure import Figure
import mainFunc
import matplotlib.pyplot as plt
import numpy as np
from gaussClass import GaussianRandomField
import utilities
from matplotlib.figure import Figure
import mainFunc
import matplotlib.pyplot as plt
import numpy as np
from gaussClass import GaussianRandomField
import utilities


nsize = 20
niter =  300
average = 20

print('Starting . . .')
#test case 1
print('started test case 1')
power0_array = [1,1,1,1,1,1,1,1,1,1,1]
power1_array = [-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]
mainFunc.MultitestBettiGenusROC(nsize,power0_array,power1_array,average,niter,1)
print('finished test case 1')

print('\n')

#test case 2
print('started test case 2')
power0_array = [-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]
power1_array = [1,1,1,1,1,1,1,1,1,1,1]
mainFunc.MultitestBettiGenusROC(nsize,power0_array,power1_array,average,niter,2)
print('finished test case 2')

print('\n')

print('. . . finished')








