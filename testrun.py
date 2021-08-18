from matplotlib.figure import Figure
import mainFunc
import matplotlib.pyplot as plt
import numpy as np
from gaussClass import GaussianRandomField
import utilities

nsize = 20
power0 = 1.1
power1 = 1
niter =  200
average = 5


mainFunc.testBettiGenusROC(nsize,power0,power1,average,niter)



