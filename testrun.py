from matplotlib.figure import Figure
import testsFunc
import matplotlib.pyplot as plt

nsize = 20
power0 = [0.9,0.5]
power1 = [1,1]
niter =  100
average = 5

# testsFunc.testBettiGenusROC(nsize,power0,power1,average,niter)
plt.plot(power0,power1)
plt.savefig('Figure/multi/test.png')