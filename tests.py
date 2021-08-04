from numpy.lib import average
from numpy.linalg.linalg import det
import utilities
import numpy as np
import topologicalFunc
import rocGen
from gaussClass import GaussianRandomField
import matplotlib.pyplot as plt

#TODO:write tests
nsize = 20
power0 = 0.9
power1 = 1
niter =  1000


def testLikelihoodROC(nsize,power_null,power_test,num_iter):
    lik0,lik1 = utilities.Generate_Likelihood_Array(nsize,power_null,power_test,num_iter)
    PFA,PD = rocGen.LikelihoodROC(lik0,lik1,10)
    utilities.plotROC(PFA,PD,nsize,niter,power0,power1,'likelihood')

def testBettiROC(Nsize,power_null,power_test,average,num_iter,filtration_threshold_start,filtration_threshold_stop):
    [Betti0,Betti1,Genus0,Genus1] = utilities.Generate_BettiGenus_array(Nsize,power_null,power_test,average,num_iter,filtration_threshold_start,filtration_threshold_stop) #Generates Betti and Genus arrays

    [PFA_betti0,PD_betti0] = rocGen.BettiROC(Betti0[:,0,:],Betti1[:,0,:],0.01) #Generates the Betti ROC curve for betti0
    [PFA_betti1,PD_betti1] = rocGen.BettiROC(Betti0[:,1,:],Betti1[:,1,:],0.01) #Generates the Betti ROC curve for betti1
    [PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1,0.01)                 #Generates the Genus ROC curve

    utilities.plotROC(PFA_betti0,PD_betti0,Nsize,num_iter,power_null,power_test,'betti',0)        #Plots the Betti0 ROC curve
    utilities.plotROC(PFA_betti1,PD_betti1,Nsize,num_iter,power_null,power_test,'betti',1)        #Plots the Betti1 ROC curve
    utilities.plotROC(PFA_Genus,PD_Genus,Nsize,num_iter,power_null,power_test,'genus')            #Plots the Genus ROC curve

    utilities.saveROC(PFA_betti0,PD_betti0,Nsize,num_iter,power_null,power_test,'betti',0)        #Saves the Betti0 ROC curve
    utilities.saveROC(PFA_betti1,PD_betti1,Nsize,num_iter,power_null,power_test,'betti',1)        #Saves the Betti1 ROC curve
    utilities.saveROC(PFA_Genus,PD_Genus,Nsize,num_iter,power_null,power_test,'Genus')            #Saves the Genus ROC curve

def testGaussianRandomField(Gauss_class_object):
    GRF = Gauss_class_object.Gen_GRF('array')
    nsize = Gauss_class_object.Nsize
    power = Gauss_class_object.n
    plt.imshow(GRF)
    plt.savefig('Figures/test/GaussianRandomField_Size{nsize}_Power{power}.png'.format(nize=nsize,power=power))

def testCorrelationMatrix(Gauss_class_object):
    nsize = Gauss_class_object.Nsize
    corr_s = Gauss_class_object.corr_s
    print('Correlation matrix'+'\n',corr_s,'\n')
    maxc = np.amax(corr_s)
    minc = np.amin(corr_s)
    average = (maxc+minc)/2
    corr_tilda = corr_s/average
    det_corr_tilda = np.log(np.longfloat(np.sqrt(abs(np.linalg.det(corr_tilda))))) + 0.5*nsize*nsize*np.log(average)
    print('Log Determinant of correlation matrix :',det_corr_tilda)
    
def testBettiGenus(Betti,Genus,nsize,power):
    plt.plot(Betti)
    plt.plot(Genus)
    plt.savefig('Figures/test/BettiGenus_Size{nsize}_Power{power}.png'.format(nize=nsize,power=power))

# a,b = utilities.readROC(20,500,0.5,1,'likelihood')
# c,d = utilities.readROC(20,500,0.9,1,'likelihood')
# g,h = utilities.readROC(20,500,0,1,'likelihood')
# k,l = utilities.readROC(20,500,-1,1,'likelihood')
# m,n = utilities.readROC(20,500,-3,1,'likelihood')
# p,q = utilities.readROC(20,500,3,1,'likelihood')


a,b = utilities.readROC(20,500,-2.5,-2,'likelihood')
c,d = utilities.readROC(20,500,-3.5,-2,'likelihood')
g,h = utilities.readROC(20,500,0.9,-2,'likelihood')
k,l = utilities.readROC(20,500,0.5,-2,'likelihood')
m,n = utilities.readROC(20,500,-3,-2,'likelihood')
p,q = utilities.readROC(20,500,3,-2,'likelihood')

plt.plot(c,d,label = '-3.5,-2')
plt.plot(m,n,label ='-3,-2')
plt.plot(a,b,label = '-2.5,-2')
plt.plot(k,l,label ='0.5,-2')
plt.plot(g,h,label = '0.9,-2')
plt.plot(p,q,label ='3,-2')
plt.legend()
plt.show()