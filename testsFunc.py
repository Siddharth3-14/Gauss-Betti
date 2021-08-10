import pandas as pd
import utilities
import numpy as np
import rocGen
from gaussClass import GaussianRandomField
import matplotlib.pyplot as plt


def testLikelihoodROC(nsize,power_null,power_test,num_iter):
    """testLikelihoodROC

    Plots the likelihood ROC curve for multiple values of null and test power spectral index.
    
    Args:
        nsize (integer): Size of the Gaussian Random Fields grid.
        power_null (array): Power spectral indices of Null Hypothesis.
        power_test (array): Power spectral indices of Test Hypothesis.
        num_iter (integer): Number of iteration for which ROC gen is run.
       
    Returns:
        None: None
    """
    num_tests = len(power_null)
    for i in range(num_tests):
        lik0,lik1 = utilities.Generate_Likelihood_Array(nsize,power_null[i],power_test[i],num_iter)
        PFA,PD = rocGen.LikelihoodROC(lik0,lik1)
        plt.plot(PFA,PD,label= 'Null = {null}, Test = {test}'.format(null=power_null[i],test=power_test[i]))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('PFA')
    plt.ylabel('PD')
    plt.title('Likelihood ROC curves for different values of null and test power {linebreak} spectral index and grid size {nsize} {linebreak} iteration {num_iter}'.format(nsize=str(nsize),num_iter=num_iter,linebreak = '\n'))
    plt.legend()
    plt.show()
    


def testBettiGenusROC(Nsize,power_null,power_test,average,num_iter):
    """testBettiGenusROC

    Plots the Betti and Genus ROC curve for multiple values of null and test power spectral index.
    
    Args:
        nsize (integer): Size of the Gaussian Random Fields grid.
        power_null (array): Power spectral indices of Null Hypothesis.
        power_test (array): Power spectral indices of Test Hypothesis.
        average (integer): Number of times topological curves are averaged
        num_iter (integer): Number of iteration for which ROC gen is run.
       
    Returns:
        None: None
    """
    num_tests = len(power_null)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for i in range(num_tests):
        [Betti0,Betti1,Genus0,Genus1] = utilities.Generate_BettiGenus_array(Nsize,power_null[i],power_test[i],average,num_iter) 

        [PFA_betti0,PD_betti0] = rocGen.BettiROC(Betti0[:,0,:],Betti1[:,0,:]) 
        [PFA_betti1,PD_betti1] = rocGen.BettiROC(Betti0[:,1,:],Betti1[:,1,:]) 
        [PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1)

        ax1.plot(PFA_betti0,PD_betti0,label= 'Null = {null}, Test = {test}'.format(null=power_null[i],test=power_test[i]))
        ax2.plot(PFA_betti1,PD_betti1,label= 'Null = {null}, Test = {test}'.format(null=power_null[i],test=power_test[i]))
        ax3.plot(PFA_Genus,PD_Genus,label= 'Null = {null}, Test = {test}'.format(null=power_null[i],test=power_test[i]))

    ax1.title.set_text('Betti0 ROC')
    ax2.title.set_text('Betti1 ROC')
    ax3.title.set_text('Genus ROC')
    plt.legend()
    plt.show()


def testAllROC(Nsize,power_null,power_test,average,num_iter):
    """testAllROC

    Plots all ROC curve for single value of null and test power spectral index.
    
    Args:
        nsize (integer): Size of the Gaussian Random Fields grid.
        power_null (float): Power spectral index of Null Hypothesis.
        power_test (float): Power spectral index of Test Hypothesis.
        average (integer): Number of times topological curves are averaged
        num_iter (integer): Number of iteration for which ROC gen is run.
       
    Returns:
        None: None
    """
    [likelihoodratio0,likelihoodratio1] = utilities.Generate_Likelihood_Array(Nsize,power_null,power_test,num_iter) 
    [Betti0,Betti1,Genus0,Genus1] = utilities.Generate_BettiGenus_array(Nsize,power_null,power_test,average,num_iter) 

    PFA_likelihood,PD_likelihood = rocGen.LikelihoodROC(likelihoodratio0,likelihoodratio1)  
    [PFA_betti0,PD_betti0] = rocGen.BettiROC(Betti0[:,0,:],Betti1[:,0,:]) 
    [PFA_betti1,PD_betti1] = rocGen.BettiROC(Betti0[:,1,:],Betti1[:,1,:]) 
    [PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1,0.01)               

    plt.plot(PFA_likelihood,PD_likelihood,'r',label='likelihood ROC')
    plt.plot(PFA_betti0,PD_betti0,'y',label='Betti0 ROC')
    plt.plot(PFA_betti1,PD_betti1,'b',label = 'Betti1 ROC')
    plt.plot(PFA_Genus,PD_Genus,'g',label = 'Genus ROC')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('PFA')
    plt.ylabel('PD')
    plt.title('ROC curves for values of null power {nullpower} and test power {testpower} {linebreak} spectral index and grid size {nsize} {linebreak} iteration {num_iter}'.format(nsize=str(Nsize),nullpower=str(power_null),testpower=str(power_test),num_iter=str(num_iter),linebreak = '\n'))
    plt.legend()
    plt.show()
    plt.savefig('Figures/type{type1}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.png'.format(type1='all',nsize=str(Nsize),num_iter=str(num_iter),H0 = str(power_null),H1 =str(power_test)))

def MultitestAllROC(Nsize,power_null,power_test,average,num_iter):
    """testBettiGenusROC

    Plots all ROC curve for multiple values of null and test power spectral index and saves them in a folder.
    
    Args:
        nsize (integer): Size of the Gaussian Random Fields grid.
        power_null (array): Power spectral indices of Null Hypothesis.
        power_test (array): Power spectral indices of Test Hypothesis.
        average (integer): Number of times topological curves are averaged
        num_iter (integer): Number of iteration for which ROC gen is run.
       
    Returns:
        None: None
    """
    num_tests = len(power_null)
    for i in range(num_tests):
        [likelihoodratio0,likelihoodratio1] = utilities.Generate_Likelihood_Array(Nsize,power_null[i],power_test[i],num_iter) 
        [Betti0,Betti1,Genus0,Genus1] = utilities.Generate_BettiGenus_array(Nsize,power_null[i],power_test[i],average,num_iter) 

        PFA_likelihood,PD_likelihood = rocGen.LikelihoodROC(likelihoodratio0,likelihoodratio1)  
        [PFA_betti0,PD_betti0] = rocGen.BettiROC(Betti0[:,0,:],Betti1[:,0,:]) 
        [PFA_betti1,PD_betti1] = rocGen.BettiROC(Betti0[:,1,:],Betti1[:,1,:]) 
        [PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1,0.01)               

        plt.plot(PFA_likelihood,PD_likelihood,'r',label='likelihood ROC')
        plt.plot(PFA_betti0,PD_betti0,'y',label='Betti0 ROC')
        plt.plot(PFA_betti1,PD_betti1,'b',label = 'Betti1 ROC')
        plt.plot(PFA_Genus,PD_Genus,'g',label = 'Genus ROC')
        plt.title('ROC curves for values of null power {nullpower} and test power {testpower} {linebreak} spectral index and grid size {nsize} {linebreak} iteration {num_iter}'.format(nsize=str(Nsize),nullpower=str(power_null),testpower=str(power_test),num_iter=str(num_iter),linebreak = '\n'))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel('PFA')
        plt.ylabel('PD')
        plt.legend()
        plt.savefig('Figures/test/type{type1}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.png'.format(type1='all',nsize=str(Nsize),num_iter=str(num_iter),H0 = str(power_null[i]),H1 =str(power_test[i])))


def testGaussianRandomField(Gauss_class_object):
    """testGaussianRandomField

    Plots the Gaussian Random field.
    
    Args:
        Gauss_class_object (object): its an instance of GaussianRandomField object
       
    Returns:
        None: None
    """
    GRF = Gauss_class_object.Gen_GRF('array')
    nsize = Gauss_class_object.Nsize
    power = Gauss_class_object.n
    plt.imshow(GRF)
    plt.savefig('Figures/test/GaussianRandomField_Size{nsize}_Power{power}.png'.format(nize=nsize,power=power))



def testCorrelationMatrix(Gauss_class_object):
    """testCorrelationMatrix

    Prints info about the original and modified correlation matrix of for an instance of GaussianRandomField.
    
    Args:
        Gauss_class_object (object): its an instance of GaussianRandomField object
       
    Returns:
        None: None
    """
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
    """testBettiGenus

    Plots the Betti and Genus curves.
    
    Args:
        Betti (array): Betti array
        Genus (array): Genus array

    Returns:
        None: None
    """
    plt.plot(Betti)
    plt.plot(Genus)
    plt.savefig('Figures/test/BettiGenus_Size{nsize}_Power{power}.png'.format(nize=nsize,power=power))
