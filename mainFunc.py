import pandas as pd
import utilities
import numpy as np
import rocGen
from gaussClass import GaussianRandomField
import matplotlib.pyplot as plt

def AllROC(Nsize,power_null,power_test,average,num_iter):
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
    print('Starting AllROC . . .')
    [likelihoodratio0,likelihoodratio1] = utilities.Generate_Likelihood_Array(Nsize,power_null,power_test,num_iter) 
    [Betti0,Betti1,Genus0,Genus1,thresholds] = utilities.Generate_BettiGenus_array(Nsize,power_null,power_test,average,num_iter) 

    PFA_likelihood,PD_likelihood = rocGen.LikelihoodROC(likelihoodratio0,likelihoodratio1)  
    [PFA_betti0,PD_betti0] = rocGen.BettiROC(Betti0[:,0,:],Betti1[:,0,:]) 
    [PFA_betti1,PD_betti1] = rocGen.BettiROC(Betti0[:,1,:],Betti1[:,1,:]) 
    [PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1)               

    tempGRF0 = GaussianRandomField(Nsize,power_null).Gen_GRF('grid')
    tempGRF1 = GaussianRandomField(Nsize,power_test).Gen_GRF('grid')
    diagnol = np.arange(0,1.1,0.1)
    plt.figure()
    GRF1 = plt.subplot2grid((4,6),(0,0),colspan=2,rowspan=2)
    GRF2 = plt.subplot2grid((4,6),(0,2),colspan=2,rowspan=2)
    lroc = plt.subplot2grid((4,6),(0,4),colspan=2,rowspan=2)
    betti0 =plt.subplot2grid((4,6),(2,0),colspan=4,rowspan=2)
    b0_roc = plt.subplot2grid((4,6),(2,4),colspan=2,rowspan=2)

    GRF1.imshow(tempGRF0)
    GRF1.set_title('Gaussian Random Field of Null Hypotheis')
    GRF2.imshow(tempGRF1)
    GRF2.set_title('Gaussian Random Field of test Hypotheis')
    lroc.plot(PFA_likelihood,PD_likelihood,label='ROC curve')
    lroc.plot(diagnol,diagnol,label='x = y')
    lroc.set_title('likelihood ROC')
    lroc.legend()
    betti0.plot(thresholds,Betti0[0,0,:],label='Null power index {null}'.format(null=str(power_null)))
    betti0.plot(thresholds,Betti1[0,0,:],label='Test power index {test}'.format(test=str(power_test)))
    betti0.set_title('Betti0 curve')
    betti0.legend()
    b0_roc.plot(PFA_betti0,PD_betti0,label='ROC curve')
    b0_roc.plot(diagnol,diagnol,label='x = y')
    b0_roc.set_title('Betti0 ROC')
    b0_roc.legend()    
    plt.tight_layout()

    plt.figure()
    betti1 = plt.subplot2grid((4,6),(0,0),colspan=4,rowspan=2)
    b1_roc = plt.subplot2grid((4,6),(0,4),colspan=2,rowspan=2)
    genus = plt.subplot2grid((4,6),(2,0),colspan=4,rowspan=2)
    groc =plt.subplot2grid((4,6),(2,4),colspan=2,rowspan=2)

    betti1.plot(thresholds,Betti0[0,1,:],label='Null power index {null}'.format(null=str(power_null)))
    betti1.plot(thresholds,Betti1[0,1,:],label='Test power index {test}'.format(test=str(power_test))) 
    betti1.set_title('Betti1 curve')
    betti1.legend()
    b1_roc.plot(PFA_betti1,PD_betti1,label='ROC curve')
    b1_roc.plot(diagnol,diagnol,label='x = y')
    b1_roc.set_title('Betti1 ROC')
    b1_roc.legend()
    genus.plot(thresholds,Genus0[0],label='Null power index {null}'.format(null=str(power_null)))
    genus.plot(thresholds,Genus1[0],label='Test power index {test}'.format(test=str(power_test)))
    genus.set_title('Genus curve')
    genus.legend()
    groc.plot(PFA_Genus,PD_Genus,label = 'ROC curve')
    groc.plot(diagnol,diagnol,label = 'X = y')
    groc.set_title('Genus ROC')
    groc.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(PFA_likelihood,PD_likelihood,'r',label='likelihood ROC')
    plt.plot(PFA_betti0,PD_betti0,'y',label='Betti0 ROC')
    plt.plot(PFA_betti1,PD_betti1,'b',label = 'Betti1 ROC')
    plt.plot(PFA_Genus,PD_Genus,'g',label = 'Genus ROC')
    plt.plot(diagnol,diagnol,label='X = y')
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel('PFA')
    plt.ylabel('PD')
    plt.title('ROC curves for values of null power {nullpower} and test power {testpower} {linebreak} spectral index and grid size {nsize} {linebreak} iteration {num_iter}'.format(nsize=str(Nsize),nullpower=str(power_null),testpower=str(power_test),num_iter=str(num_iter),linebreak = '\n'))
    plt.legend()
    print('. . . AllROC finished')
    plt.show()


def MultiAllROC(Nsize,power_null,power_test,average,num_iter):
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
    print('Started MultiAllROC . . .')
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
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('PFA')
        plt.ylabel('PD')
        plt.legend()
        plt.savefig('Figures/test/type{type1}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.png'.format(type1='all',nsize=str(Nsize),num_iter=str(num_iter),H0 = str(power_null[i]),H1 =str(power_test[i])))
    print('. . . MultiAllROC finished')

def MultitestLikelihoodROC(nsize,power_null,power_test,num_iter):
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
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('PFA')
    plt.ylabel('PD')
    plt.title('Likelihood ROC curves for different values of null and test power {linebreak} spectral index and grid size {nsize} {linebreak} iteration {num_iter}'.format(nsize=str(nsize),num_iter=num_iter,linebreak = '\n'))
    plt.legend()
    print('. . . Finished the likelihood testrun')
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
    diagnol = np.arange(0,1,0.1)


    [Betti0,Betti1,Genus0,Genus1,thresholds] = utilities.Generate_BettiGenus_array(Nsize,power_null,power_test,average,num_iter) 

    [PFA_betti0,PD_betti0] = rocGen.BettiROC(Betti0[:,0,:],Betti1[:,0,:]) 
    [PFA_betti1,PD_betti1] = rocGen.BettiROC(Betti0[:,1,:],Betti1[:,1,:]) 
    [PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1)

    fig1 = plt.figure()
    ax11 = fig1.add_subplot(311)
    ax12 = fig1.add_subplot(312)
    ax13 = fig1.add_subplot(313)
    ax11.plot(thresholds,Betti0[5,0,:],label= 'power index = {null}'.format(null=power_null))
    ax11.plot(thresholds,Betti1[5,0,:],label ='power index = {test}'.format(test=power_test))
    ax12.plot(thresholds,Betti0[5,1,:],label= 'power index = {null}'.format(null=power_null))
    ax12.plot(thresholds,Betti1[5,1,:],label ='power index = {test}'.format(test=power_test))
    ax13.plot(thresholds,Genus0[5,:],label= 'power index = {null}'.format(null=power_null))
    ax13.plot(thresholds,Genus1[5,:],label ='power index = {test}'.format(test=power_test))
    ax11.title.set_text('Betti0 ROC')
    ax12.title.set_text('Betti1 ROC')
    ax13.title.set_text('Genus ROC')
    ax11.legend()
    ax12.legend()
    ax13.legend()
    fig1.tight_layout()

    fig2 = plt.figure()
    ax21 = fig2.add_subplot(131)
    ax22 = fig2.add_subplot(132)
    ax23 = fig2.add_subplot(133)
    ax21.plot(PFA_betti0,PD_betti0,label= 'Null = {null}, Test = {test}'.format(null=power_null,test=power_test))
    ax21.plot(diagnol,diagnol,label='x = y')
    ax22.plot(PFA_betti1,PD_betti1,label= 'Null = {null}, Test = {test}'.format(null=power_null,test=power_test))
    ax22.plot(diagnol,diagnol,label='x = y')
    ax23.plot(PFA_Genus,PD_Genus,label= 'Null = {null}, Test = {test}'.format(null=power_null,test=power_test))
    ax23.plot(diagnol,diagnol,label='x = y')

    ax21.title.set_text('Betti0 ROC')
    ax22.title.set_text('Betti1 ROC')
    ax23.title.set_text('Genus ROC')
    fig2.legend()
    ax21.legend()
    ax22.legend()
    ax23.legend()
    fig2.tight_layout()
    print('. . . Finished the test Betti Genus ROC ')

    plt.show()


def MultitestBettiGenusROC(Nsize,power_null,power_test,average,num_iter):
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
    print('. . . Finished Multi test Betti and Genus ROC')
    plt.show()


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

