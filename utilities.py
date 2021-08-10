import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
from gaussClass import GaussianRandomField
import topologicalFunc
import math
import pandas as pd

def KLdivergence(x,y1,y2):
    """KLdivergence

    Calculates the KL divergence for 2 different Gaussian Random Field.
    
    Args:
        x (array): 
        y1 (array): Gausian Random Field of null hypothesis as a 1-D array
        y2 (array): Gaussian Random Field of test hypothesis as a 1-D array 
 
               
    Returns:
       float: KL divergence
    """

    f1, edges = np.histogram(y1, density=True, bins=x)
    binWidth1 = edges[1] - edges[0]
    f2, edges = np.histogram(y2, density=True, bins=x)
    binWidth2 = edges[1] - edges[0]
    sumkl = 0
    for i in range(0, len(f1)):
        if (f1[i] != 0 and f2[i] != 0):
            sumkl = sumkl+((f1*binWidth1)
                        [i]*np.log((f1*binWidth1)[i]/(f2*binWidth2)[i]))
    print("KL_fixed = ", sumkl)
    return sumkl




def Generate_Likelihood_Array(Nsize,power_index_null,power_index_test,iteration):
    """Generate_Likelihood_Array

    Generates the array of likelihood ratios for making ROC curves.
    
    Args:
        Nsize (integer): grid size of the Gaussian Random Field 
        power_index_null (float): Power spectral index of Null Hypothesis
        power_index_test (float): Power spectral index of Test Hypothesis
        iteration (integer): Size of the likelihood ratio array generated 
 
               
    Returns:
       numpy array: array of likelihood ratios
    """
    def Calcualte_log_det(corr):
        maxc = np.amax(corr)
        minc = np.amin(corr)
        average = (maxc + minc)/2
        corr_tilda = corr/average

        det_corr_tilda = np.log(np.longfloat(np.sqrt(abs(np.linalg.det(corr_tilda))))) + 0.5*Nsize*Nsize*np.log(average)
        return det_corr_tilda

    def likelihoodratio(X0,X1):
        trans_X0 = np.transpose(X0)
        trans_X1 = np.transpose(X1)

        type_test_n0 = -0.5*np.dot(trans_X1, np.dot(inv_corr0, X1))
        type_test_n1 = -0.5*np.dot(trans_X1, np.dot(inv_corr1, X1))

        type_null_n0 = -0.5*np.dot(trans_X0, np.dot(inv_corr0, X0))
        type_null_n1 = -0.5*np.dot(trans_X0, np.dot(inv_corr1, X0))


        type_null = type_null_n0 + det_corr1 - type_null_n1 - det_corr0
        type_test = type_test_n1 + det_corr0 - type_test_n0 - det_corr1
       
        return type_null,type_test
        
    Gaussian0 = GaussianRandomField(Nsize,power_index_null)
    Gaussian1 = GaussianRandomField(Nsize,power_index_test)

    corr0 = Gaussian0.corr_s
    corr1 = Gaussian1.corr_s

    det_corr0 = Calcualte_log_det(corr0)
    det_corr1 = Calcualte_log_det(corr1)

    inv_corr0 = np.linalg.inv(corr0)
    inv_corr1 = np.linalg.inv(corr1)
    
    likelihoodratio0 = []
    likelihoodratio1 = []
    for _ in range(iteration):
        tempGaussian0 = Gaussian0.Gen_GRF(type = 'array')
        tempGaussian1 = Gaussian1.Gen_GRF(type = 'array')
        tempType_null,tempType_test = likelihoodratio(tempGaussian0,tempGaussian1)
        likelihoodratio0.append(tempType_null)
        likelihoodratio1.append(tempType_test)

    print('Finished generating the likelihood arrays')
    return np.array([likelihoodratio0,likelihoodratio1])

def Generate_BettiGenus_array(Nsize,power_index_null,power_index_test,average,iteration,filtration_threshold_start=-4,filtration_threshold_stop=4,type1='lower'):
    """Generate_Likelihood_Array

    Generates the Betti and Genus curves for specified parameters.
    
    Args:
        Nsize (integer): grid size of the Gaussian Random Field 
        power_index_null (float): Power spectral index of Null Hypothesis
        power_index_test (float): Power spectral index of Test Hypothesis
        average (integer): No. of times the betti curves need to be averaged
        iteration (integer): Size of the arrays generated
        filtration_threshold_start (float): Start value for generating filtraion from dionysus
        filtration_threshold_stop (float): Stop value for generation filtration from dionysus
        type : Type of filtration accepted values are 'lower' 'upper 
 
               
    Returns:
       numpy array: array of Betti and Genus curves
    """
    Gaussian0 = GaussianRandomField(Nsize,power_index_null)
    Gaussian1 = GaussianRandomField(Nsize,power_index_test)
    size = int((filtration_threshold_stop-filtration_threshold_start)/0.01)
    Betti_array0 = []
    Betti_array1 = []
    Genus_array0 = []
    Genus_array1 = []

    for _ in range(iteration):
        BettiAVG0 = np.zeros((3,size))
        BettiAVG1 = np.zeros((3,size))
        for _ in range(average):
            temp_filtration0 = topologicalFunc.GaussianFiltration(Gaussian0.Gen_GRF())
            temp_filtration1 = topologicalFunc.GaussianFiltration(Gaussian1.Gen_GRF())
            temp_betti0 = topologicalFunc.GenerateBettiP(temp_filtration0,filtration_threshold_start,filtration_threshold_stop,type1)
            temp_betti1 = topologicalFunc.GenerateBettiP(temp_filtration1,filtration_threshold_start,filtration_threshold_stop,type1)
            BettiAVG0 += temp_betti0
            BettiAVG1 += temp_betti1
        BettiAVG0 = BettiAVG0/average
        BettiAVG1 = BettiAVG1/average
        GenusAVG0 = topologicalFunc.GenerateGenus(BettiAVG0)
        GenusAVG1 = topologicalFunc.GenerateGenus(BettiAVG1)
    
        Genus_array0.append(GenusAVG0)
        Genus_array1.append(GenusAVG1)
        Betti_array0.append(BettiAVG0)
        Betti_array1.append(BettiAVG1)
    print('Finished generating Betti and Genus arrays')
    return [np.array(Betti_array0),np.array(Betti_array1),np.array(Genus_array0),np.array(Genus_array1)]


def plotROC(PFA,PD,nsize,num_iter,H0,H1,type1,Betti='default'):
    """plotROC

    Plots the PFA and PD ROC graph with the labels provided through parameters.
    
    Args:
        PFA (array): numpy vector. The PFA array generated during ROC gen.
        PD (array): numpy vector. THe PD array generated during ROC gen.
        nsize (integer): Size of the Gaussian Random Fields grid.
        num_iter (integer): Number of iteration for which ROC gen is run.
        H0 (float): Power spectral index of Null Hypothesis.
        H1 (float): Power spectral index of Test Hypothesis.
        type1 (string): type1 of the ROC curve generated takes value 'likelihood','betti','genus
        Betti (integer): Dimension of Betti curve not needed when type = likelihood

        
    Returns:
        None: None
    """
    if type1 =='betti':
        plt.plot(PFA,PD)
        plt.xlim(-.1,1.1)
        plt.ylim(-.1,1.1)
        plt.xlabel('PFA')
        plt.ylabel('PD')
        title = '{type1} ROC Betti{Betti} {linebreak} Grid size = {nsize} iteration = {num_iter} {linebreak} power spectral index of null hypothesis:{H0}, test hypothesis:{H1} '.format(type1 = type1,Betti=str(Betti),nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1),linebreak='\n' )
        plt.title(title)
        # plt.show()
        print('Finished saving the {type1} plot'.format(type1=type1))
        plt.savefig('Figures/{type1}B{Betti}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.png'.format(type1=type1,Betti=str(Betti),nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1)))


    else:
        plt.figure()
        plt.plot(PFA,PD)
        plt.xlim(-.1,1.1)
        plt.ylim(-.1,1.1)
        plt.xlabel('PFA')
        plt.ylabel('PD')
        title = '{type1} ROC {linebreak} Grid size = {nsize} iteration = {num_iter} {linebreak} power spectral index of null hypothesis:{H0}, test hypothesis:{H1} '.format(type1 = type1,nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1),linebreak='\n' )
        plt.title(title)
        # plt.show()
        plt.savefig('Figures/{type1}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.png'.format(type1=type1,nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1)))
        print('Finished saving the {type1} plot'.format(type1=type1))


def saveROC(PFA,PD,nsize,num_iter,H0,H1,type1,Betti='default'):
    """SaveROC

    Saves the PFA and PD array with the labels provided through parameters.
    
    Args:
        PFA (array): numpy vector. The PFA array generated during ROC gen.
        PD (array): numpy vector. THe PD array generated during ROC gen.
        nsize (integer): Size of the Gaussian Random Fields grid.
        num_iter (integer): Number of iteration for which ROC gen is run.
        H0 (float): Power spectral index of Null Hypothesis.
        H1 (float): Power spectral index of Test Hypothesis.
        type1 (string): type1 of the ROC curve generated takes value 'likelihood','betti','genus
        Betti (integer): Dimension of Betti curve not needed when type = likelihood
    Returns:
       None: None
    """
    length = PFA.shape[0]
    if type1 == 'betti':
        file = open('Files/{type1}B{Betti}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.txt'.format(type1=type1,Betti=str(Betti),nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1)),'w+')
        file.write('PFA' + '\t' + 'PD' +'\n')
        for i in range(length):
            file.write(str(PFA[i]) + '\t' + str(PD[i])+'\n')
        file.close()
        print('Finished saving the {type1} PFA and PD arrays'.format(type1=type1))

    else:
        file = open('Files/{type1}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.txt'.format(type1=type1,nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1)),'w+')
        file.write('PFA' + '\t' + 'PD' +'\n')
        for i in range(length):
            file.write(str(PFA[i]) + '\t' + str(PD[i])+'\n')
        file.close()
        print('Finished saving the {type1} PFA and PD arrays'.format(type1=type1))




def readROC(nsize,num_iter,H0,H1,type1,Betti='default'):
    """readROC

    Reads the PFA and PD array from the files generated using saveROC.
    
    Args:
        nsize (integer): Size of the Gaussian Random Fields grid.
        num_iter (integer): Number of iteration for which ROC gen is run.
        H0 (integer): Power spectral index of Null Hypothesis.
        H1 (integer): Power spectral index of Test Hypothesis.
        type1 (string): type1 of the ROC curve generated takes value 'likelihood','betti','genus
        Betti (integer): Dimension of Betti curve not needed when type = likelihood

    Returns:
       Numpy Array: Returns PFA and PD arrays 
    """

    if type1 == 'betti':
        filename = 'Files/{type1}B{Betti}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.txt'.format(type1=type1,Betti=str(Betti),nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1))
        Data = pd.read_csv(filename,delimiter='\s+')
        PFA=Data['PFA']
        PD=Data['PD']
        print('Finished reading the PFA and PD arrays')

        return np.array([PFA,PD])

        # PFA = Data[0]
        # PD = Data[1]
        # return [PFA,PD]
    else:
        filename = 'Files/{type1}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.txt'.format(type1=type1,nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1))
        Data = pd.read_csv(filename,delimiter='\s+')
        PFA=Data['PFA']
        PD=Data['PD']
        print('Finished reading the PFA and PD arrays')
        return np.array([PFA,PD])

