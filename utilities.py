import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
from gaussClass import GaussianRandomField
import topologicalFunc
import math

#TODO: solve the infinity problem
#TODO: complete the likelihood ratio function
#TODO: finish documentation for the KLdivergence

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


def Generate_Likelihood_Array(Nsize,n0,n1,iteration):
    Gaussian0 = GaussianRandomField(Nsize,n0)
    Gaussian1 = GaussianRandomField(Nsize,n1)

    corr0 = Gaussian0.corr_s
    corr1 = Gaussian1.corr_s

    inv_corr0 = np.linalg.inv(corr0)
    inv_corr1 = np.linalg.inv(corr1)

    det_corr0 = np.longfloat(np.sqrt(abs(np.linalg.det(corr0))))
    det_corr1 = np.longfloat(np.sqrt(abs(np.linalg.det(corr1))))
    print(det_corr1,det_corr0)
    x = np.log(det_corr0)
    y = np.log(det_corr1)
    diff = (y - x)
    print(diff)

    det_corr0 = np.longfloat(np.sqrt(  (abs(np.linalg.det(corr0)))**(1/(Nsize*Nsize))  ))
    det_corr1 = np.longfloat(np.sqrt(  (abs(np.linalg.det(corr1)))**(1/(Nsize*Nsize))  ))
    print(det_corr1,det_corr0)
    x = np.log(det_corr0)
    y = np.log(det_corr1)
    diff = Nsize*Nsize*(y - x)

    print(diff)
    return None


    def likelihoodratio(X0,X1):
        trans_X0 = np.transpose(X0)
        trans_X1 = np.transpose(X1)
        type1_n0 = -0.5*np.dot(trans_X1, np.dot(inv_corr0, X1))
        type1_n1 = -0.5*np.dot(trans_X1, np.dot(inv_corr1, X1))
        p0 = np.exp(type1_n0)
        p1 = np.exp(type1_n1)

        type1 = diff + np.log(p1) - np.log(p0)

        type2_n0 = -0.5*np.dot(trans_X0, np.dot(inv_corr0, X0))
        type2_n1 = -0.5*np.dot(trans_X0, np.dot(inv_corr1, X0))
        q0 = np.exp(type2_n0)
        q1 = np.exp(type2_n1)

        type2 = -diff + np.log(q0) - np.log(q1)
        return type1,type2

    likelihoodratio0 = []
    likelihoodratio1 = []
    for _ in range(iteration):
        tempGaussian0 = Gaussian0.Gen_GRF(type = 'array')
        tempGaussian1 = Gaussian1.Gen_GRF(type = 'array')
        tempType0,tempType1 = likelihoodratio(tempGaussian0,tempGaussian1)
        likelihoodratio0.append(tempType0)
        likelihoodratio1.append(tempType1)

    return likelihoodratio0,likelihoodratio1

def Generate_BettiGenus_array(Nsize,n0,n1,average,iteration,thresholds_start,thresholds_stop,type='lower'):
    Gaussian0 = GaussianRandomField(Nsize,n0)
    Gaussian1 = GaussianRandomField(Nsize,n1)
    size = int((thresholds_stop-thresholds_start)/0.01)
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
            temp_betti0 = topologicalFunc.GenerateBettiP(temp_filtration0,thresholds_start,thresholds_stop,type)
            temp_betti1 = topologicalFunc.GenerateBettiP(temp_filtration1,thresholds_start,thresholds_stop,type)
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

    return [np.array(Betti_array1),np.array(Betti_array0),np.array(Genus_array0),np.array(Genus_array1)]

def plotROC(PFA,PD,nsize,num_iter,H0,H1,Betti,type):
    """plotROC

    Plots the PFA and PD ROC graph with the labels provided through parameters.
    
    Args:
        PFA (array): numpy vector. The PFA array generated during ROC gen.
        PD (array): numpy vector. THe PD array generated during ROC gen.
        nsize (integer): Size of the Gaussian Random Fields grid.
        num_iter (integer): Number of iteration for which ROC gen is run.
        H0 (integer): Power spectral index of Null Hypothesis.
        H1 (integer): Power spectral index of Test Hypothesis.
        type (string): type of the ROC curve generated
        
    Returns:
        None: None
    """
    plt.plot(PFA,PD)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('PFA')
    plt.ylabel('PD')
    title = '{type} ROC Betti{Betti} {linebreak} Grid size = {nsize} iteration = {num_iter} {linebreak} power spectral index of null hypothesis:{H0}, test hypothesis:{H1} '.format(type = type,Betti=str(Betti),nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1),linebreak='\n' )
    plt.title(title)
    plt.savefig('Figures/{type}B{Betti}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.png'.format(type=type,Betti=str(Betti),nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1)))

def SaveROC(PFA,PD,nsize,num_iter,H0,H1,Betti,type):
    """SaveROC

    Saves the PFA and PD array with the labels provided through parameters.
    
    Args:
        PFA (array): numpy vector. The PFA array generated during ROC gen.
        PD (array): numpy vector. THe PD array generated during ROC gen.
        nsize (integer): Size of the Gaussian Random Fields grid.
        num_iter (integer): Number of iteration for which ROC gen is run.
        H0 (integer): Power spectral index of Null Hypothesis.
        H1 (integer): Power spectral index of Test Hypothesis.
        type (string): type of the ROC curve generated

    Returns:
       None: None
    """
    length = PFA.shape[0]
    file = open('Files/{type}B{Betti}Nsize{nsize}Iter{num_iter}n{H0}n{H1}.txt'.format(type=type,Betti=str(Betti),nsize=str(nsize),num_iter=str(num_iter),H0 = str(H0),H1 =str(H1)),'w+')
    file.write('PFA' + '\t' + 'PD' +'\n')
    for i in range(length):
        file.write(str(PFA[i]) + '\t' + str(PD[i])+'\n')
    file.close()
