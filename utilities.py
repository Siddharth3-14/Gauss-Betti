import numpy as np
import matplotlib.pyplot as plt

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
