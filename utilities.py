import numpy as np
import matplotlib.pyplot as plt

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

def likelihoodratio(correlation0,correlation1,X0,X1):
    """likelihoodratio

    Calculates the likelihood ratio for the False alarm and detection.
    
    Args:
        correlation0 (array): Correlation matrix for the Gaussian Random Field for null hypothesis.
        correlation1 (array): Correlation matrix for the Gaussian Random Field for test hypothesis.
        X0 (array): Gausian Random Field of null hypothesis as a 1-D array
        X1 (array): Gaussian Random Field of test hypothesis as a 1-D array 
       
        
    Returns:
       float: likelihood ratio
    """


    X0 = X0 - np.mean(X0)
    det_corr0 = np.sqrt(abs(np.linalg.det(correlation0)))
    inv_corr0 = np.linalg.inv(correlation0)
    Trans_X0 = np.transpose(X0)
    

    X1 = X1 - np.mean(X1)
    det_corr1 = np.sqrt(abs(np.linalg.det(correlation1)))
    inv_corr1 = np.linalg.inv(correlation1)
    Trans_X1 = np.transpose(X1)



    return None

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
