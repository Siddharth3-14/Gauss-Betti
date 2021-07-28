import numpy as np

def LikelihoodROC(likelihoodratio0,likelihoodratio1,threshold_step):
    """LikelihoodROC

    Generates PFA and PD values from likelihood ratios of the Gaussian Random Field using the specified parameters
    
    Args:
        likelihoodratio0 (array):arrays containing likelihood ratios of the Gaussian Random field of type 1 see doc for likelihoodratio .
        likelihoodratio1: 2-D arrays containing likelihood ratios of the Gaussian Random field of type 2 see doc for likelihoodratio
        threshold_step (float): Step value for generating threshold array.

    Returns:
        Array: PFA and PD values
    """
    peak0 = np.amax(likelihoodratio0)
    peak1 = np.amax(likelihoodratio1)
    trough0 = np.amin(likelihoodratio0)
    trough1 = np.amin(likelihoodratio1)
    threshold_start = np.amin([trough1,trough0])-3*threshold_step
    threshold_stop = np.amax([peak1,peak0])+3*threshold_step
    thresholds = np.arange(threshold_start,threshold_stop,threshold_step)
    iteration = likelihoodratio0.shape[0]
    PFA_array = []
    PD_array = []
    for lambd in thresholds:
        PFA = 0
        PD = 0
        for i in range(iteration):
            if likelihoodratio0[i] > lambd:
                PFA += 1
            if likelihoodratio1[i] > lambd:
                PD += 1

        PFA = PFA/iteration
        PD = PD/iteration
        PFA_array.append(PFA)
        PD_array.append(PD)

    print('Finished generating likelihood ROC curves')
    return np.array([PFA_array,PD_array])


def BettiROC(Betti_array0,Betti_array1,threshold_step):
    """BettiROC

    Generates PFA and PD values from Betti curves using the specified parameters
    
    Args:
        Betti_array0 (array): multiple Betti curves for null hypothesis generated from GenerateBetti function for a one dimension.
        Betti_array1 (array): multiple Betti curves for test hypothesis generated from GenerateBetti function for a one dimension.
        threshold_step (float): Step value for generating threshold array.

    Returns:
        Array: PFA and PD values
    """
    peak0 = np.amax(Betti_array0)
    peak1 = np.amax(Betti_array1)
    index = np.argmax(Betti_array1[0])
    threshold_start = np.amin([peak0,peak1]) - 20
    threshold_stop = np.amax([peak0,peak1]) + 20
    thresholds = np.arange(threshold_start,threshold_stop,threshold_step)
    iteration = Betti_array1.shape[0]
    PFA_array = []
    PD_array = []
    for lambd in thresholds:
        PFA = 0
        PD = 0
        for i in range(iteration):
            if Betti_array1[i,index] > lambd:
                PD += 1
            if Betti_array0[i,index] > lambd:
                PFA += 1
        PFA = PFA/iteration
        PD = PD/iteration
        PFA_array.append(PFA)
        PD_array.append(PD)
    print('Finished generating Betti ROC curves')
    return np.array([PFA_array,PD_array])

def GenusROC(Genus_array0,Genus_array1,threshold_step):
    """GenusROC

    Generates PFA and PD values from Genus curves using the specified parameters
    
    Args:
        Genus_array0 (array): multiple Genus curves for null hypothesis generated from GenerateBetti function for a one dimension.
        Genus_array1 (array): multiple Genus curves for test hypothesis generated from GenerateBetti function for a one dimension.
        index (integer): index at which the Genus value is calculated and compared with threshold.
        threshold_step (float): Step value for generating threshold array.

    Returns:
        Array: PFA and PD values
    """
    peak0 = np.amax(Genus_array0)
    peak1 = np.amax(Genus_array1)
    index = np.argmax(Genus_array1[0])
    threshold_start = np.amin([peak0,peak1]) - 20
    threshold_stop = np.amax([peak0,peak1]) + 20
    thresholds = np.arange(threshold_start,threshold_stop,threshold_step)
    iteration = Genus_array1.shape[0]
    PFA_array = []
    PD_array = []
    for lambd in thresholds:
        PFA = 0
        PD = 0
        for i in range(iteration):
            if Genus_array1[i,index] > lambd:
                PD += 1
            if Genus_array0[i,index] > lambd:
                PFA += 1
        PFA = PFA/iteration
        PD = PD/iteration
        PFA_array.append(PFA)
        PD_array.append(PD)
    print('Finished generating Genus ROC curves')
    return np.array([PFA_array,PD_array])

