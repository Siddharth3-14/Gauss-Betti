import numpy as np

def LikelihoodROC(likelihoodratio0,likelihoodratio1):
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
    minimum = np.amin([trough1,trough0])
    maximum = np.amax([peak1,peak0])
    threshold_step = (maximum - minimum)/500
    threshold_start = minimum-3*threshold_step
    threshold_stop = maximum+3*threshold_step
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


def BettiROC(Betti_array_null,Betti_array_test,power_null,power_test):
    """BettiROC

    Generates PFA and PD values from Betti curves using the specified parameters
    
    Args:
        Betti_array_null (array): multiple Betti curves for null hypothesis generated from GenerateBetti function for a one dimension.
        Betti_array_test (array): multiple Betti curves for test hypothesis generated from GenerateBetti function for a one dimension.
        threshold_step (float): Step value for generating threshold array.

    Returns:
        Array: PFA and PD values
    """
    def smolPeak(Array):        
        iteration = Array.shape[0]
        minimumPeak = np.amax(Array[0])
        for i in range(1,iteration):
            if minimumPeak > np.amax(Array[i]):
                minimumPeak = np.amax(Array[i])
        return minimumPeak
        
    peak0 = np.amax(Betti_array_null)
    peak1 = np.amax(Betti_array_test)
    trough0 = smolPeak(Betti_array_null)
    trough1 = smolPeak(Betti_array_test)
    index = int((np.argmax(Betti_array_test[0])+np.argmax(Betti_array_null[0]))/2) 
    minimum = np.amin([trough0,trough1])
    maximum = np.amax([peak0,peak1])
    threshold_step = 0.1
    threshold_start = 0
    threshold_stop = 400
    thresholds = np.arange(threshold_start,threshold_stop,threshold_step)
    iteration = Betti_array_test.shape[0]
    PFA_array = []
    PD_array = []
    for lambd in thresholds:
        PFA = 0
        PD = 0
        for i in range(iteration):
            if Betti_array_null[i,index] > lambd:
                PFA += 1
            if Betti_array_test[i,index] > lambd:
                PD += 1

        PFA = PFA/iteration
        PD = PD/iteration
        PFA_array.append(PFA)
        PD_array.append(PD)
    print('Finished generating Betti ROC curves')
    if power_test > power_null:
        return np.array([PFA_array,PD_array])
    else:
        return np.array([PD_array,PFA_array])

def GenusROC(Genus_array_null,Genus_array_test,power_null,power_test):
    """GenusROC

    Generates PFA and PD values from Genus curves using the specified parameters
    
    Args:
        Genus_array_null (array): multiple Genus curves for null hypothesis generated from GenerateBetti function for a one dimension.
        Genus_array_test (array): multiple Genus curves for test hypothesis generated from GenerateBetti function for a one dimension.
        index (integer): index at which the Genus value is calculated and compared with threshold.
        threshold_step (float): Step value for generating threshold array.

    Returns:
        Array: PFA and PD values
    """
    def smolPeak(Array):        
        iteration = Array.shape[0]
        minimumPeak = np.amax(Array[0])
        for i in range(1,iteration):
            if minimumPeak > np.amax(Array[i]):
                minimumPeak = np.amax(Array[i])
        return minimumPeak
        
    peak0 = np.amax(Genus_array_null)
    peak1 = np.amax(Genus_array_test)
    trough0 = smolPeak(Genus_array_null)
    trough1 = smolPeak(Genus_array_test)
    index = int((np.argmax(Genus_array_null[0])+np.argmax(Genus_array_test[0]))/2)
    minimum = np.amin([trough0,trough1])
    maximum = np.amax([peak0,peak1])
    threshold_step = (maximum - minimum)/500
    threshold_start = minimum-3*threshold_step
    threshold_stop = maximum+3*threshold_step
    thresholds = np.arange(threshold_start,threshold_stop,threshold_step)
    iteration = Genus_array_test.shape[0]
    PFA_array = []
    PD_array = []
    for lambd in thresholds:
        PFA = 0
        PD = 0
        for i in range(iteration):
            if Genus_array_null[i,index] > lambd:
                PFA += 1
            if Genus_array_test[i,index] > lambd:
                PD += 1

        PFA = PFA/iteration
        PD = PD/iteration
        PFA_array.append(PFA)
        PD_array.append(PD)
    print('Finished generating Genus ROC curves')
    if power_test>power_null:
        return np.array([PFA_array,PD_array])
    else:
        return np.array([PD_array,PFA_array])

