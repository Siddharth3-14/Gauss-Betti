import numpy as np

def LikelihoodROC(type1,type2,threshold_start,threshold_stop,threshold_step):
    """LikelihoodROC

    Generates PFA and PD values from likelihood ratios of the Gaussian Random Field using the specified parameters
    
    Args:
        type1 (array): 2-D arrays containing lieklihood ratios of the Gaussian Random field of type 1 see doc for likelihoodratio .
        type2 (array): 2-D arrays containing lieklihood ratios of the Gaussian Random field of type 2 see doc for likelihoodratio
        threshold_start (float): Start value for generating threshold array.
        threshold_stop (float): Stop value for generating threshold array.
        threshold_step (float): Step value for generating threshold array.

    Returns:
        Array: PFA and PD values
    """
    thresholds = np.arange(threshold_start,threshold_stop,threshold_step)
    iteration = type1.shapep[0]
    PFA_array = []
    PD_array = []
    for lambd in thresholds:
        PFA = 0
        PD = 0
        for i in range(iteration):
            if type1[i] > lambd:
                PD += 1
            elif type2[i] > lambd:
                PFA += 1
        PFA = PFA/iteration
        PD = PD/iteration
        PFA_array.append(PFA)
        PD_array.append(PD)
    return [PFA_array,PD_array]


def BettiROC(Betti_array,index,threshold_start,threshold_stop,threshold_step):
    """BettiROC

    Generates PFA and PD values from Betti curves using the specified parameters
    
    Args:
        Betti_array (array): multiple Betti curves generated from GenerateBetti function for a one dimension.
        index (integer): index at which the Genus value is calculated and compared with threshold.
        threshold_start (float): Start value for generating threshold array.
        threshold_stop (float): Stop value for generating threshold array.
        threshold_step (float): Step value for generating threshold array.

    Returns:
        Array: PFA and PD values
    """
    thresholds = np.arange(threshold_start,threshold_stop,threshold_step)
    iteration = Betti_array.shapep[0]
    PFA_array = []
    PD_array = []
    for lambd in thresholds:
        PFA = 0
        PD = 0
        for i in range(iteration):
            if Betti_array[i][index] > lambd:
                PD += 1
            elif Betti_array[i][index] < lambd:
                PFA += 1
        PFA = PFA/iteration
        PD = PD/iteration
        PFA_array.append(PFA)
        PD_array.append(PD)
    return [PFA_array,PD_array]


def GenusROC(Genus_array,index,threshold_start,threshold_stop,threshold_step):
    """GenusROC

    Generates PFA and PD values from Genus curves using the specified parameters
    
    Args:
        Genus_array (array): multiple Genus curves generated from GenerateGenus function.
        index (integer): index at which the Genus value is calculated and compared with threshold.
        threshold_start (float): Start value for generating threshold array.
        threshold_stop (float): Stop value for generating threshold array.
        threshold_step (float): Step value for generating threshold array.

    Returns:
        Array: PFA and PD values
    """
    thresholds = np.arange(threshold_start,threshold_stop,threshold_step)
    iteration = Genus_array.shapep[0]
    PFA_array = []
    PD_array = []
    for lambd in thresholds:
        PFA = 0
        PD = 0
        for i in range(iteration):
            if Genus_array[i][index] > lambd:
                PD += 1
            elif Genus_array[i][index] < lambd:
                PFA += 1
        PFA = PFA/iteration
        PD = PD/iteration
        PFA_array.append(PFA)
        PD_array.append(PD)
    return [PFA_array,PD_array]

