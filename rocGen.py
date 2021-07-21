import numpy as np

def LikelihoodROC(type1,type2,threshold_start,threshold_stop,threshold_step):
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
    return PFA_array,PD_array

    return None


def BettiROC(Betti_array,index,threshold_start,threshold_stop,threshold_step):
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
    return PFA_array,PD_array


def GenusROC(Genus_array,index,threshold_start,threshold_stop,threshold_step):
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
    return PFA_array,PD_array

