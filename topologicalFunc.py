import numpy as np
import matplotlib.pyplot as plt
import dionysus as dio


def GaussianFiltration(GaussianRandomField, type='lower'):
    """GaussianFiltration

    Generates Filtration for the Gaussian Random Field.
    
    Args:
        GaussianRandomField (array): numpy 2-D array. The Gaussian Random Field generated from the class using Gen_GRF method.
        type (string): Takes ipnut either 'lower' or 'upper' for lower or upper filtration.
        nsize (integer): Size of the Gaussian Random Fields grid.

    Returns:
       Dionysus object: Filtration Diagram
    """
    if type == 'lower':
        f_lower_star = dio.fill_freudenthal(GaussianRandomField)
        lower_star_persistence = dio.homology_persistence(f_lower_star)
        dgms = dio.init_diagrams(lower_star_persistence, f_lower_star)
        return dgms
    elif type == 'upper':
        f_upper_star = dio.fill_freudenthal(GaussianRandomField, reverse=True)
        upper_star_persistence = dio.homology_persistence(f_upper_star)
        dgms = dio.init_diagrams(upper_star_persistence, f_upper_star)
        return dgms
    else:
        print('wrong input')
        return None


def GenerateBettiP(Filtraion,thresholds_start,thresholds_stop,type='lower'):
    """GenerateBettiP

    Generates the Betti numbers from the Filtration diagram.
    
    Args:
        Filtration (Dionysus object): Output of GaussianFiltration.
        thresholds_start (float): start value for generating superlevels of the Gaussian Random field .
        thresholds_stop (float): stop value for generating superlevels of the Gaussian Random field.

    Returns:
       Numpy array: Multidimensaion array contaiing Betti numbers for different dimensions
    """
    thresholds = np.arange(thresholds_start,thresholds_stop,0.01)
    Betti_p = []
    if type == 'lower':
        for i, dgm in enumerate(Filtraion):
            Betti_temp = []
            for nu in thresholds:
                Betti_num = 0
                for pt in dgm:
                    if (pt.birth <= nu and pt.death > nu):
                        Betti_num += 1
                Betti_temp.append(Betti_num)
            Betti_p.append(Betti_temp)
        return np.array(Betti_p)

    elif type == 'upper':
        for i, dgm in enumerate(Filtraion):
            Betti_temp = []
            for nu in thresholds:
                Betti_num = 0
                for pt in dgm:
                    if (pt.birth >= nu and pt.death < nu):
                        Betti_num += 1
                Betti_temp.append(Betti_num)
            Betti_p.append(Betti_temp)
        return np.array(Betti_p)

def GenerateGenus(Betti_array):
    """GenerateGenus

    Generates the Genus curve for gaussian random field using Betti arrays.
    
    Args:
        Betti array (numpy array): Betti array from GenerateBettiP.

    Returns:
       Numpy array: 1-D array contaiing Genus curve for the Gaussian random field.
    """
    dimmensions = Betti_array.shape[0]
    genus = Betti_array.shape[1]
    for i in dimmensions:
        genus += ((-1)**i)*Betti_array[i]
    return genus

def GenerateBettiAVG():


    return None

def GenerateGenusAVG():

    
    return None
