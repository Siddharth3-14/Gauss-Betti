import utilities
import topologicalFunc
import numpy as np
import matplotlib.pyplot as plt
import rocGen

Nsize = 12
power_index_null = 0
power_index_test = 1
iteration = 5
average = 5
filtration_threshold_start = -4
filtration_thhreshold_stop = 4

# type1= utilities.Generate_Likelihood_Array(Nsize,power_index_test,power_index_null,4)


# [Betti0,Betti1,Genus0,Genus1] = utilities.Generate_BettiGenus_array(Nsize,power_index_null,power_index_test,iteration,average,filtration_threshold_start,filtration_thhreshold_stop)

# [PFA_betti,PD_betti] = rocGen.BettiROC(Betti0[:,0,:],Betti1[:,0,:],_,_,_,_)
# utilities.plotROC(PFA_betti,PD_betti,Nsize,iteration,power_index_null,power_index_test,'Betti')
# utilities.SaveROC(PFA_betti,PD_betti,Nsize,iteration,power_index_null,power_index_test,'Betti')

# [PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1)
# utilities.plotROC(PFA_Genus,PD_Genus,Nsize,iteration,power_index_null,power_index_test,'Genus')
# utilities.SaveROC(PFA_Genus,PD_Genus,Nsize,iteration,power_index_null,power_index_test,'Genus')

