import utilities
import topologicalFunc
import numpy as np
import matplotlib.pyplot as plt
import rocGen

print('Starting...')

Nsize = 20                           #Size of the grid
power_index_null = -3.5             #Power Index of Null hypothesis 
power_index_test = 1                #Power index of Test hypothesis
iteration = 500                     #Iterations
average = 20                        #Number of average
filtration_threshold_start = -4     #Filtraion thresholds start value
filtration_threshold_stop = 4      #Filtration thresholds stop value


[likelihoodratio0,likelihoodratio1] = utilities.Generate_Likelihood_Array(Nsize,power_index_null,power_index_test,iteration) #Generates the likelihood ratios arrays
PFA_likelihood,PD_likelihood = rocGen.LikelihoodROC(likelihoodratio0,likelihoodratio1,100)                                     #Generates the ROC curve for the likelihood ratio
utilities.plotROC(PFA_likelihood,PD_likelihood,Nsize,iteration,power_index_null,power_index_test,'likelihood')               #Plots the ROC curve
utilities.saveROC(PFA_likelihood,PD_likelihood,Nsize,iteration,power_index_null,power_index_test,'likelihood')               #Saves the ROC curve in txt file

# [Betti0,Betti1,Genus0,Genus1] = utilities.Generate_BettiGenus_array(Nsize,power_index_null,power_index_test,average,iteration,filtration_threshold_start,filtration_threshold_stop) #Generates Betti and Genus arrays

# [PFA_betti0,PD_betti0] = rocGen.BettiROC(Betti0[:,0,:],Betti1[:,0,:],0.01) #Generates the Betti ROC curve for betti0
# [PFA_betti1,PD_betti1] = rocGen.BettiROC(Betti0[:,1,:],Betti1[:,1,:],0.01) #Generates the Betti ROC curve for betti1
# [PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1,0.01)                 #Generates the Genus ROC curve

# utilities.plotROC(PFA_betti0,PD_betti0,Nsize,iteration,power_index_null,power_index_test,'betti',0)        #Plots the Betti0 ROC curve
# utilities.plotROC(PFA_betti1,PD_betti1,Nsize,iteration,power_index_null,power_index_test,'betti',1)        #Plots the Betti1 ROC curve
# utilities.plotROC(PFA_Genus,PD_Genus,Nsize,iteration,power_index_null,power_index_test,'genus')            #Plots the Genus ROC curve

# utilities.saveROC(PFA_betti0,PD_betti0,Nsize,iteration,power_index_null,power_index_test,'betti',0)        #Saves the Betti0 ROC curve
# utilities.saveROC(PFA_betti1,PD_betti1,Nsize,iteration,power_index_null,power_index_test,'betti',1)        #Saves the Betti1 ROC curve
# utilities.saveROC(PFA_Genus,PD_Genus,Nsize,iteration,power_index_null,power_index_test,'Genus')            #Saves the Genus ROC curve

print('...Finished')