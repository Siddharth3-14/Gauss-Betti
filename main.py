import utilities
import topologicalFunc
import numpy as np
import matplotlib.pyplot as plt
import rocGen

print('Starting...')

Nsize = 5                           #Size of the grid
power_index_null = 0.5              #Power Index of Null hypothesis 
power_index_test = 1                #Power index of Test hypothesis
iteration = 3                       #Iterations
average = 5                         #Number of average
filtration_threshold_start = -4     #Filtraion thresholds start value
filtration_thhreshold_stop = 4      #Filtration thresholds stop value
betti_dimension = 0                 #Dimension of Betti for which the ROC curve will be generated

[likelihoodratio0,likelihoodratio1] = utilities.Generate_Likelihood_Array(Nsize,power_index_null,power_index_test,iteration) #Generates the likelihood ratios arrays
PFA_likelihood,PD_likelihood = rocGen.LikelihoodROC(likelihoodratio0,likelihoodratio1,5)                                     #Generates the ROC curve for the likelihood ratio
utilities.plotROC(PFA_likelihood,PD_likelihood,Nsize,iteration,power_index_null,power_index_test,'likelihood')               #Plots the ROC curve
utilities.saveROC(PFA_likelihood,PD_likelihood,Nsize,iteration,power_index_null,power_index_test,'likelihood')               #Saves the ROC curve in txt file


[Betti0,Betti1,Genus0,Genus1] = utilities.Generate_BettiGenus_array(Nsize,power_index_null,power_index_test,average,iteration,filtration_threshold_start,filtration_thhreshold_stop) #Generates Betti and Genus arrays


[PFA_betti,PD_betti] = rocGen.BettiROC(Betti0[:,betti_dimension,:],Betti1[:,betti_dimension,:],0.01)    #Generates the Betti ROC curve
[PFA_Genus,PD_Genus] = rocGen.GenusROC(Genus0,Genus1,0.01)                                              #Generates the Genus ROC curve

utilities.plotROC(PFA_betti,PD_betti,Nsize,iteration,power_index_null,power_index_test,'betti',0)       #Plots the Betti ROC curve
utilities.plotROC(PFA_Genus,PD_Genus,Nsize,iteration,power_index_null,power_index_test,'genus')         #Plots the Genus ROC curve

utilities.saveROC(PFA_betti,PD_betti,Nsize,iteration,power_index_null,power_index_test,'betti',0)       #Saves the Betti ROC curve
utilities.saveROC(PFA_Genus,PD_Genus,Nsize,iteration,power_index_null,power_index_test,'Genus')         #Saves the Genus ROC curve

print('...Finished')