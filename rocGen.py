import numpy as np
import topologicalFunc




def ROC_GEN_B(Nsize, n_0, n_1, iteration, threshold_start, threshold_stop, threshold_step):
    lambd = np.arange(threshold_start, threshold_stop, threshold_step)
    print('start 0')
    Betti_list0 = BettiList(Nsize, n_0, 100, iteration)
    print('end 0 start 1')
    Betti_list1 = BettiList(Nsize, n_1, 100, iteration)
    print('end 1')
    PD_array = []
    PFA_array = []
    for nu in lambd:
        print((nu-threshold_start)*100/(threshold_stop - threshold_start), "% complete")
        PD = 0
        PFA = 0
        for i in range(iteration):
            Betti0 = np.amax(Betti_list0[i])
            Betti1 = np.amax(Betti_list1[i])
            if Betti1 > nu:
                PD += 1
            if Betti0 > nu:
                PFA += 1
        PD_temp = PD/iteration
        PFA_temp = PFA/iteration
        PD_array.append(PD_temp)
        PFA_array.append(PFA_temp)
    return PD_array, PFA_array

Pd_array,Pfa_array = ROC_GEN_B(16,0.9,1,200,20,40,0.02)

length = len(Pd_array)
file = open('Betti0_roc_0_1.txt','w+')
for i in range(length):
    file.write(str(Pd_array[i]) + '\t' + str(Pfa_array[i])+'\n')
file.close()

plt.plot(Pfa_array,Pd_array)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('pfa')
plt.ylabel('pd')
plt.title('Betti0 roc nsize = 16, iteration = 200, n=0 and n=1')
plt.show()