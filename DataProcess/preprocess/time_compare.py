#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt



data_A = np.genfromtxt('pre_result__spmm.txt', delimiter=' ')
data_B = np.genfromtxt('pre_result_ASpT.txt', delimiter=' ')
data_C = np.genfromtxt('preprocess_Swift.txt', delimiter=' ')

colors1 = [(250/255, 130/255, 0/255, 1.0)]

colors2 = [(254/255, 183/255, 5/255, 1.0)]

colors3 = [ (33/255, 158/255, 188/255, 1.0)]

colors4 = [(2/255, 48/255, 74/255, 1.0)]

x_A = data_A[:, 3]  
  

y_B = data_A[:, 4] #Sputnik
 
y_C = data_A[:, 5] #RoDe

x_D = data_B[:, 3]
y_D = data_B[:, 4] #ASpT


x_E = data_C[:, 6]
y_E = data_C[:, 16] #Swift

threshold = 1.0 


x_A_log = np.log10(x_A)
x_D_log = np.log10(x_D)
x_E_log = np.log10(x_E)


y_B_log = np.log10(y_B)
y_C_log = np.log10(y_C)
y_D_log = np.log10(y_D)
y_E_log = np.log10(y_E)

plt.figure(figsize=(10, 5)) 

plt.scatter(x_D_log, y_D_log, c=colors2, s=45,marker = '>', label='ASpT')
plt.scatter(x_A_log, y_C_log, c=colors4, s=45,marker = '+', label='RoDe')
plt.scatter(x_A_log, y_B_log, c=colors3, s=45, marker = 's',label='Sputnik')
plt.scatter(x_E_log, y_E_log, c=colors1, s=45,marker = 'x',label='Swift')



plt.tick_params(labelsize=25)
plt.legend(loc='upper right', ncol=4 ,columnspacing=0.05,markerscale=2,bbox_to_anchor=(1.0, 1.18),prop = {'size':20}, facecolor='none', edgecolor='none')



plt.xlabel('nnz of sparse matrix (log10 scale)',fontsize=25)

plt.ylabel('Time (ms)',fontsize=25)


plt.savefig('time_preprocess.png',bbox_inches='tight')
