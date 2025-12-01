#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt



data_A = np.genfromtxt('ASpT_Swift.txt', delimiter=' ')
x_A = data_A[:, 2]  # blk_ratio

y_mean = data_A[:,5]

threshold = 1.0 

plt.scatter(x_A, y_mean, c='pink', s=5, label='average speedup')

plt.axhline(y=threshold, color='red', linestyle='--', label='Divide Line')




my_y_ticks = np.arange(0, 10, 2)


plt.tick_params(labelsize=16)



plt.title('')
plt.xlabel('Ratio',fontsize=20)
plt.ylabel('Speedup',fontsize=20)


plt.savefig('analysisASpT.png',bbox_inches='tight')