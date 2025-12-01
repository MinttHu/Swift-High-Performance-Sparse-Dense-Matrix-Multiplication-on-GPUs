#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np



data_A = np.genfromtxt('mtxClassification.txt', delimiter=' ')
categories = data_A[:, 2]

values1 = data_A[:, 5]  # 
values2 = data_A[:, 8] 


colors1 = [(250/255, 130/255, 0/255, 1.0)]

colors2 = [(254/255, 183/255, 5/255, 1.0)]

colors3 = [ (33/255, 158/255, 188/255, 1.0)]

colors4 = [(2/255, 48/255, 74/255, 1.0)]

bar_width = 1.0

fig, ax = plt.subplots(figsize=(20, 10))

x_A_log = np.log10(categories)

plt.bar(np.arange(len(categories)), values1, color=colors1,  width=bar_width, label='Regular')
plt.bar(np.arange(len(categories)), values2, color=colors3,  bottom=values1, width=bar_width, label='Irregular')


plt.tick_params(labelsize=40)

plt.xlim(-0.5, len(categories) - 0.5)
plt.ylim(0, max(values1))
plt.tick_params(labelsize=45)

plt.xticks([])
plt.xlabel('nnz of sparse matrix',fontsize=50)
plt.ylabel('Ratio',fontsize=50)

plt.legend(loc='upper right', ncol=2 ,bbox_to_anchor=(1.020, 1.2),prop = {'size':55}, facecolor='none', edgecolor='none')

plt.savefig('irr_regular_ratio.png',bbox_inches='tight')



