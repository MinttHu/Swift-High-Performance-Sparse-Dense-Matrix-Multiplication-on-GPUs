#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np


data_A = np.genfromtxt('preprocess_Swift.txt', delimiter=' ')
categories = data_A[:, 6]

values2 = data_A[:, 10]
values3 = data_A[:, 14]
values5 = values2 + values3

colors1 = [(250/255, 130/255, 0/255, 1.0)]

colors2 = [(254/255, 183/255, 5/255, 1.0)]

colors3 = [ (33/255, 158/255, 188/255, 1.0)]

colors4 = [(2/255, 48/255, 74/255, 1.0)]

bar_width = 1.0
plt.figure(figsize=(10, 5)) 


x_A_log = np.log10(categories)


plt.bar(np.arange(len(categories)), values2, color=colors2,  width=bar_width, label='Sorting')
plt.bar(np.arange(len(categories)), values3, color=colors4,  bottom=values2, width=bar_width, label='Blocking')



plt.xlim(-0.5, len(categories) - 0.5)
plt.ylim(0, max(values5))
plt.tick_params(labelsize=25)

plt.xticks([])


plt.legend(loc='upper right', ncol=4 ,bbox_to_anchor=(1.007, 1.18),prop = {'size':25}, facecolor='none', edgecolor='none')
plt.xlabel(' nnz of sparse matrix ',fontdict={'weight':'normal','size': 25})

plt.ylabel('Ratio',fontdict={'weight':'normal','size': 25})

plt.savefig('sort_block_compare.png',bbox_inches='tight')



