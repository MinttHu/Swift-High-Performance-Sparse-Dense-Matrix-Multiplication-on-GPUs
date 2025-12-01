#! /usr/bin/python3

import matplotlib.pyplot as plt


x = [32, 48, 128, 256,512,1024,2048]
y1 = [1.31, 2.62, 5.23, 9.53,19.1,38.26,76.5]
y2 = [7.78,24.29,75.34,173.78,355.08,731.14,1489.72]



plt.plot(x, y1, marker='o', label='even distribution')
plt.plot(x, y2, marker='s', label='uneven distribution')

plt.tick_params(labelsize=16)
plt.gca().set_aspect(0.6)




plt.xlabel("Column of dense matrix (N)",fontsize=20)
plt.ylabel("Time (ms)",fontsize=20)
plt.legend()
plt.grid(True)
plt.savefig('motivation_loadbalance.png',bbox_inches='tight')
