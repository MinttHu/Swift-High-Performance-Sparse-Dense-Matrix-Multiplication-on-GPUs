#! /usr/bin/python3

import matplotlib.pyplot as plt

x = [48, 96, 192, 384,758]
y1 = [38.99, 24.61, 17.24, 10.8,7.20]
y2 = [35.37, 22.42, 15.68, 10.65,6.94]



plt.plot(x, y1, marker='o', label='FP32')
plt.plot(x, y2, marker='s', label='FP64')

plt.tick_params(labelsize=16)
plt.gca().set_aspect(8.5)

plt.ylim(0, 40)


plt.axhline(y=6, color='red', linestyle='--', linewidth=1)


plt.xlabel("Column of dense matrix (N)",fontsize=20)
plt.ylabel("Average speed up",fontsize=20)
plt.legend()
plt.grid(True)
plt.savefig('irregular_polyline.png',bbox_inches='tight')
