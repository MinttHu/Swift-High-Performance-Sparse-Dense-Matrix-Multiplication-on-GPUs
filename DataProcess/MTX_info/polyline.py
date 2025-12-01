#! /usr/bin/python3

import matplotlib.pyplot as plt


x_labels = ["0-0.001", "0.001-0.002", "0.002-0.003", "0.003-0.004", "0.004-0.005","0.005-0.006","0.006-0.007","0.007-0.008","0.008-0.009","0.009-0.01","0.01-0.1",">0.1"]

x = range(len(x_labels))

y1 = [4.45, 11.91, 16.48, 25.01, 16.95,20.89,19.50,24.04,39.10,18.08,32.12,55.67]

plt.figure(figsize=(8,5))
plt.plot(x, y1, marker='o')

plt.xticks(x, x_labels, rotation=45,fontsize=10)

plt.xlabel("Range of Sparsity",fontsize=20)
plt.ylabel("Average speed up",fontsize=20)

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig('polyline_sparsity.png',bbox_inches='tight')
