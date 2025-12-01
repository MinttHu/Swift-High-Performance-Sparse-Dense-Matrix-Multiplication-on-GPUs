#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def plot_speedup(filename, output_image):
    print(f"process: {filename}")

    data = np.genfromtxt(filename, delimiter=' ')
    x = data[:, 8]
    y = data[:, 13]

    threshold = 1.0

    x_log = np.log10(x)

    plt.figure(figsize=(6,6))
    plt.scatter(x_log, y, c='purple', s=5, label='Speedup')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Divide Line')

    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 10, 1))
    plt.tick_params(labelsize=16)

    plt.xlabel('nnz of sparse matrix (log10 scale)', fontsize=20)
    plt.ylabel('Speedup', fontsize=20)

    plt.savefig(output_image, bbox_inches='tight')
    plt.close()

    print(f"complete: {output_image}\n")



plot_speedup('results_CMA_VS_nonCMA_32.txt', 'speedup_CMA(32).png')
plot_speedup('results_CMA_VS_nonCMA_128.txt', 'speedup_CMA(128).png')
