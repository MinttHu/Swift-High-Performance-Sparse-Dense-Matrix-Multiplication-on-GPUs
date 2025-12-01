#! /usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

def plot_opt_vs_nonopt(input_file, output_file):
    print(f"process: {input_file}")

    data_A = np.genfromtxt(input_file, delimiter=' ')
    with open(input_file, 'r') as file_a:
        categories = [line.split()[0] for line in file_a]

    values1 = data_A[:, 10]  # without opt
    values2 = data_A[:, 12]  # with opt

    colors1 = [(144/255, 201/255, 231/255, 1.0)] 
    colors2 = [(254/255, 183/255, 5/255, 1.0)]

    plt.figure(figsize=(20, 8)) 

    bar_width = 0.35
    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width

    plt.bar(bar_positions1, values1, color=colors1, hatch='//', width=bar_width, edgecolor='white', label='Without Optimization')
    plt.bar(bar_positions2, values2, color=colors2, hatch='x', width=bar_width, edgecolor='white', label='Optimization')

    plt.xticks(bar_positions1 + bar_width / 2, categories, rotation=70)
    plt.tick_params(labelsize=25)

    plt.xlabel('Matrix Name', fontsize=35)
    plt.ylabel('Time (ms)', fontsize=35)

    plt.legend(loc='upper right', ncol=2, bbox_to_anchor=(1.02, 1.24), prop={'size':43}, facecolor='none', edgecolor='none')

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Complete: {output_file}\n")



plot_opt_vs_nonopt('results_irrOpt_VS_nonOpt_32.txt', 'speedup_irr(32).png')
plot_opt_vs_nonopt('results_irrOpt_VS_nonOpt_128.txt', 'speedup_irr(128).png')
