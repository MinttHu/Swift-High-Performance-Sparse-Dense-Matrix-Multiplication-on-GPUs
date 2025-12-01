#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

configs = [
    ("gemean_fp32_n32.txt",  "32"),
    ("gemean_fp32_n128.txt", "128"),
]

for filename, bn in configs:
    print(f"Processing {filename} ...")

    data_A = np.genfromtxt(filename, delimiter=' ')
    x_A = data_A[:, 11]  # nnz

    metrics = [
        (data_A[:, 12], 'Spuntik',  f'speedup_spuntik(f32-{bn}).png',  'blue'),
        (data_A[:, 13], 'cuSPARSE', f'speedup_cusparse(f32-{bn}).png', 'yellow'),
        (data_A[:, 14], 'RoDe',     f'speedup_rode(f32-{bn}).png',     'black'),
        (data_A[:, 15], 'ASpT',     f'speedup_aspt(f32-{bn}).png',     'pink'),
    ]

    threshold = 1.0

    for y_data, label, outname, color in metrics:
        plt.figure(figsize=(6, 6))

        plt.scatter(x_A, y_data, c=color, s=5, label=label)
        plt.axhline(y=threshold, color='red', linestyle='--')

        plt.tick_params(labelsize=16)

        plt.xlabel('Matrix ID', fontsize=20)
        plt.ylabel('Speedup', fontsize=20)

        plt.savefig(outname, bbox_inches='tight')
        plt.close()

print("Done.")
