#! /usr/bin/python3

import math

def calculate_geometric_mean(numbers):
    product = 1.0
    for num in numbers:
        product *= num
    return math.pow(product, 1 / len(numbers))


def process_file(filename):
    print(f"\nProcessing: {filename}")

    with open(filename, 'r') as f:
        lines = f.readlines()

    Sputiki = []
    cusparse = []
    RoDe = []
    ASpT = []
    Swift = []

    for line in lines:
        cols = line.split()
        Sputiki.append(float(cols[2]))
        cusparse.append(float(cols[4]))
        RoDe.append(float(cols[6]))
        ASpT.append(float(cols[8]))
        Swift.append(float(cols[10]))

    chunk_size = 100

    def chunk_geomean(arr):
        chunks = [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]
        means = [calculate_geometric_mean(chunk) for chunk in chunks]
        return calculate_geometric_mean(means)

    final_a = chunk_geomean(Sputiki)
    final_b = chunk_geomean(cusparse)
    final_c = chunk_geomean(RoDe)
    final_d = chunk_geomean(ASpT)
    final_e = chunk_geomean(Swift)

    
    sput_fast = final_a / final_e
    cusp_fast = final_b / final_e
    rode_fast = final_c / final_e
    aspt_fast = final_d / final_e

    print(f'Geometric mean of sputiki: {final_a:.4f}')
    print(f'Geometric mean of cusparse: {final_b:.4f}')
    print(f'Geometric mean of Rode: {final_c:.4f}')
    print(f'Geometric mean of ASpT: {final_d:.4f}')
    print(f'Geometric mean of Swift: {final_e:.4f}')

    print(f'Swift faster than sputiki (GMean): {sput_fast:.2f}')
    print(f'Swift faster than cusparse (GMean): {cusp_fast:.2f}')
    print(f'Swift faster than Rode (GMean):    {rode_fast:.2f}')
    print(f'Swift faster than ASpT (GMean):    {aspt_fast:.2f}')



process_file('gemean_fp32_n32.txt')
process_file('gemean_fp32_n128.txt')
