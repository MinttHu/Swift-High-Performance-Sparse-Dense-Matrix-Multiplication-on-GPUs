#! /usr/bin/python3

import math

def calculate_geometric_mean(numbers):
    product = 1
    for num in numbers:
        product *= num
    return math.pow(product, 1/len(numbers))

def process_file(filename):
    print(f"Process: {filename}")
    with open(filename, 'r') as file:
        data = file.readlines()

    cusparse = []
    Swift = []

    for line in data:
        columns = line.split()
        cusparse.append(float(columns[10]))
        Swift.append(float(columns[12]))

    chunk_size = 100

    def chunked_geomean(values):
        geomean_list = []
        for i in range(0, len(values), chunk_size):
            chunk = values[i:i+chunk_size]
            geomean_list.append(calculate_geometric_mean(chunk))
        return calculate_geometric_mean(geomean_list)

    final_cusparse = chunked_geomean(cusparse)
    final_Swift = chunked_geomean(Swift)

    cusparse_Swift = final_cusparse / final_Swift

    print(f"Geometric mean of cusparse: {final_cusparse:.4f}")
    print(f"Geometric mean of Swift: {final_Swift:.4f}")
    print(f"Swift faster than cusparse (GMean): {cusparse_Swift:.2f}")




process_file('FP32_K24.txt')
process_file('FP32_K48.txt')
process_file('FP32_K96.txt')
process_file('FP32_K192.txt')
process_file('FP32_K384.txt')
process_file('FP32_K768.txt')

process_file('FP64_K24.txt')
process_file('FP64_K48.txt')
process_file('FP64_K96.txt')
process_file('FP64_K192.txt')
process_file('FP64_K384.txt')
process_file('FP64_K768.txt')   

