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

    average = []
    Swift = []

    for line in data:
        columns = line.split()
        average.append(float(columns[8]))
        Swift.append(float(columns[7]))

    chunk_size = 100

    def chunked_geomean(values):
        geomean_list = []
        for i in range(0, len(values), chunk_size):
            chunk = values[i:i+chunk_size]
            geomean_list.append(calculate_geometric_mean(chunk))
        return calculate_geometric_mean(geomean_list)

    final_average = chunked_geomean(average)
    final_Swift = chunked_geomean(Swift)

    average_Swift = final_average / final_Swift

    print(f"Geometric mean of average: {final_average:.4f}")
    print(f"Geometric mean of Swift: {final_Swift:.4f}")
    print(f"Swift faster than average (GMean): {average_Swift:.2f}")




process_file('0-1.txt')
process_file('1-2.txt')
process_file('2-3.txt') 
process_file('3-4.txt')
process_file('4-5.txt')
process_file('5-6.txt')
process_file('6-7.txt')
process_file('7-8.txt')
process_file('8-9.txt')
process_file('9-10.txt')
process_file('10-100.txt')
process_file('100.txt')




