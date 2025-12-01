#! /usr/bin/python3

def process_file(filename):
    print(f"processing: {filename}")

    with open(filename, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        columns = line.split()

        column1 = float(columns[10])
        column5 = float(columns[12])

        column1_scaled = column1 * 1e6
        column5_scaled = column5 * 1e6

        if column5_scaled != 0:
            result1 = column1_scaled / column5_scaled
        else:
            result1 = 'N/A'

        new_line = line.strip() + f' {result1}\n'
        new_lines.append(new_line)

    with open(filename, 'w') as file:
        file.writelines(new_lines)

    print(f"complete: {filename}\n")


process_file('ideal-cma.txt')
process_file('ideal-cma-128.txt')
