#! /usr/bin/python3

def process_file(filename):
    print(f"process: {filename}")

    with open(filename, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        columns = line.split()

        column1 = float(columns[2])  # Sputiki
        column2 = float(columns[4])  # cuSPARSE
        column3 = float(columns[6])  # RoDe
        column4 = float(columns[8])  # ASpT
        column5 = float(columns[10]) # Swift

        
        column1_scaled = column1 * 1e6
        column2_scaled = column2 * 1e6
        column3_scaled = column3 * 1e6
        column4_scaled = column4 * 1e6
        column5_scaled = column5 * 1e6

        if column5_scaled != 0:
            result1 = column1_scaled / column5_scaled  # Sputiki / Swift
            result2 = column2_scaled / column5_scaled  # cuSPARSE / Swift
            result3 = column3_scaled / column5_scaled  # RoDe / Swift
            result4 = column4_scaled / column5_scaled  # ASpT / Swift
        else:
            result1 = result2 = result3 = result4 = 'N/A'

        new_line = line.strip() + f' {result1} {result2} {result3} {result4}\n'
        new_lines.append(new_line)

    with open(filename, 'w') as file:
        file.writelines(new_lines)

    print(f"complete: {filename}\n")


process_file('gemean_fp32_n32.txt')
process_file('gemean_fp32_n128.txt')
