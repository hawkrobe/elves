import csv
import numpy as np
import itertools
chainNum = 0

freqs = ['HF', 'LF']
test_angles = np.linspace(0, .5, num=11).round(3).tolist()
lf_costs = np.arange(1, 4.5, .5).tolist()
hf_costs = [1]
noise = [.1, .15,.2, .25]

combos = itertools.product(['HF'], ['LF'], ['HF'], lf_costs, hf_costs, noise, test_angles)

with open('input_grid.csv', 'w') as csv_file :
    writer = csv.writer(csv_file, delimiter=',')
    for freq1, freq2, freq3, lf_cost, hf_cost, noise, test_angle in combos:
        writer.writerow([chainNum, freq1, freq2, freq3, lf_cost, hf_cost, noise, test_angle])
        chainNum += 1