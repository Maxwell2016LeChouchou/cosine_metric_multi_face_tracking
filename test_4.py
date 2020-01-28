import csv
import os
import numpy as np

gt = '/home/maxwell/Desktop/yt_test_data/tara_gt.txt'

csvfile = open(gt)
reader = csv.reader(csvfile)
rows = []
for row in reader:
    row = [int(row[0]), -1, row[2], row[3], row[4], row[5], 1, -1, -1, -1]
    rows.append(row)
rows = sorted(rows, key=lambda rows: rows[0])
# for row in rows:
#     print(row)
a = np.array(rows)
np.savetxt('/home/maxwell/Desktop/yt_test_data/tara_det.txt', a, fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s,%s")

