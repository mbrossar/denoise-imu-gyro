import numpy as np

paths = [
    '/home/mines/workspace/eval/EUROC/algorithms/Orinet/MH_04_difficult.csv',
    '/home/mines/workspace/eval/EUROC/algorithms/Orinet/MH_02_easy.csv',
    '/home/mines/workspace/eval/EUROC/algorithms/Orinet/V1_03_difficult.csv',
    '/home/mines/workspace/eval/EUROC/algorithms/Orinet/V2_02_medium.csv',
    '/home/mines/workspace/eval/EUROC/algorithms/Orinet/V1_01_easy.csv',
]

for path in paths:
    new_path = path[:-3] + "txt"
    data = np.genfromtxt(path, delimiter=",", skip_header=1)[::10]
    header = "timestamp(s) tx ty tz qx qy qz qw"
    new_data = np.zeros((data.shape[0], 8))
    new_data[:, 0] = data[:, 0]/1e9
    new_data[:, 4:7] = data[:, 3:6]
    new_data[:, 7] = data[:, 2]
    np.savetxt(new_path, new_data, header=header, fmt='%1.9f')
