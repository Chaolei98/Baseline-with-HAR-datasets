import numpy as np
import os

def make_x(path):
    file_path = path
    all_file_names = os.listdir(file_path)
    x = []
    for file_name in all_file_names:
        x.append(np.genfromtxt(path + '/' + file_name))
    return np.transpose(x, (1, 2, 0))

train_x = make_x('./train/Inertial Signals')
test_x = make_x('./test/Inertial Signals')
train_y = np.genfromtxt('./train/y_train.txt', dtype=int).reshape(-1)
test_y = np.genfromtxt('./test/y_test.txt', dtype=int).reshape(-1)
