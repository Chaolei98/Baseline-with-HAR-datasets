import os
import pandas as pd
import scipy.io as io
import numpy as np
from sklearn.model_selection import train_test_split

def window(data, size, stride):
    '''将数组data按照滑窗尺寸size和stride进行切割'''
    x = []
    for i in range(0, data.shape[0], stride):
        if i+size <= data.shape[0]: #不足一个滑窗大小的数据丢
                x.append(data[i: i + size])
    return x

def merge(path, size, stride):
    '''合并数据
    path: USC-HAD路径'''
    result = [[] for i in range(12)]    #result的索引就是对应的动作标签

    subject_list = os.listdir(path)
    os.chdir(path)
    for subject in subject_list:
        if not os.path.isdir(subject):  #如果不是保存数据的文件夹，就跳过
            continue
        mat_list = os.listdir(subject)
        os.chdir(subject)
        for mat in mat_list:
            category = int(mat[1:-6])-1
            content = io.loadmat(mat)['sensor_readings']
            x = window(content, size, stride)
            result[category].extend(x)
        os.chdir('../../')
    os.chdir('../../')
    return result

def split(result, test_size):
    '''划分数据集
    test_size:测试集样本数量占比'''
    x_train, x_test, y_train, y_test = [], [], [], []
    for i, data in enumerate(result):
        label = [i for n in range(len(data))]
        x_train_, x_test_, y_train_, y_test_ = train_test_split(data, label, test_size=test_size, shuffle=True)
        x_train.extend(x_train_)
        y_train.extend(y_train_)
        x_test.extend(x_test_)
        y_test.extend(y_test_)
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    result = merge(r'/USC-HAD', 512, 256)
    x_train, y_train, x_test, y_test = split(result, 0.2)
    np.save('./USC-HAD/x_train', x_train)
    np.save('./USC-HAD/x_test', x_test)
    np.save('./USC-HAD/y_train', y_train)
    np.save('./USC-HAD/y_test', y_test)