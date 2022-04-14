import scipy.io as scio
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np


data = scio.loadmat('./data/acc_data.mat')['acc_data']
label = scio.loadmat('./data/acc_labels.mat')['acc_labels'][:, 0]   #标签有3轴，第1轴是动作种类
# 注意，读取出来的data是字典格式，可以通过函数type(data)查看。
# print(type(data))
# print(list(data.keys()))

data = data.reshape(data.shape[0], 3, 151).transpose(0, 2, 1)   #453:是三个151，分别表示加速度计x，y，z的三个窗口
# print(data.shape)

categories = len(list(Counter(label).keys()))   #17种动作种类
print(data.shape)

#每种动作按照8/2分
x_train, x_test, y_train, y_test = [], [], [], []
for i in range(1, categories+1):
    cur_data = data[label == i]
    cur_label = label[label == i]
    cur_x_train, cur_x_test, cur_y_train, cur_y_test = train_test_split(cur_data, cur_label, test_size=0.2, shuffle=True)
    x_train += cur_x_train.tolist()
    x_test += cur_x_test.tolist()
    y_train += cur_y_train.tolist()
    y_test += cur_y_test.tolist()


np.save('./x_train', np.array(x_train))
np.save('./x_test', np.array(x_test))
np.save('./y_train', np.array(y_train)-1)   #从0开始
np.save('./y_test', np.array(y_test)-1)