import pandas as pd
import numpy as np
import sklearn.model_selection as sm

#读取csv数据为FrameData类型
data = pd.read_csv('../data/ml-20m/ratings.csv')
# print(type(data))
# print(data.columns)
# print(data.index.max())
# print(data.count())
# print(data.shape)
# del data['timestamp']
# print(data.columns)

#将FrameData数据类型转换为nArray数据类型
data = data.as_matrix()
# print(data)
x = data[:,:2]
y = data[:,2:3]
# print(x,y)

#将x,y分为测试集和训练集
train_x, test_x, train_y, test_y = sm.train_test_split(
    x, y, test_size=0.125, random_state = 5)
# print(train_x, test_x, train_y, test_y)

