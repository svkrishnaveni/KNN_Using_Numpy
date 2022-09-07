#!/usr/bin/env python
'''
This script loads data and evaluates implementation of KNN classifier for given Data
Author: Sai Venkata Krishnaveni Devarakonda
Date: 02/13/2022
'''

import numpy as np
import math
import re
from utilities import Load_2a3a_traindata, Load_2a3a_testdata, Load_2c2d3a3d_traindata
from utilities import cartesian_distance, knn_classifier, leave_one_out_knn

#path for train data and test data
str_path_2a3a_train = 'data_train_2a3a.txt'
str_path_2a3a_test = 'data_test_2a3a.txt'
str_path_2c2d3c3d_program = 'data_2c2d3c3d_program.txt'

# List of K
k = [1,3,5]

#2c) Performing LOO for a list of k =[1,3,5] using 3 train features(height,weight,age)
print('################ Running KNN using 3 features ################')
train_data,train_labels=Load_2c2d3a3d_traindata(str_path_2c2d3c3d_program,remove_age=False)
perf_k = [leave_one_out_knn(train_data,train_labels,i) for i in k]
print('Accuracy of LOO for KNN for k=1 is ',perf_k[0])
print('Accuracy of LOO for KNN for k=3 is ',perf_k[1])
print('Accuracy of LOO for KNN for k=5 is ',perf_k[2])
best_performer=max(perf_k)
print('Best performance is observed for k= {} with percentage of correct predictions: {}'.format(k[np.argmax(perf_k)],best_performer))


#2d) Performing LOO for a list of k =[1,3,5] using 2 train features(height,weight)
print('\n')

print('################ Running KNN using 2 features ################')
train_data,train_labels=Load_2c2d3a3d_traindata(str_path_2c2d3c3d_program,remove_age=True)
perf_k = [leave_one_out_knn(train_data,train_labels,i) for i in k]
print('Accuracy of LOO for KNN for k=1 is ',perf_k[0])
print('Accuracy of LOO for KNN for k=3 is ',perf_k[1])
print('Accuracy of LOO for KNN for k=5 is ',perf_k[2])
best_performer=max(perf_k)
print('Best performance is observed for k= {} with percentage of correct predictions: {}'.format(k[np.argmax(perf_k)],best_performer))



