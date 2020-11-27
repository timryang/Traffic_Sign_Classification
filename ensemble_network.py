# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:34:25 2020

@author: timot
"""


import numpy as np
import seaborn as sn
import pandas as pd
import os
import time
import pickle
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from utils import *

#%% Inputs

train_dir = ['..\CURE-TSR\Real_Train\LensBlur-3']
test_dir = ['..\CURE-TSR\Real_Test\LensBlur-1','..\CURE-TSR\Real_Test\LensBlur-2','..\CURE-TSR\Real_Test\LensBlur-3','..\CURE-TSR\Real_Test\LensBlur-4','..\CURE-TSR\Real_Test\LensBlur-5']

# Historgram of oriented gradients
do_hog = True

# RGB vs grayscale
do_RGB = True

# HOG params
orientations = 8
ppc = 4
cpb = 2
block_norm = 'L2'

# SVM params
C = 1
kernel = 'rbf'
gamma = 'scale'
degree = 3
coef0 = 0
tol = 1e-3
max_iter = -1 #-1
decision_function_shape = 'ovr'

#%% Preprocessing / load data

# Generate training dataset
# x_train, labels_train, type_train, level_train = vectorize_and_label(train_dir, do_RGB=do_RGB, do_hog=do_hog,
#                                                                       orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm)

# Load Cat3 SVM_RGB_HOG training data
x_train = np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\x_train_2.csv',delimiter=',')
labels_train = np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\labels_train_2.csv',delimiter=',')
type_train = np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\type_train_2.csv',delimiter=',')
level_train = np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\level_train_2.csv',delimiter=',')

#%% Shuffle data

# Shuffle training data
permutation = np.random.permutation(x_train.shape[0])
x_train_shuff = x_train[permutation,:]
labels_train_shuff = labels_train[permutation]
type_train_shuff = type_train[permutation]
level_train_shuff = level_train[permutation]

#%% Create classifier

# svm_clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,\
#                   tol=tol, max_iter=max_iter, decision_function_shape=decision_function_shape)

# start_train = time.time()
# svm_clf.fit(x_train_shuff, type_train_shuff)
# end_train = time.time()
# total_time_mins = (end_train-start_train)/60

# Save model
# pkl_filename = "svm_clf_type.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(svm_clf, file)

# Load model
with open(r'..\Ensemble\svm_clf_type.pkl', 'rb') as file:
    svm_clf = pickle.load(file)

#%% Test classifier

x_test, labels_test, type_test, level_test = vectorize_and_label(test_dir, do_RGB=do_RGB, do_hog=do_hog,
                                                                     orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm)
predictions = svm_clf.predict(x_test)
class_report = classification_report(type_test, predictions)
print(class_report)

conf_mat = confusion_matrix(type_test,predictions)
df_cm = pd.DataFrame(conf_mat, index = np.unique(type_test),
                  columns = np.unique(type_test))
plt.figure(figsize = (10,7))
plt.title('Confusion Matrix')
sn.heatmap(df_cm, annot=True, fmt='d')

