# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:01:27 2020

@author: timot
"""


import numpy as np
import seaborn as sn
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from utils import *

#%% Inputs

# Data directories
# train_dir = ['CURE-TSR\Real_Train\ChallengeFree'] # Train 1
train_dir = ['CURE-TSR\Real_Train'] # Train 2
# test_dir = ['CURE-TSR\Real_Test\ChallengeFree'] # Test 1
test_dir = ['CURE-TSR\Real_Test'] # Test 2
iterate_challenges = True # Iterate through challenges

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
max_iter = -1
decision_function_shape = 'ovr'

#%% Preprocessing

# Generate training dataset
x_train, labels_train, type_train, level_train = vectorize_and_label(train_dir, do_RGB=do_RGB, do_hog=do_hog,
                                                                     orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm)

# Shuffle training data
permutation = np.random.permutation(x_train.shape[0])
x_train_shuff = x_train[permutation,:]
labels_train_shuff = labels_train[permutation]
type_train_shuff = type_train[permutation]
level_train_shuff = level_train[permutation]

#%% Create classifier

svm_clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,\
                  tol=tol, max_iter=max_iter, decision_function_shape=decision_function_shape)
    
#%% Train classifier

svm_clf.fit(x_train_shuff, labels_train_shuff)

#%% Classify test dataset

iterate_dirs = ['CodecError','Darkening','Decolorization','DirtyLens','Exposure','GaussianBlur','Haze','LensBlur','Noise','Rain','Shadow','Snow']

if iterate_challenges:
    test_dir_joined = [os.path.join(test_dir[0],'ChallengeFree')]
    x_test, labels_test, type_test, level_test = vectorize_and_label(test_dir_joined, do_RGB=do_RGB, do_hog=do_hog,
                                                                     orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm)
    predictions = svm_clf.predict(x_test)
    accuracy_all = np.zeros((len(iterate_dirs),6))
    accuracy_all[:,0] = np.round(np.where((labels_test-predictions)==0)[0].shape[0]/predictions.shape[0],2)
    for idx,sub_dir in enumerate(iterate_dirs):
        for i_level in range(5):
            sub_dir_str = sub_dir+'-'+str(i_level+1)
            test_dir_joined = [os.path.join(test_dir[0],sub_dir_str)]
            x_test, labels_test, type_test, level_test = vectorize_and_label(test_dir_joined, do_RGB=do_RGB, do_hog=do_hog,
                                                                     orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm)
            predictions = svm_clf.predict(x_test)
            accuracy_all[idx,i_level+1] = np.round(np.where((labels_test-predictions)==0)[0].shape[0]/predictions.shape[0],2)
        
        plt.figure()
        plt.title(sub_dir)
        plt.ylabel('Accuracy')
        plt.xlabel('Challenge Level')
        plt.xticks(np.arange(0,6))
        plt.ylim((0.2,1))
        plt.plot(np.arange(0,6),accuracy_all[idx,:],'-o')
        plt.grid()
        plt.show()
    
else:
    x_test, labels_test, type_test, level_test = vectorize_and_label(test_dir, do_RGB=do_RGB, do_hog=do_hog,
                                                                     orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm)
    predictions = svm_clf.predict(x_test)
    class_report = classification_report(labels_test, predictions)
    print(class_report)
    
    conf_mat = confusion_matrix(labels_test,predictions)
    df_cm = pd.DataFrame(conf_mat, index = np.unique(labels_test),
                      columns = np.unique(labels_test))
    plt.figure(figsize = (10,7))
    plt.title('Confusion Matrix')
    sn.heatmap(df_cm, annot=True, fmt='d')
