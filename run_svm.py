# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:01:27 2020

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

# Data directories
# train_dir = ['..\CURE-TSR\Real_Train\ChallengeFree'] # Train 1
train_dir = ['..\CURE-TSR\Real_Train'] # Train 2
# train_dir = ['..\CURE-TSR\Real_Train\ChallengeFree','..\CURE-TSR\\3_Unreal_Test'] # Train 3
# test_dir = ['..\CURE-TSR\Real_Test\ChallengeFree'] # Test 1
test_dir = ['..\CURE-TSR\Real_Test'] # Test 2
iterate_challenges = True # Iterate through challenges

# Restore images
restore = True

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
max_iter = 1000 #-1
decision_function_shape = 'ovr'

#%% Preprocessing / load data

print('\nGenerating training data...\n')
# Generate training dataset
x_train, labels_train, type_train, level_train = vectorize_and_label(train_dir, do_RGB=do_RGB, do_hog=do_hog,
                                                                      orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm, restore=restore)

# Save data
np.savetxt('x_train_2_restore_singlechexp.csv',x_train,delimiter=',')
np.savetxt('labels_train_train_2_restore_singlechexp.csv',labels_train,delimiter=',')
np.savetxt('type_train_2_restore_singlechexp.csv',type_train,delimiter=',')
np.savetxt('level_train_2_restore_singlechexp.csv',level_train,delimiter=',')

# Load training data
# x_train = np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\x_train_2.csv',delimiter=',')
# labels_train = np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\labels_train_2.csv',delimiter=',')
# type_train = np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\type_train_2.csv',delimiter=',')
# level_train = np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\level_train_2.csv',delimiter=',')

#%% Shuffle data

# Shuffle training data
permutation = np.random.permutation(x_train.shape[0])
x_train_shuff = x_train[permutation,:]
labels_train_shuff = labels_train[permutation]
type_train_shuff = type_train[permutation]
level_train_shuff = level_train[permutation]

#%% Create and train classifier
print('\nTraining classifier...\n')
svm_clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,\
                  tol=tol, max_iter=max_iter, decision_function_shape=decision_function_shape)

start_train = time.time()
svm_clf.fit(x_train_shuff, labels_train_shuff)
end_train = time.time()
total_time_mins = (end_train-start_train)/60

# Save model
pkl_filename = "svm_clf_2_restore_single_channel_exposure.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svm_clf, file)

# Load Train 3 model
# with open(r'..\Cat4_SVM_RGB_HOG\svm_clf_3.pkl', 'rb') as file:
#     svm_clf = pickle.load(file)

#%% Classify test dataset
print('\nTesting classifier...\n')
iterate_dirs = ['CodecError','Darkening','Decolorization','DirtyLens','Exposure','GaussianBlur','Haze','LensBlur','Noise','Rain','Shadow','Snow']

if iterate_challenges:
    test_dir_joined = [os.path.join(test_dir[0],'ChallengeFree')]
    x_test, labels_test, type_test, level_test = vectorize_and_label(test_dir_joined, do_RGB=do_RGB, do_hog=do_hog,
                                                                     orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm, restore=restore)
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
        
    np.savetxt('Cat3_restoresinglechexposure_SVM_RGB_HOG.csv',accuracy_all,delimiter=',')
    
else:
    x_test, labels_test, type_test, level_test = vectorize_and_label(test_dir, do_RGB=do_RGB, do_hog=do_hog,
                                                                     orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm, restore=restore)
    predictions = svm_clf.predict(x_test)
    class_report = classification_report(labels_test, predictions)
    print(class_report)
    
    conf_mat = confusion_matrix(labels_test,predictions)
    df_cm = pd.DataFrame(conf_mat, index = np.unique(labels_test),
                      columns = np.unique(labels_test))
    plt.figure(figsize = (10,7))
    plt.title('Confusion Matrix')
    sn.heatmap(df_cm, annot=True, fmt='d')