# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:02:20 2020

@author: timot
"""


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.color import rgb2gray
from utils import image_restoration

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

#%% All Results

labels = ['CodecError','Darkening','Decolorization','DirtyLens','Exposure','GaussianBlur','Haze','LensBlur','Noise','Rain','Shadow','Snow']

alexnet2 = np.genfromtxt(r'..\Cat3_Alexnet\data_all.csv',delimiter=',')
svm2 = 100*np.genfromtxt(r'..\Cat3_SVM_RGB_HOG\Cat3_SVM_RGB_HOG.csv',delimiter=',')
# svm3 = 100*np.genfromtxt(r'..\Cat4_SVM_RGB_HOG\Cat4_SVM_RGB_HOG.csv',delimiter=',')
# svm2_restore = 100*np.genfromtxt(r'..\Cat3_Restore_SVM_RGB_HOG\Cat3_Restore_SVM_RGB_HOG.csv',delimiter=',')
# svm2_restore_singlech = 100*np.genfromtxt(r'..\Cat3_Restore_SVM_RGB_HOG\Cat3_restore_singlechexp_SVM_RGB_HOG.csv',delimiter=',')
svm2_restore_bilat = 100*np.genfromtxt(r'..\Cat3_Restore_SVM_RGB_HOG\Cat3_restore_bilat_SVM_RGB_HOG.csv',delimiter=',')
alexnet2_restore = np.genfromtxt(r'..\Cat3_Alexnet_Restore\data_all_restore.csv',delimiter=',')

plt.figure(figsize=(40,20))
for idx,i_label in enumerate(labels):
    ax = plt.subplot(2,6,idx+1)
    plt.plot(np.arange(0,6),svm2[idx,:],'-o',label='SVM RGB HOG')
    plt.plot(np.arange(0,6),alexnet2[idx,:],'-o',label='AlexNet')
    # plt.plot(np.arange(0,6),svm3[idx,:],'-o',label='SVM RGB HOG w/ Unreal')
    # plt.plot(np.arange(0,6),svm2_restore[idx,:],'-o',label='SVM RGB HOG w/ Restore')
    # plt.plot(np.arange(0,6),svm2_restore_singlech[idx,:],'-o',label='SVM RGB HOG w/Restore')
    plt.plot(np.arange(0,6),svm2_restore_bilat[idx,:],'--o',label='SVM RGB HOG w/Restore')
    plt.plot(np.arange(0,6),alexnet2_restore[idx,:],'k--o',label='AlexNet w/Restore')
    plt.title(i_label, fontsize=30)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Challenge Level', fontsize=20)
    plt.xticks(np.arange(0,6))
    plt.ylim((20,100))
    plt.grid()
handles, leg_labs = ax.get_legend_handles_labels()
plt.figlegend(handles, leg_labs, loc = 'lower center', ncol=5, labelspacing=1, fontsize=30 )
plt.show()

#%% Mean Calcs

svm2_mean = (np.sum(svm2[:,1::])+svm2[0,0])/61
alexnet2_mean = (np.sum(alexnet2[:,1::])+alexnet2[0,0])/61
svm2_restore_mean = (np.sum(svm2_restore_bilat[:,1::])+svm2_restore_bilat[0,0])/61
alexnet2_restore_mean = (np.sum(alexnet2_restore[:,1::])+alexnet2_restore[0,0])/61

#%% AlexNet Pre vs Post Restoration

alexnet2_mean = np.mean(alexnet2[:,1::],axis=1)
alexnet2_restore_mean = np.mean(alexnet2_restore[:,1::],axis=1)
alexnet_diff = alexnet2_restore_mean-alexnet2_mean

svm2_mean = np.mean(svm2[:,1::],axis=1)
svm2_restore_mean = np.mean(svm2_restore_bilat[:,1::],axis=1)
svm_diff = svm2_restore_mean-svm2_mean
    
#%% Histogram

im2 = rgb2gray(plt.imread(r'..\CURE-TSR\Real_Test\Haze-2\01_01_12_02_0116.bmp'))
im5 = rgb2gray(plt.imread(r'..\CURE-TSR\Real_Test\Haze-5\01_01_12_05_0116.bmp'))

plt.figure()
plt.hist(im2.ravel()*255, bins=256, label='Haze Challenge 2')
plt.hist(im5.ravel()*255, bins=256, label='Haze Challenge 5')
plt.ylim([0,120])
plt.grid(axis='y')
plt.title('Haze Challenge Level Histograms')
plt.xlabel('Pixel Value Bin')
plt.ylabel('Count')
plt.legend()
plt.show()

#%% Restored Image

im = plt.imread(r'..\CURE-TSR\Real_Test\Haze-5\01_01_12_05_0116.bmp')
im_r = image_restoration(im,RGB=True,tensor=False)

plt.figure()
plt.subplot(1,2,1)
plt.title('Degraded')
plt.imshow(im)
plt.axis('off')
plt.subplot(1,2,2)
plt.title('Restored')
plt.imshow(im_r)
plt.axis('off')