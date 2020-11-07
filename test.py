# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 07:41:40 2020

@author: timot
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as tmodels

import os
import numpy as np
import time
import utils
from matplotlib import pyplot as plt

from train import evaluate

#%% Inputs

testdir = ['/content/drive/My Drive/ECE6258_Project/CURE-TSR/Real_Test']
resume = '/content/drive/My Drive/ECE6258_Project/checkpoints/AlexNet_Train2/model_best.pth.tar'
batch_size = 256
workers = 0
lr = 0.001
momentum = 0.9
weight_decay = 1e-4

#%% Import model

model = tmodels.alexnet(pretrained=True)
model.classifier[4] = nn.Linear(4096,1024)
model.classifier[6] = nn.Linear(1024,14)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
print("=> creating model %s " % model.__class__.__name__)

# define loss function (criterion) and optimizer
if torch.cuda.is_available():
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

print("=> loading checkpoint '{}'".format(resume))
checkpoint = torch.load(resume)
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {}, best_prec1 @ Source {})"
      .format(resume, checkpoint['epoch'], best_prec1))

#%% Iterate and test

iterate_dirs = ['CodecError','Darkening','Decolorization','DirtyLens','Exposure','GaussianBlur','Haze','LensBlur','Noise','Rain','Shadow','Snow']

i_testdir = [os.path.join(testdir[0],'ChallengeFree')]

# Transform for AlexNet
transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_dataset = utils.CURETSRDataset(i_testdir, transform)
# Load dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True)

# Evaluate
loss, top1, top5 = evaluate(test_loader, model, criterion)
accuracy_all = np.zeros((len(iterate_dirs),6))
accuracy_all[:,0] = top1
print('ChallengeFree: {top1:.3f}'.format(top1=top1))

for idx,sub_dir in enumerate(iterate_dirs):
    for i_level in range(5):
        sub_dir_str = sub_dir+'-'+str(i_level+1)
        i_testdir = [os.path.join(testdir,sub_dir_str)]
        # Transform for AlexNet
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_dataset = utils.CURETSRDataset(i_testdir, transform)
        # Load dataset
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=workers, pin_memory=True)
        
        # Evaluate
        loss, top1, top5 = evaluate(test_loader, model, criterion)
        accuracy_all = np.zeros((len(iterate_dirs),6))
        accuracy_all[idx,i_level+1] = top1
        print('{sub_dir_str}: {top1:.3f}'.format(sub_dir_str=sub_dir_str,top1=top1))
    
    plt.figure()
    plt.title(sub_dir)
    plt.ylabel('Accuracy')
    plt.xlabel('Challenge Level')
    plt.xticks(np.arange(0,6))
    plt.ylim((0.2,1))
    plt.plot(np.arange(0,6),accuracy_all[idx,:],'-o')
    plt.grid()
    plt.show()

np.savetxt('/content/drive/My Drive/ECE6258_Project/accuracy_all.csv',accuracy_all,delimiter=',')