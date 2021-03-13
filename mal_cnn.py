#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:09:18 2021

@author: minato
"""
import torch.nn as nn



class malaria_CNN(nn.Module):
   def __init__(self):
      super(malaria_CNN,self).__init__()
      
      #Convolution layers 
      conv_kernels=[16,32,64,128,64]
      conv_layers=[]
      input=3
      for n,i in enumerate(conv_kernels):
         conv_layers.append(nn.Conv2d(in_channels=input,out_channels=i,
                                      kernel_size=(5,5),stride=1,padding=1))
         conv_layers.append(nn.ReLU())
         conv_layers.append(nn.Dropout2d(p=0.3,inplace=True))
         if n%2==0:
            conv_layers.append(nn.MaxPool2d(3))
         input=i
      self.Conv_Seq=nn.Sequential(*conv_layers)      
      
      #Linear layers
      linear_neurons=[1500,600,40]
      linear_layers=[]
      input=2304
      for i in linear_neurons:
         linear_layers.append(nn.Linear(in_features=input,out_features=i))
         linear_layers.append(nn.ReLU(inplace=True))
         linear_layers.append(nn.Dropout2d(p=0.5))
         input=i
      linear_layers.append(nn.Linear(input,2))    
      self.Linear_Seq= nn.Sequential(*linear_layers)
                               
   def forward(self,X):
      X = self.Conv_Seq(X)
      samples=X.shape[0]
      X = X.view(samples,-1)
      X = self.Linear_Seq(X)
      return X
