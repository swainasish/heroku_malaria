#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:15:42 2021
Malaria_image_detect_Flask_web_app

@author: swain_asish
"""
from flask import Flask,render_template,request,redirect
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn


app = Flask(__name__)
app.secret_key = 'kfjfjk'

@app.route('/',methods=['GET',"POST"])

def home():
   return render_template('home.html')



@app.route('/mal_result',methods=['GET',"POST"])

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


def mal_result():
   if request.method=="POST":
        image = request.files['malaria']
        means=torch.tensor([0.5253, 0.4226, 0.4563])
        stds =torch.tensor([0.3299, 0.2679, 0.2841])
        image_transformation=transforms.Compose([
                      transforms.Resize(250),
                      transforms.CenterCrop(225),
                      transforms.ToTensor(),
                      transforms.Normalize(means,stds) 
        ])
        
        image_tensor = Image.open(image)
        img_tensor = image_transformation(image_tensor)
        img_size =img_tensor.shape
        img_tensor = img_tensor.view(1,img_size[0],img_size[1],img_size[2])
        model = torch.load('malaria_binary_6epoch.pt',torch.device('cpu'))
        ypred = model.forward(img_tensor)
        ypred = torch.max(ypred,1)[1]
        result =np.array(ypred)
        if result == 1:
            result = 'Congratulations , No malaria'
        else:
            result = 'Need Doctor\'s attention, High chance of malaria positive '    
        return render_template('mal_result.html',final_result=result)
        #except:
         # str2 = 'please upload the relevant image'
         # return render_template('mal_result.html',final_result=str2)

         
   return render_template('mal_preview.html')


@app.route('/preview1',methods=['GET','POST'])

def xyz():
   final_result = 'Congratulations , No malaria'
   return render_template('mal_result.html',final_result=final_result)


@app.route('/preview2',methods=['GET','POST']) 

def xyz1():
   result = 'Need Doctor\'s attention, High chance of malaria positive '    
   return render_template('mal_result.html',final_result=result)


if __name__=="__main__":
   app.run(debug=True)
