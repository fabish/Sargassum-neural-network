#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:56:07 2021

@author: courtrai
"""

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F



class ResidualBlock2D(nn.Module):
    def __init__(self, channels , k=3, p=1):
        super(ResidualBlock2D, self).__init__()
        self.net = nn.Sequential(
			nn.Conv2d( channels, channels, kernel_size=k, padding=p),
			nn.BatchNorm2d( channels),
			nn.PReLU(),
        	
        	    nn.Conv2d( channels,  channels, kernel_size=k, padding=p),
        	    nn.BatchNorm2d( channels)
        )
    def forward(self, x):
        return x + self.net(x)



class SargassumNet2DV3L_3x3_Residual(nn.Module ):
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64)):
        super(SargassumNet2DV3L_3x3_Residual, self).__init__()
        
        
        
        self.dims=dims
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        self.res3=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.res4=  ResidualBlock2D(dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        self._initialize_weights()
        
    def _initialize_weights(self):
       
        init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes3ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res4.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res4.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
    def forward(self, x):
     
        out = self.layer1(x)     
        out = self.res1(out)
        x_size=out.size()
       
        out,idx1 = self.maxPool1(out)
       
        out=self.convMaxToRes2(out)
        out = self.res2(out)
        
        out = self.res3(out)
        out=self.convRes3ToUnPool(out)
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        out = self.res4(out)
        out =self.last(out)
    
        return out    

 
    
   
    
    
 
