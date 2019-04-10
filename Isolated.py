#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:20:43 2018

@author: Zhehao Li
"""

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab 
import seaborn as sns
from scipy.stats import norm
from sklearn.utils import shuffle
from matplotlib.font_manager import _rebuild
_rebuild()
#plt.rcParams['font.sans-serif'] = ['SimHei'] 
#plt.rcParams['axes.unicode_minus'] = False 




#######################################################################################################

#######################################################################################################


def PlotDistribution(groupA,step,name='Symmetry',c='orange',xlimit=1000,ylimit=300,bins_num=300):
    
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    ax1.hist(groupA, bins=bins_num, rwidth=0.9,label=name+'Steps'+str(step), color=c)
    ax1.set_xlim(0,xlimit)
    ax1.set_ylim(0,ylimit)
    ax1.set_ylabel('Agents Quantity')
    ax1.grid(axis='y', alpha=0.75)
    ax1.legend(loc='upper right',fontsize='small')
        
    plt.savefig(name + 'Steps'+str(step)+'.jpg')

def TwoBodyTransactions(Agents_inventory,agents_num,mode=1): #为方便两两交换，强制设置agents数量为偶数
    
    Inventory = shuffle(Agents_inventory)  #打乱原有inventory的排列顺序
    half_num = np.int(agents_num/2)         #将总人数对半分
    
    Group_1 = pd.Series(data = Inventory[:half_num].values) #胜利组
    Group_2 = pd.Series(data = Inventory[half_num:].values) #失败组
    
    new_Group_1 = pd.Series(data = Group_1) #复制胜利组inventory
    new_Group_2 = pd.Series(data = Group_2) #复制失败组inventory

    Mu = pd.Series(data = 0.1*np.random.rand(half_num)) #生成每组交换系数
    
    if mode == 1:    #这样的交换满足时间反演对称性
        Cash_avg = (Group_1 + Group_2)/2   
        CashFlow = Mu * Cash_avg           #生成每组交换现金流
        Mask = Group_2 > CashFlow          #判断失败租的agents的资金是否满足交易条件    
        new_Group_1[Mask] = Group_1[Mask] + CashFlow[Mask]  #更新胜利组的资金
        new_Group_2[Mask] = Group_2[Mask] - CashFlow[Mask]  #更新失败组的资金
    
    elif mode == 2:  #这样的交换不满足时间反演对称性
        CashFlow = Mu * Group_2           #生成每组交换现金流
        new_Group_1 = Group_1 + CashFlow  #更新胜利组的资金
        new_Group_2 = Group_2 - CashFlow  #更新失败组的资金
    
    Temp = pd.concat([new_Group_1,new_Group_2]) #合并胜利组与失败组
    new_Inventory = pd.Series(data = Temp.values) #建立新的inventory
    
    return new_Inventory

def linearfunc(x, a, b):
    return -a * x - b

def expfunc(x, a, b, c):
    return a * np.exp(-b * x) + c

def PlotExponentialDistribution(Data):
    fig = plt.figure(figsize=(8,4),dpi=200)
    ax1 = fig.add_subplot(111)
    font1 = {'family':'Times New Roman', 'weight':'normal', 'size':10}
    
    n_1, x_bins_1, patches_1 = ax1.hist(Data, bins=300, density=1, facecolor='green', alpha=0.5, label='Cash Distribution')  
    coefs_1, errors_1 = scipy.optimize.curve_fit(expfunc, x_bins_1[:-1], n_1,p0=[0.1,0.1,0.1])
    
    ax1.plot(x_bins_1[:-1], expfunc(x_bins_1[:-1], *coefs_1), 'r--', label='Exponential Distribution Curve') 
    ax1.vlines([99,101], ymin=0, ymax=1, transform=ax1.get_xaxis_transform(), colors='black', linewidth=0.4)
    ax1.grid(False)
    ax1.legend(loc='upper right', prop=font1)
    ax1.set_xlabel('Cash $m$', fontdict = font1)  
    ax1.set_ylabel('Probability $P(m)$', fontdict = font1)  
    ax1.set_xlim(0,600)
    ax1.set_ylim(0,0.012)
    
    plt.axes([0.67, 0.5, .2, .2])
    coefs_2, errors_2 = scipy.optimize.curve_fit(linearfunc, x_bins_1[:100], np.log(n_1[:100]), p0=[0.2,1])
    plt.scatter(x_bins_1[:-1],np.log(n_1), s=0.2, color='green', alpha=0.6, zorder=1)
    plt.plot(x_bins_1[:-1], linearfunc(x_bins_1[:-1], *coefs_2), 'r--', label='Fit Curve',linewidth = 0.6, zorder=2) 
    plt.xlim(0,300)
    plt.ylim(-8,-4)
    plt.xlabel('Cash $m$', fontsize='x-small')
    plt.ylabel('Log $P(m)$', fontsize='x-small')
    
    plt.savefig('Isolated_Symmetry.jpg')

def PlotGaussainDistribution(Data):
    fig = plt.figure(figsize=(8,4),dpi=200)
    ax1 = fig.add_subplot(111)
    font1 = {'family':'Times New Roman', 'weight':'normal', 'size':10}
    
    mu = Data.mean()
    sigma = Data.std()
    
    n_2, x_bins_2, patches_2 = ax1.hist(Data, bins=200, density=1,facecolor='green', alpha=0.5, label='Cash Distribution')  
    y_2 = mlab.normpdf(x_bins_2, mu, sigma)#画一条逼近的曲线  
    
    ax1.plot(x_bins_2, y_2, 'r--', label='Gaussian Distribution Curve')
    ax1.vlines([99,101], ymin=0, ymax=1, transform=ax1.get_xaxis_transform(), colors='black', linewidth=0.7)
    ax1.grid(False)
    ax1.legend(loc='upper right', prop=font1)
    ax1.set_xlabel('Cash $m$', fontdict = font1)  
    ax1.set_ylabel('Probability $P(m)$', fontdict = font1)  
    ax1.set_xlim(0,300)
    ax1.set_ylim(0,0.0200)
    plt.savefig('Isolated_Asymmetry.jpg') 



#######################################################################################################

#######################################################################################################


# 0. Set up the paratmeters



M = 5*100000
N = 5000
steps = 2001

# 1. Initialize
Agents_Inventory_1 = pd.Series(data = np.ones(N)*M/N)
Agents_Inventory_2 = pd.Series(data = np.ones(N)*M/N)


# 2. Loop
for s in range(steps):
    np.random.seed(s)
    '''
    if s%100 == 0:
        PlotDistribution(Agents_Inventory_1,s,name='Symmetry',c='orange',xlimit=1000,ylimit=300,bins_num=300)
        PlotDistribution(Agents_Inventory_2,s,name='Asymmetry',c='yellowgreen',xlimit=400,ylimit=100,bins_num=200)
    '''
    Agents_Inventory_1 = TwoBodyTransactions(Agents_Inventory_1,N,mode=1) 
    Agents_Inventory_2 = TwoBodyTransactions(Agents_Inventory_2,N,mode=2) 
       
    
PlotExponentialDistribution(Agents_Inventory_1)

PlotGaussainDistribution(Agents_Inventory_2)



''' seaborn画图(适用简单的正态分布)'''

'''
plt.figure(figsize=(8,4),dpi=100)
sns.set(style='whitegrid',palette='RdBu')
sns.distplot(Agents_Inventory_2, fit=norm, bins=200, kde=True, hist=True, 
             kde_kws={"linewidth":1.5, "label":"Kernal Desntiy Estimation"}, 
             hist_kws={"histtype":"step","linewidth": 1.0,"label": "Probability Distribution"})
plt.show()
'''

