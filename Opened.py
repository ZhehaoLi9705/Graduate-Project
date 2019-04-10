#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:27:38 2019

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
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 



#######################################################################################################

#######################################################################################################


def PlotDistribution(groupA, step, name='Symmetry', c='orange', xlimit=3500, ylimit=150, bins_num=400):
    
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    ax1.hist(groupA, bins=bins_num, rwidth=0.9,label=name+'Steps'+str(step), color=c)
    ax1.set_xlim(0,xlimit)
    ax1.set_ylim(0,ylimit)
    ax1.set_ylabel('Agents Quantity')
    ax1.grid(axis='y', alpha=0.75)
    ax1.legend(loc='upper right',fontsize='small')
        
    plt.savefig(name + 'Steps'+str(step)+'.jpg')

def TwoBodyTransactions(Agents_inventory, agents_num, mode=1): #为方便两两交换，强制设置agents数量为偶数
    
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
  
def SelectAgents(Agents_inventory, moneyflow, agents_num):
    
    Inventory = Agents_inventory
    Selected = np.random.choice(np.arange(0,agents_num,1),np.int(agents_num/100)) #选出占总数1/100个agents
    Inventory[Selected] = Inventory[Selected] + moneyflow/100 #增加他们的cash
    
    return Inventory
   
def AgentsChange(Agents_inventory, in_num, out_num, cash_add):
    
    Inventory = Agents_inventory
    Drop_Index = Inventory.sample(out_num).index # 选出减少的agents的编号
    Inventory = Inventory.drop(Drop_Index) # 删除减少agents对应的cash明细
    
    Increase_Inventory = pd.Series(data = cash_add * np.ones(in_num)) # 新增的agents对应的cash
    
    Temp_Inventory = pd.concat([Inventory,Increase_Inventory]) # 将agents减少后的列表与新增agents列表合并
    Final_Inventory = pd.Series(data = Temp_Inventory.values) # 重新建立列表
    
    return Final_Inventory

def linearfunc(x, a, b):
    return a * x + b

def expfunc_normal(x, a, b, c):
    return a * np.exp(b * x) + c

def expfunc_inverse(x, a, b, c):
    return a * np.exp(-b * x) + c
    
def PlotExponentialDistribution(Data,steps_name,mu1=0.12,mu2=0.15,mu3=0.65):
    fig = plt.figure(figsize=(8,4),dpi=150)
    ax1 = fig.add_subplot(111)
    font1 = {'family':'Times New Roman', 'weight':'normal', 'size':10}
    
    n, x_bins, patches = ax1.hist(Data, bins=300, density=1, facecolor='green', alpha=0.5, label='Cash Distribution')  
    
    end_1 = np.int(mu1*len(x_bins))
    start_2 = np.int(mu2*len(x_bins))
    end_2 = np.int(mu3*len(x_bins))
    
    coefs_1, errors_1 = scipy.optimize.curve_fit(expfunc_normal, x_bins[:end_1], n[:end_1], p0=[0.01,0.01,0.01])
    coefs_2, errors_2 = scipy.optimize.curve_fit(expfunc_inverse, x_bins[start_2:end_2], n[start_2:end_2], p0=[0.01,0.004,0.01])
    
    ax1.plot(x_bins[:end_1], expfunc_normal(x_bins[:end_1], *coefs_1), 'r--', label='Normal Exponential Curve')
    ax1.plot(x_bins[start_2:end_2], expfunc_inverse(x_bins[start_2:end_2], *coefs_2), 'r--', label='Inverse Exponential Curve')
    ax1.vlines(100, ymin=0, ymax=1, transform=ax1.get_xaxis_transform(), colors='black', linewidth=0.5)
    
    ax1.grid(False)
    ax1.legend(loc='upper right', prop=font1)
    ax1.set_xlabel('Cash $m$', fontdict = font1)  
    ax1.set_ylabel('Probability $P(m)$', fontdict = font1)  
    ax1.set_xlim(0,3000)
    ax1.set_ylim(0,0.0040)
    
    
    plt.axes([0.67, 0.45, .2, .2])
    coefs_3, errors_3 = scipy.optimize.curve_fit(linearfunc, x_bins[:end_1], np.log(n[:end_1]), p0=[0.2,1])
    coefs_4, errors_4 = scipy.optimize.curve_fit(linearfunc, x_bins[start_2:100], np.log(n[start_2:100]), p0=[-0.05,-1])
      
    plt.scatter(x_bins[:end_1], np.log(n[:end_1]), s=0.2, color='green', alpha=0.6, zorder=1)
    plt.scatter(x_bins[start_2:end_2], np.log(n[start_2:end_2]), s=0.2, color='green', alpha=0.6, zorder=2)
    
    plt.plot(x_bins[:end_1], linearfunc(x_bins[:end_1], *coefs_3), 'r--', label='Fit Curve',linewidth = 0.6, zorder=3)
    plt.plot(x_bins[start_2:end_2], linearfunc(x_bins[start_2:end_2], *coefs_4), 'r--', label='Fit Curve',linewidth = 0.6, zorder=4) 
    
    plt.xlim(0,1500)
    plt.ylim(-10,-5)
    plt.xlabel('Cash $m$', fontsize='x-small')
    plt.ylabel('Log $P(m)$', fontsize='x-small')
    
    plt.savefig('Opened_Symmetry'+str(steps_name)+'.jpg')

def PlotGaussainDistribution(Data,steps_name):
    fig = plt.figure(figsize=(8,4),dpi=150)
    ax1 = fig.add_subplot(111)
    font1 = {'family':'Times New Roman', 'weight':'normal', 'size':10}
    
    mu = Data.mean()
    sigma = Data.std()
    
    n_2, x_bins_2, patches_2 = ax1.hist(Data, bins=200, density=1, facecolor='green', alpha=0.5, label='Cash Distribution')  
    y_2 = mlab.normpdf(x_bins_2, mu, sigma)#画一条逼近的曲线  
    
    ax1.plot(x_bins_2, y_2, 'r--', label='Gaussian Distribution Curve')  
    ax1.vlines(100, ymin=0, ymax=1, transform=ax1.get_xaxis_transform(), colors='black', linewidth=0.5)
    ax1.grid(False)
    ax1.legend(loc='upper right', prop=font1)
    ax1.set_xlabel('Cash $m$', fontdict = font1)  
    ax1.set_ylabel('Probability $P(m)$', fontdict = font1)  
    ax1.set_xlim(0,1200)
    ax1.set_ylim(0,0.0090)
    plt.savefig('Opened_Asymmetry'+str(steps_name)+'.jpg') 


#######################################################################################################

#######################################################################################################

#0. Set up the paratmeters


M = 5*100000 # initial amount of cash
N = 5000 # initial number of agetns
steps = 3001
count = 0


#1. Initialize data
Agents_Inventory_1 = pd.Series(data = np.ones(N)*M/N) #每个人拥有的现金
Agents_Inventory_2 = pd.Series(data = np.ones(N)*M/N) #每个人拥有的现金
Flow_In = pd.Series(data = M*np.random.rand(steps)/100) #流入现金流，提前预设

Agents_In = pd.Series(data = 2 * np.random.randint(0,high=30,size=steps)) # 新增agents的数量，强制设定为偶数
Agents_Out = pd.Series(data = 2 * np.random.randint(0,high=30,size=steps)) # 减少agents的数量，强制设定为偶数
Agents_Change = pd.concat([Agents_In,Agents_Out],axis=1)
Agents_Change.columns = ['IN','OUT']


'''开放系统中，无论交易规则是否满足时间反演对称性，稳定时的分布都不是玻尔兹曼分布'''
'''但是对称性破缺的系统，贫困能被消除'''



#2. Loop
for s in range(steps):
    np.random.seed(s)
    '''
    if (s+1)%300 == 0:        
        PlotDistribution(Agents_Inventory_1,s,name='Symmetry',c='orange',xlimit=5000,ylimit=130,bins_num=400)
        PlotDistribution(Agents_Inventory_2,s,name='Asymmetry',c='yellowgreen',xlimit=2000,ylimit=70,bins_num=300)
    '''
    if s > 0 and s%500 == 0:
        PlotExponentialDistribution(Agents_Inventory_1, s, mu1=0.12, mu2 =(0.15-0.008*count),mu3=0.65)
        PlotGaussainDistribution(Agents_Inventory_2, s)
        count += 1
    
    Agents_Inventory_1 = SelectAgents(Agents_Inventory_1, Flow_In[s], Agents_Inventory_1.count())
    Agents_Inventory_2 = SelectAgents(Agents_Inventory_2, Flow_In[s], Agents_Inventory_2.count())
    
    Cash_Add_1 = Agents_Inventory_1.sum() / Agents_Inventory_1.count()
    Cash_Add_2 = Agents_Inventory_2.sum() / Agents_Inventory_2.count()
    
    Agents_Inventory_1 = AgentsChange(Agents_Inventory_1, Agents_Change['IN'][s], Agents_Change['OUT'][s], Cash_Add_1) 
    Agents_Inventory_2 = AgentsChange(Agents_Inventory_2, Agents_Change['IN'][s], Agents_Change['OUT'][s], Cash_Add_2) 

    Agents_Inventory_1 = TwoBodyTransactions(Agents_Inventory_1, Agents_Inventory_1.count(), mode=1)
    Agents_Inventory_2 = TwoBodyTransactions(Agents_Inventory_2, Agents_Inventory_2.count(), mode=2) 



#对称交易
'''
n_1, x_bins_1, patches_1 = plt.hist(Agents_Inventory_1, bins=300, normed=1, facecolor='green', alpha=0.5, label='Cash Distribution'+' Steps '+str(3000))  

start = np.int(0.10 * len(x_bins_1))

end = np.int(0.65*len(x_bins_1))

plt.plot(x_bins_1[start:end],n_1[start:end])  
'''

