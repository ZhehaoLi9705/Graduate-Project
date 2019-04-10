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
    
    Total_tax = 0.0
    
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
    
    elif mode == 2:  #这样的交换不满足时间反演对称性（类似于税收）
        CashFlow = Mu * Group_2           #生成每组交换现金流
        new_Group_1 = Group_1 + CashFlow  #更新胜利组的资金
        new_Group_2 = Group_2 - CashFlow  #更新失败组的资金
        
    elif mode == 3:  #对每笔交易施加税收
        Tax_Rate = 0.05
        Cash_avg = (Group_1 + Group_2)/2    
        CashFlow = Mu * Cash_avg           #生成每组交换现金流
        Mask = Group_2 > CashFlow          #判断失败租的agents的资金是否满足交易条件    
        new_Group_1[Mask] = Group_1[Mask] + (1-Tax_Rate) * CashFlow[Mask]  #更新胜利组的资金
        new_Group_2[Mask] = Group_2[Mask] - CashFlow[Mask]  #更新失败组的资金
        Total_tax = Tax_Rate * CashFlow[Mask].sum()
    
    Temp = pd.concat([new_Group_1,new_Group_2]) #合并胜利组与失败组
    new_Inventory = pd.Series(data = Temp.values) #建立新的inventory
    
    return new_Inventory,Total_tax

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

def Tax_CashRedistribution(Agents_inventory, tax_rate=0.05): #总资产扣除？？？？？  而且物理上好像没法做到？？
    Inventory = Agents_inventory
    
    Tax_Inventory = tax_rate * Inventory
    Total_tax = Tax_Inventory.sum()
    N = Tax_Inventory.count()
    
    Inventory = Inventory - Tax_Inventory
    Inventory = Inventory + Total_tax/N  #国家的税收进行国防，基础设施，交通的投资，对于每个人的福利是同样的
    
    return Inventory


'''税收可以破坏时间反演对称性
1. 对每笔收入扣税并不破坏 时间反演对称性，因此无法促进财富均衡分配，消除贫困人口；
2. 对总资金扣税才会破坏 时间反演对称性，能消除贫困人口'''
def linearfunc(x, a, b):
    return -a * x - b

def expfunc(x, a, b, c):
    return a * np.exp(-b * x) + c

def PlotExponentialDistribution_Isolated(Data, steps_name):
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
    ax1.set_xlim(0,1000)
    ax1.set_ylim(0,0.0150)
    
    plt.axes([0.67, 0.5, .2, .2])
    coefs_2, errors_2 = scipy.optimize.curve_fit(linearfunc, x_bins_1[:100], np.log(n_1[:100]), p0=[0.2,1])
    plt.scatter(x_bins_1[:-1],np.log(n_1), s=0.2, color='green', alpha=0.6, zorder=1)
    plt.plot(x_bins_1[:-1], linearfunc(x_bins_1[:-1], *coefs_2), 'r--', label='Fit Curve',linewidth = 0.6, zorder=2) 
    plt.xlim(0,500)
    plt.ylim(-10,-4)
    plt.xlabel('Cash $m$', fontsize='x-small')
    plt.ylabel('Log $P(m)$', fontsize='x-small')

    plt.savefig('Isolated_Symmetry'+str(steps_name)+'.jpg')

def PlotGaussainDistribution_Isolated(Data, steps_name):
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
    ax1.set_ylim(0,0.040)
    plt.savefig('Isolated_Asymmetry'+str(steps_name)+'.jpg')

def expfunc_normal(x, a, b, c):
    return a * np.exp(b * x) + c

def expfunc_inverse(x, a, b, c):
    return a * np.exp(-b * x) + c
    

def PlotExponentialDistribution_Opened(Data,steps_name,mu1=0.12,mu2=0.15,mu3=0.65):
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
    ax1.set_ylim(0,0.0060)
    
    
    plt.axes([0.67, 0.45, .2, .2])
    coefs_3, errors_3 = scipy.optimize.curve_fit(linearfunc, x_bins[:end_1], np.log(n[:end_1]), p0=[0.2,1])
    coefs_4, errors_4 = scipy.optimize.curve_fit(linearfunc, x_bins[start_2:100], np.log(n[start_2:100]), p0=[-0.05,-1])
      
    plt.scatter(x_bins[:end_1], np.log(n[:end_1]), s=0.2, color='green', alpha=0.6, zorder=1)
    plt.scatter(x_bins[start_2:end_2], np.log(n[start_2:end_2]), s=0.2, color='green', alpha=0.6, zorder=2)
    
    plt.plot(x_bins[:end_1], linearfunc(x_bins[:end_1], *coefs_3), 'r--', label='Fit Curve',linewidth = 0.6, zorder=3)
    plt.plot(x_bins[start_2:end_2], linearfunc(x_bins[start_2:end_2], *coefs_4), 'r--', label='Fit Curve',linewidth = 0.6, zorder=4) 
    
    plt.xlim(0,2000)
    plt.ylim(-10,-5)
    plt.xlabel('Cash $m$', fontsize='x-small')
    plt.ylabel('Log $P(m)$', fontsize='x-small')
    
    plt.savefig('Opened_Symmetry'+str(steps_name)+'.jpg')




def PlotGaussainDistribution_Opened(Data,steps_name):

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
    ax1.set_xlim(0,1500)
    ax1.set_ylim(0,0.0150)
    plt.savefig('Opened_Asymmetry'+str(steps_name)+'.jpg') 




#######################################################################################################

#######################################################################################################

#0. Set up the paratmeters




'''税收都会破坏系统的微观&宏观时间反演对称性'''

M = 5*100000
N = 5000
steps = 3001

#1. Initialize
Agents_Inventory_1 = pd.Series(data = np.ones(N)*M/N)
Agents_Inventory_2 = pd.Series(data = np.ones(N)*M/N)
Agents_Inventory_3 = pd.Series(data = np.ones(N)*M/N)
Agents_Inventory_4 = pd.Series(data = np.ones(N)*M/N)
Flow_In = pd.Series(data = M*np.random.rand(steps)/100) #流入现金流，提前预设

Total_Tax_1 = 0.0
Total_Tax_2 = 0.0
Total_Tax_3 = 0.0
Total_Tax_4 = 0.0

Agents_In = pd.Series(data = 2 * np.random.randint(0,high=30,size=steps)) # 新增agents的数量，强制设定为偶数
Agents_Out = pd.Series(data = 2 * np.random.randint(0,high=30,size=steps)) # 减少agents的数量，强制设定为偶数
Agents_Change = pd.concat([Agents_In,Agents_Out],axis=1)
Agents_Change.columns = ['IN','OUT']

count_opened = 0

Agents_Inventory_1_Dict = {}
Agents_Inventory_2_Dict = {}

'''开放系统中，无论交易规则是否满足时间反演对称性，稳定时的分布都不是玻尔兹曼分布'''
'''但是对称性破缺的系统，贫困能被消除'''


#2. Loop
'''开放系统'''

for s in range(steps):
    np.random.seed(s)
    '''
    if s%300 == 0:
        PlotDistribution(Agents_Inventory_1,s,name='Symmetry',c='orange',xlimit=3000,ylimit=130,bins_num=400)
        PlotDistribution(Agents_Inventory_2,s,name='Asymmetry',c='yellowgreen',xlimit=1500,ylimit=70,bins_num=300)
    '''
    if s > 0 and s%500 == 0:
        Agents_Inventory_1_Dict[str(s)] = Agents_Inventory_1
        Agents_Inventory_2_Dict[str(s)] = Agents_Inventory_2
        #PlotExponentialDistribution_Opened(Agents_Inventory_1, s, mu1=0.12, mu2 =(0.16-0.005*count_opened),mu3=0.65)
        #PlotGaussainDistribution_Opened(Agents_Inventory_2, s)
        count_opened += 1
    
        
    '''对称性征税''' 
    #资金注入
    Agents_Inventory_1 = SelectAgents(Agents_Inventory_1, Flow_In[s], Agents_Inventory_1.count())
    #人员流动
    Cash_Add_1 = Agents_Inventory_1.sum() / Agents_Inventory_1.count()  #注入平均资金
    Agents_Inventory_1 = AgentsChange(Agents_Inventory_1, Agents_Change['IN'][s], Agents_Change['OUT'][s], Cash_Add_1)
    #交易
    Agents_Inventory_1,Total_Tax_1 = TwoBodyTransactions(Agents_Inventory_1, Agents_Inventory_1.count(), mode=3)
    Agents_Inventory_1 = Agents_Inventory_1 + Total_Tax_1/Agents_Inventory_1.count() #对称性保持，对收入征税
    
    '''非对称性征税'''
    #资金注入
    Agents_Inventory_2 = SelectAgents(Agents_Inventory_2, Flow_In[s], Agents_Inventory_2.count())
    #人员流动
    Cash_Add_2 = Agents_Inventory_2.sum() / Agents_Inventory_2.count()
    Agents_Inventory_2 = AgentsChange(Agents_Inventory_2, Agents_Change['IN'][s], Agents_Change['OUT'][s], Cash_Add_2) 
    #交易
    Agents_Inventory_2,Total_Tax_2 = TwoBodyTransactions(Agents_Inventory_2, Agents_Inventory_2.count(), mode=1) 
    Agents_Inventory_2 = Tax_CashRedistribution(Agents_Inventory_2) #破坏时间反演对称性，对财产征税


PlotExponentialDistribution_Opened(Agents_Inventory_1_Dict['500'], steps_name=500, mu1=0.12, mu2 = 0.16, mu3=0.65)

PlotExponentialDistribution_Opened(Agents_Inventory_1_Dict['1000'], steps_name=1000, mu1=0.12, mu2 = 0.155, mu3=0.65)

PlotExponentialDistribution_Opened(Agents_Inventory_1_Dict['1500'], steps_name=1500, mu1=0.12, mu2 = 0.15, mu3=0.65)

PlotExponentialDistribution_Opened(Agents_Inventory_1_Dict['2000'], steps_name=2000, mu1=0.12, mu2 = 0.145, mu3=0.65)

PlotExponentialDistribution_Opened(Agents_Inventory_1_Dict['2500'], steps_name=2500, mu1=0.12, mu2 = 0.15, mu3=0.65)

PlotExponentialDistribution_Opened(Agents_Inventory_1_Dict['3000'], steps_name=3000, mu1=0.12, mu2 =0.165, mu3=0.65)



PlotGaussainDistribution_Opened(Agents_Inventory_2_Dict['500'], steps_name=500)

PlotGaussainDistribution_Opened(Agents_Inventory_2_Dict['1000'], steps_name=1000)

PlotGaussainDistribution_Opened(Agents_Inventory_2_Dict['1500'], steps_name=1500)

PlotGaussainDistribution_Opened(Agents_Inventory_2_Dict['2000'], steps_name=2000)

PlotGaussainDistribution_Opened(Agents_Inventory_2_Dict['2500'], steps_name=2500)

PlotGaussainDistribution_Opened(Agents_Inventory_2_Dict['3000'], steps_name=3000)





'''孤立系统'''
for s in range(steps):
    np.random.seed(s)
    '''
    if s%300 == 0:
        PlotDistribution(Agents_Inventory_3,s,name='Symmetry',c='orange',xlimit=1000,ylimit=300,bins_num=200)
        PlotDistribution(Agents_Inventory_4,s,name='Asymmetry',c='yellowgreen',xlimit=300,ylimit=200,bins_num=200)
    '''
    if s > 0 and s%500 == 0:
        PlotExponentialDistribution_Isolated(Agents_Inventory_3, s)
        PlotGaussainDistribution_Isolated(Agents_Inventory_4, s)
    
    
    '''对称性征税''' 
    #对收入征税，指数分布，对调整收入分布作用很小，依然有贫困人口
    Agents_Inventory_3,Total_Tax_3 = TwoBodyTransactions(Agents_Inventory_3, N, mode=3)
    Agents_Inventory_3 = Agents_Inventory_3 + Total_Tax_3 / Agents_Inventory_3.count() 

    '''非对称性征税'''
    #对财产征税； 正态分布， 对调整收入分布有交大作用，能消除贫困
    Agents_Inventory_4,Total_Tax_4 = TwoBodyTransactions(Agents_Inventory_4, N, mode=1)
    Agents_Inventory_4 = Tax_CashRedistribution(Agents_Inventory_4) 
    

    
    
    
    
    
    