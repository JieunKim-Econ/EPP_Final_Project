#!/usr/bin/env python
# coding: utf-8

# In[7]:


import json
import localreg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels as sm

import econtools
import econtools.metrics as mt


# In[8]:


#==================Female Labor Supply==================
cls=pd.read_stata("data/data_lfs_20110196.dta")


# In[9]:


# Control variables
cls['m2']=cls['m']*cls['m']

# No father present
cls['nodad']=0
cls.loc[cls['dadid']==0, 'nodad']=1

# Mother not married 
cls['smom']=0
cls.loc[cls['eciv']!=2, 'smom']=1

#Mother single
cls['single']=0
cls.loc[cls['eciv']==1, 'single']=1

#Mother separated or divorced
cls['sepdiv']=0
cls.loc[cls['eciv']==4, 'sepdiv']=1

# No partner in the household
cls['nopart']=0
cls.loc[cls['partner']==0, 'nopart']=1


# Married mom
cls['married']=0
cls.loc[cls['eciv']==2, 'married']=1


#Create interaction dummies
cls['ipost_1']=cls['post']*cls['m']

gbm_cls=cls.groupby(['m'], as_index=False).agg({'age':'mean','immig':'mean','married':'mean','univ':'mean','post':'mean','ipost_1':'mean'})
gbm_cls.loc[gbm_cls['immig']==0, 'immig']= np.NaN


# In[10]:


#==================Regression to check Balance in Covariate
##==================Figure 2-1. Age of Mother

cov_X = ['age','immig','married','univ','post','m','ipost_1']
cov_index = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]

result_age=mt.reg(cov_index[(gbm_cls['m']> -30) & (gbm_cls['m']< 20)], 'age', ['post','m','ipost_1'], addcons=True)
predict_age = result_age.yhat.round(0)


# In[11]:


predict_plot1 = pd.DataFrame(predict_age)
predict_plot1.rename(columns={0:'predict_age'}, inplace=True)


# In[12]:


for i in range(0,47):
    gbm_cls.loc[gbm_cls['m']==-29+i, 'predict_age']= predict_plot1.iloc[i,0]


# In[13]:


def plot_cov_age(data):
    plt.grid(True)
    
    cls1 = gbm_cls[['predict_age','m']]
    age_plot = cls1.dropna()
    age_untreat = age_plot[age_plot['m'] < 0]
    age_treat = age_plot[age_plot['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),age_untreat['predict_age'], 1))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),age_treat['predict_age'], 1))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
    
    plt.xlim(-30, 10, 10)
    plt.ylim(30, 36, 2)
    
    return


# In[14]:


def plot_figure1_age(data):
    cov_plot = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]
    plt.xlim(-30, 10, 10)
    plt.ylim(30, 36, 2)
    plt.axvline(x=0, color='r')
    #plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Average age of mother')
    plt.plot(gbm_cls.m, gbm_cls.age, 'o')
    plt.grid(True)
    plot_cov_age(data)

    #plt.title("Figure 2-1. Average age of the mother by month of birth")
    return


# In[15]:


##==================Figure 2-2. Fraction foreign mothers by month of birth
cov_X = ['age','immig','married','univ','post','m','ipost_1']
cov_index = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]

result_imm=mt.reg(cov_index[(gbm_cls['m']> -30) & (gbm_cls['m']< 20)], 'immig', ['post','m'], addcons=True)
predict_immig = result_imm.yhat.round(4)

predict_plot2 = pd.DataFrame(predict_immig)
predict_plot2.rename(columns={0:'predict_immig'}, inplace=True)


# In[16]:


for i in range(0,46):
    gbm_cls.loc[gbm_cls['m']==-29+i, 'predict_immig']= predict_plot2.iloc[i,0]


# In[17]:


def plot_cov_immig(data):
    plt.grid(True)
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 0.4, 0.1)
    
    cls_imm = gbm_cls[['predict_immig','m']]
    imm_plot = cls_imm.dropna()
    imm_untreat = imm_plot[imm_plot['m'] < 0]
    imm_treat = imm_plot[imm_plot['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),imm_untreat['predict_immig'], 1))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,17,1),imm_treat['predict_immig'], 1))
    plt.plot(np.arange(0,17,1), poly_fit(np.arange(0,17,1)), c='green',linestyle='-')    
    
    
    return


# In[18]:


def plot_figure2_immig(data):
    cov_plot = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]
    plt.xlim(-30, 20,10)
    plt.ylim(0, 0.4, 0.1)
    
    plt.axvline(x=0, color='r')
    #plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Fraction Foreign Mothers')
    plt.plot(gbm_cls.m, gbm_cls.immig, 'o')
    plt.grid(True)
    plot_cov_immig(gbm_cls)

    #plt.title("Figure 2-2. Fraction foreign mothers by month of birth")
    return


# In[19]:


##==================Figure 2-3. Fraction married mothers by month of birth
cov_X = ['age','immig','married','univ','post','m','ipost_1']
cov_index = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]

result_mar=mt.reg(cov_index[(gbm_cls['m']> -30) & (gbm_cls['m']< 20)], 'married', ['post','m'], addcons=True)
predict_married = result_mar.yhat.round(4)

predict_plot3 = pd.DataFrame(predict_married)
predict_plot3.rename(columns={0:'predict_married'}, inplace=True)


# In[20]:


for i in range(0,47):
    gbm_cls.loc[gbm_cls['m']==-29+i, 'predict_married']= predict_plot3.iloc[i,0]


# In[21]:


def plot_cov_married(data):
    plt.grid(True)
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 1, 0.2)
    
    cls_mar = gbm_cls[['predict_married','m']]
    ls_plot3 = cls_mar.dropna()
    mar_untreat = ls_plot3[ls_plot3['m'] < 0]
    mar_treat = ls_plot3[ls_plot3['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),mar_untreat['predict_married'], 1))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),mar_treat['predict_married'], 1))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
      
    return


# In[22]:


def plot_figure3_married(data):
    cov_plot = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 1, 0.2)
     
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')

    plt.plot(gbm_cls.m, gbm_cls.married, 'o')
    plt.grid(True)
    plot_cov_married(data)

    return


# In[23]:


##==================Figure 2-4: Fraction mothers with university degree by month of birth
cov_X = ['age','immig','married','univ','post','m','ipost_1']
cov_index = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]

result_univ=mt.reg(cov_index[(gbm_cls['m']> -30) & (gbm_cls['m']< 20)], 'univ', ['post','m','ipost_1'], addcons=True)
predict_univ = result_univ.yhat.round(2)

predict_plot4 = pd.DataFrame(predict_univ)
predict_plot4.rename(columns={0:'predict_univ'}, inplace=True)


# In[24]:


for i in range(0,47):
    gbm_cls.loc[gbm_cls['m']==-29+i, 'predict_univ']= predict_plot4.iloc[i,0]


# In[25]:


def plot_cov_univ(data):
    plt.grid(True)
    plt.xlim(-30, 20, 10)
    plt.ylim(0.1, 0.5, 0.1)
    
    cls_uni = gbm_cls[['predict_univ','m']]
    ls_plot4 = cls_uni.dropna()
    uni_untreat = ls_plot4[ls_plot4['m'] < 0]
    uni_treat = ls_plot4[ls_plot4['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),uni_untreat['predict_univ'], 1))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),uni_treat['predict_univ'], 1))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
      
    return


# In[26]:


def plot_figure4_uni(data):
    cov_plot = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0.1, 0.5, 0.1)
     
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    plt.plot(gbm_cls.m, gbm_cls.univ.round(3), 'o')
    plt.grid(True)
    plot_cov_univ(data)

    
    return


# In[27]:


def plot_figure2(data): 
    plt.figure(figsize=(10,8), dpi= 80)
    plt.subplot(2,2,1)
    plot_figure1_age(data)
    plot_cov_age(data)
    plt.title("Average age of the mother by month of birth",fontsize = 10)

    
    plt.subplot(2,2,2)
    plot_figure2_immig(data)
    plot_cov_immig(data)
    plt.title("Fraction foreign mothers by month of birth",fontsize = 10)
    
    
    plt.subplot(2,2,3)
    plot_figure3_married(data)
    plot_cov_married(data)
    plt.title("Fraction married mothers by month of birth",fontsize = 10)

    
    plt.subplot(2,2,4)
    plot_figure4_uni(data)
    plot_cov_univ(data)
    plt.title("Fraction mothers with university degree by month of birth",fontsize = 10)
    
    plt.suptitle('Figure 2. Balance in Covariates', verticalalignment='bottom', fontsize=14)    
    
    return

