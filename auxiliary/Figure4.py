#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import localreg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels as sm

import econtools
import econtools.metrics as mt


# In[2]:


##====================[Female Labor Supply]========================
ls=pd.read_stata("data/data_lfs_20110196.dta")


# In[3]:


# Control variables
ls['m2']=ls['m']*ls['m']

# No father present
ls['nodad']=0
ls.loc[ls['dadid']==0, 'nodad']=1

# Mother not married 
ls['smom']=0
ls.loc[ls['eciv']!=2, 'smom']=1

# Single mother
ls['single']=0
ls.loc[ls['eciv']==1, 'single']=1

# Seprated or Divorced mother 
ls['sepdiv']=0
ls.loc[ls['eciv']==4, 'sepdiv']=1

# No partner in the household
ls['nopart']=0
ls.loc[ls['partner']==0, 'nopart']=1


# In[4]:


##==================Probability of the mother being in the maternity leave period at the time of the interview
ls['pleave']=0

ls.loc[(ls['q']==1) & (ls['m']==2)|(ls['q']==2) & (ls['m']==5)|(ls['q']==3) & (ls['m']==8)|(ls['q']==4) & (ls['m']==11) ,'pleave']=0.17
ls.loc[((ls['q']==1) & (ls['m']==3)) | ((ls['q']==2) & (ls['m']==6))  | ((ls['q']==3) & (ls['m']==9)) |((ls['q']==4) & (ls['m']==12)), 'pleave'] = 0.5
ls.loc[((ls['q']==1) & (ls['m']==4)) | ((ls['q']==2) & (ls['m']==7)) | ((ls['q']==3) & (ls['m']==10))  | ((ls['q']==4) & (ls['m']==13)), 'pleave'] = 0.83
ls.loc[((ls['q']==1) & (ls['m']>4) & (ls['m']<9)) | ((ls['q']==2) & (ls['m']>7) & (ls['m']<12)) | ((ls['q']==3) & (ls['m']>10) & (ls['m']<15))| ((ls['q']==4) & (ls['m']>13)), 'pleave'] = 1


# In[5]:


# Create iq dummies: iq_1 for quarter 1, iq_2 for quarter 2, and so on
for j in range(1,5):
    ls['iq_'+str(j)] = 0
    for i in range(len(ls)):
        if ls.loc[i,'q'] == j:
            ls.loc[i, 'iq_'+str(j)] = 1

# Create interaction dummies
ls['ipost_1']=ls['post']*ls['m']
ls['ipost_2'] =ls['post']*ls['m2']


# In[6]:


gbm_ls=ls.groupby(['m'], as_index=False).agg({'work':'mean', 'work2':'mean','post':'mean','m2':'mean','ipost_1':'mean','ipost_2':'mean','age':'mean','age2':'mean',
                                              'age3':'mean', 'immig':'mean','primary':'mean','hsgrad':'mean','univ':'mean','sib':'mean',
                                              'pleave':'mean','iq_2':'mean','iq_3':'mean','iq_4':'mean'})


# In[7]:


#===============================[Regression Discontinuity Design] Working Last Week ===============================#
pX_ls =['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
result_ls=mt.reg(gbm_ls[(gbm_ls['m']> -30) & (gbm_ls['m']< 12)], 'work', pX_ls, addcons=True)
predict_work = result_ls.yhat


# In[8]:


predict_plot_ls = pd.DataFrame(predict_work)
predict_plot_ls.rename(columns={0:'predict_work'}, inplace=True)


# In[10]:


for i in range(0,41):
    gbm_ls.loc[gbm_ls['m']== -29+i, 'predict_work']= predict_plot_ls.iloc[i,0]


# In[11]:


gbm_ls.loc[gbm_ls['m']>=12, 'predict_work'] = np.nan


# In[12]:


def plot_RRD_curve_work(data):
    plt.grid(True)
    
    ls_p1 = gbm_ls[['predict_work','m']]
    ls_plot = ls_p1.dropna()
    ls_untreat = ls_plot[ls_plot['m'] < 0]
    ls_treat = ls_plot[ls_plot['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),ls_untreat['predict_work'], 2))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,12,1),ls_treat['predict_work'], 2))
    plt.plot(np.arange(0,12,1), poly_fit(np.arange(0,12,1)), c='green',linestyle='-')    
    
    plt.xlim(-30, 10, 10)
    plt.ylim(0.2, 0.6, 0.1)
    
    return


# In[13]:


def plot_figure4(data):
    work_plot = gbm_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
    plt.xlim(-30, 10, 10)
    plt.ylim(0.2, 0.6, 0.1)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Proportion working by month of birth, LFS 2008')
    plt.plot(gbm_ls.m, gbm_ls.work, 'o')
    plt.grid(True)
    plot_RRD_curve_work(data)

    plt.title("Figure 4. Proportion working by month of birth, LFS 2008")
    return

