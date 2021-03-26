#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels as sm

import econtools
import econtools.metrics as mt


# In[2]:


##====================[Household Expenditure]========================
# Control variables
# Characteristics: Age of mom and dad, Education of mom and dad, Immigrant mom, Single mom, Mom or Dad NOT PRESENT

dc=pd.read_stata("data/data_hbs_20110196.dta")


# In[3]:


# Age of mom and dad

dc['agemom'].fillna(0, inplace=True)
dc['agedad'].fillna(0, inplace=True)


# In[4]:


# Mom or dad not present

del dc['nomom']
dc['nomom'] = 0
dc.loc[dc['agemom']==0, 'nomom'] = 1

del dc['nodad']
dc['nodad'] = 0
dc.loc[dc['agedad']==0, 'nodad'] = 1


# In[5]:


# Education of mom and dad

dc['sec1mom']=0
dc['sec1dad']=0
dc['sec2mom']=0
dc['sec2dad']=0
dc['unimom']=0
dc['unidad']=0

dc.loc[dc['educmom']==3, 'sec1mom'] = 1
dc.loc[dc['educdad']==3, 'sec1dad'] = 1

dc.loc[(dc['educmom']>3)&(dc['educmom']<7), 'sec2mom'] = 1
dc.loc[(dc['educdad']>3)&(dc['educdad']<7), 'sec2dad'] = 1

dc.loc[(dc['educmom']==7)|(dc['educmom']==8), 'unimom'] = 1
dc.loc[(dc['educdad']==7)|(dc['educdad']==8), 'unidad'] = 1


# In[6]:


# Immigrant

dc['immig']=0
dc.loc[(dc['nacmom']==2) | (dc['nacmom']==3), 'immig'] = 1


# In[7]:


# Mom not married

dc['smom'] = 0
dc.loc[dc['ecivmom']!=2, 'smom'] = 1


# In[8]:


# Siblings

dc['sib']=0
dc.loc[dc['nmiem2']>1, 'sib'] = 1

dc['age2']=dc['agemom']*dc['agemom']
dc['age3']=dc['agemom']*dc['agemom']*dc['agemom']

dc['daycare_bin']=0
dc.loc[(dc['m_exp12312']>0) &(dc['m_exp12312']!=np.nan), 'daycare_bin'] = 1


# In[9]:


bim_dc=dc.groupby(['month'], as_index=False).agg({'gastmon':'mean', 'c_m_exp':'mean','m_exp12312':'mean','daycare_bin':'mean','post':'mean','nomom':'mean','agemom':'mean','sec1mom':'mean',
                                                       'sec2mom':'mean','unimom':'mean','immig':'mean','sib':'mean'})
bim_dc['month2'] = bim_dc['month']*bim_dc['month']
bim_dc['age2']=bim_dc['agemom']*bim_dc['agemom']
bim_dc['age3']=bim_dc['agemom']*bim_dc['agemom']*bim_dc['agemom']


# In[19]:


# Create Bimonthly daycare expenditure

for i in range (0,24):
    bim_dc.loc[(bim_dc['month'] ==-29+(2*i)), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-29+(2*i),'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-30+(2*i),'m_exp12312'].sum())/2


# In[20]:


#Create Bimonthly "Binary" daycare expenditure

for i in range (0,24):
    bim_dc.loc[(bim_dc['month'] ==-29+(2*i)), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-29+(2*i),'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-30+(2*i),'daycare_bin'].sum())/2


# In[21]:


# Create interaction dummies
bim_dc['ipost_1']=bim_dc['post']*bim_dc['month']
bim_dc['ipost_2'] =bim_dc['post']*bim_dc['month2']


# In[22]:


#====================Figure 5-1. Day care expenditure by month of birth
##====================[RDD: Regression]: Private Childcare========================
X_dce = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']
dce_result=mt.reg(bim_dc[(bim_dc['month']> -30) & (bim_dc['month']< 20)], 'bim_dce', X_dce, addcons=True)
predict_dce = dce_result.yhat.round(0)    


# In[23]:


predict_intg=predict_dce.astype(int)
predict_dce = pd.DataFrame(predict_intg)
predict_dce.rename(columns={0:'predict_bim_dce'}, inplace=True)


# In[24]:


for i in range(0,24):
    bim_dc.loc[bim_dc['month']==-29+(2*i), 'predict_bim_dce']= predict_dce.iloc[i,0]


# In[25]:


def plot_RRD_curve_dce(data):
    
    plt.grid(True)
    p_dce = bim_dc[['predict_bim_dce','month']]
    plot_dce = p_dce.dropna()
    
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 1000, 200)
       
    untreat_dce = plot_dce[plot_dce['month'] < 0]
    m, b = np.polyfit(np.arange(-29,0,2),untreat_dce['predict_bim_dce'], 1)
    plt.plot(np.arange(-29,0,2), m*np.arange(-29,0,2) + b, color='orange')
    
    treat_dce = plot_dce[plot_dce['month'] >= 0]
    m, b = np.polyfit(np.arange(0,18,2),treat_dce['predict_bim_dce'], 1)
    plt.plot(np.arange(0,18,2), m*np.arange(0,18,2) + b, color='green')   
 
    return


# In[26]:


def plot_figure5_dce(data):
    dce_plot = bim_dc[['predict_bim_dce','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 1000, 200)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Day care expenditure')
    plt.plot(bim_dc.month, bim_dc.bim_dce, 'o')
    plt.grid(True)
    plot_RRD_curve_dce(data)

    plt.title("Figure 5-1 Day care expenditure by month of birth")
    return


# In[27]:


#===========================Figure 5-2. Fraction with positive day care expenditure by month of birth
##=========================[RDD: Regression]: BINARY Private childcare ========================
X_bdce = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']
bdce_result=mt.reg(bim_dc[(bim_dc['month']> -30) & (bim_dc['month']< 20)], 'bim_bindce', X_bdce, addcons=True)
predict_bin_dce = bdce_result.yhat.round(2)    


# In[28]:


predict_bindce = pd.DataFrame(predict_bin_dce)
predict_bindce.rename(columns={0:'predict_bim_bindce'}, inplace=True)


# In[29]:


for i in range(0,24):
    bim_dc.loc[bim_dc['month']==-29+(2*i), 'predict_bim_bindce']= predict_bindce.iloc[i,0]


# In[30]:


def plot_RRD_curve_bindce(data):
    
    plt.grid(True)
    p_bdce = bim_dc[['predict_bim_bindce','month']]
    plot_bdce = p_bdce.dropna()
    
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 0.7, 0.1)
       
    untreat_bdce = plot_bdce[plot_bdce['month'] < 0]
    m, b = np.polyfit(np.arange(-29,0,2),untreat_bdce['predict_bim_bindce'], 1)
    plt.plot(np.arange(-29,0,2), m*np.arange(-29,0,2) + b, color='orange')
    
    treat_bdce = plot_bdce[plot_bdce['month'] >= 0]
    m, b = np.polyfit(np.arange(0,18,2),treat_bdce['predict_bim_bindce'], 1)
    plt.plot(np.arange(0,18,2), m*np.arange(0,18,2) + b, color='green')   
 
    return


# In[31]:


def plot_figure5_bdce(data):
    bdce_plot = bim_dc[['predict_bim_dce','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
    plt.xlim(0, 0.7, 0.1)
    plt.ylim(0, 1000, 200)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    plt.plot(bim_dc.month, bim_dc.bim_bindce, 'o')
    plt.grid(True)
    plot_RRD_curve_bindce(data)

    plt.title("Figure 5-2 Fraction with positive day care expenditure by month of birth")
    return


# In[32]:


def plot_figure5(data):   
    plt.figure(figsize=(13, 4))
    plt.subplot(1, 2, 1)
    plot_figure5_dce(data)
    plot_RRD_curve_dce(data)
    plt.title("Day care expenditure by month of birth",fontsize = 13)

    
    plt.subplot(1, 2, 2)
    plot_figure5_bdce(data)
    plot_RRD_curve_bindce(data)
    plt.title("Fraction with positive day care expenditure by month of birth",fontsize = 13)
    plt.suptitle("Figure 5. Day Care Expenditure by Month of Birth (HBS 2008)", verticalalignment='bottom', fontsize=14)   
    return

