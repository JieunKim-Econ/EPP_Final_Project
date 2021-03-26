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


##====================[Household Expenditure]========================
# Control variables
# Characteristics: Age of mom and dad, Education of mom and dad, Immigrant mom, Single mom, Mom or Dad NOT PRESENT

plot_he=pd.read_stata("data/data_hbs_20110196.dta")


# In[3]:


# Age of mom and dad
plot_he['agemom'].fillna(0, inplace=True)
plot_he['agedad'].fillna(0, inplace=True)


# In[4]:


# Mom or Dad NOT PRESENT

del plot_he['nomom']
plot_he['nomom'] = 0
plot_he.loc[plot_he['agemom']==0, 'nomom'] = 1


# In[5]:


del plot_he['nodad']
plot_he['nodad'] = 0
plot_he.loc[plot_he['agedad']==0, 'nodad'] = 1


# In[6]:


# Education of mom and dad
plot_he['sec1mom']=0
plot_he['sec1dad']=0
plot_he['sec2mom']=0
plot_he['sec2dad']=0
plot_he['unimom']=0
plot_he['unidad']=0

plot_he.loc[plot_he['educmom']==3, 'sec1mom'] = 1
plot_he.loc[plot_he['educdad']==3, 'sec1dad'] = 1

plot_he.loc[(plot_he['educmom']>3)&(plot_he['educmom']<7), 'sec2mom'] = 1
plot_he.loc[(plot_he['educdad']>3)&(plot_he['educdad']<7), 'sec2dad'] = 1

plot_he.loc[(plot_he['educmom']==7)|(plot_he['educmom']==8), 'unimom'] = 1
plot_he.loc[(plot_he['educdad']==7)|(plot_he['educdad']==8), 'unidad'] = 1


# In[7]:


# Immigrant
plot_he['immig']=0
plot_he.loc[(plot_he['nacmom']==2) | (plot_he['nacmom']==3), 'immig'] = 1


# In[8]:


# Mom not married
plot_he['smom'] = 0
plot_he.loc[plot_he['ecivmom']!=2, 'smom'] = 1


# In[9]:


# Siblings
plot_he['sib']=0
plot_he.loc[plot_he['nmiem2']>1, 'sib'] = 1

plot_he['age2']=plot_he['agemom']*plot_he['agemom']
plot_he['age3']=plot_he['agemom']*plot_he['agemom']*plot_he['agemom']

plot_he['daycare_bin']=0
plot_he.loc[(plot_he['m_exp12312']>0) &(plot_he['m_exp12312']!=np.nan), 'daycare_bin'] = 1


# In[10]:


gbm_he=plot_he.groupby(['month'], as_index=False).agg({'gastmon':'mean', 'c_m_exp':'mean','m_exp12312':'mean','post':'mean','nomom':'mean','agemom':'mean','sec1mom':'mean',
                                                       'sec2mom':'mean','unimom':'mean','immig':'mean','sib':'mean'})
gbm_he['month2'] = gbm_he['month']*gbm_he['month']
gbm_he['age2']=gbm_he['agemom']*gbm_he['agemom']


# In[11]:


#=====================Figure 3-1. Total Expenditure
#Create interaction dummies
gbm_he['ipost_1']=gbm_he['post']*gbm_he['month']
gbm_he['ipost_2'] =gbm_he['post']*gbm_he['month2']

##====================[Regression]========================
pX_he = ['month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2'] 
result_he=mt.reg(gbm_he[(gbm_he['month']>-30) & (gbm_he['month']< 20)], 'gastmon', pX_he, addcons=True)
predict_gast = result_he.yhat.round(0)
predict_gast_int=predict_gast.astype(int)


# In[12]:


predict_plot_gast = pd.DataFrame(predict_gast_int)
predict_plot_gast.rename(columns={0:'predict_gast'}, inplace=True)


# In[33]:


for i in range (0,47):
    gbm_he.loc[gbm_he['month']== -29+i, 'predict_gast']= predict_plot_gast.iloc[i,0]


# In[34]:


def plot_RRD_curve_tot(data):
    plt.grid(True)
    
    he_p1 = gbm_he[['predict_gast','month']]
    he_plot = he_p1.dropna()
    he_untreat = he_plot[he_plot['month'] < 0]
    he_treat = he_plot[he_plot['month'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),he_untreat['predict_gast'], 3))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),he_treat['predict_gast'], 3))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
    
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 60000, 10000)
 
    return


# In[35]:


def plot_figure3_tot(data):
    tot_plot = gbm_he[['predict_gast','month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 60000, 10000)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0=July 2007)')
    plt.plot(gbm_he.month, gbm_he.gastmon, 'o')
    plt.grid(True)
    plot_RRD_curve_tot(data)
    return


# In[36]:


#==================Figure 3-2. Child-related expenditure
##====================[Regression]========================
pX_he = ['month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2'] 
result_cre=mt.reg(gbm_he[(gbm_he['month']>-30) & (gbm_he['month']< 20)], 'c_m_exp',pX_he, addcons=True)
predict_cre = result_cre.yhat.round(0)    


# In[37]:


predict_cre_int=predict_cre.astype(int)
predict_plot_cre = pd.DataFrame(predict_cre_int)
predict_plot_cre.rename(columns={0:'predict_cre'}, inplace=True)


# In[38]:


for i in range (0,47):
    gbm_he.loc[gbm_he['month']== -29+i, 'predict_cre']= predict_plot_cre.iloc[i,0]


# In[39]:


def plot_RRD_curve_cre(data):
    plt.grid(True)
    
    cre_p = gbm_he[['predict_cre','month']]
    cre_plot = cre_p.dropna()
    cre_untreat = cre_plot[cre_plot['month'] < 0]
    cre_treat = cre_plot[cre_plot['month'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),cre_untreat['predict_cre'], 2))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),cre_treat['predict_cre'], 2))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
    
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 10000, 2000)
 
    return


# In[41]:


def plot_figure3_cre(data):
    cre_plot = gbm_he[['predict_cre','month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 60000, 10000)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0=July 2007)')
    plt.plot(gbm_he.month, gbm_he.c_m_exp, 'o')
    plt.grid(True)
    plot_RRD_curve_cre(data)
    return


# In[42]:


def plot_figure3(data):   
    plt.figure(figsize=(13, 4))
    plt.subplot(1, 2, 1)
    plot_figure3_tot(data)
    plot_RRD_curve_tot(data)
    plt.title("Total expenditure by month of birth",fontsize = 13)

    plt.subplot(1, 2, 2)
    plot_figure3_cre(data)
    plot_RRD_curve_cre(data)
    plt.title("Child-related expenditure by month of birth",fontsize = 13)
    plt.suptitle("Figure 3. Household Expenditure (Annual) by Month of Birth (HBS 2008)", verticalalignment='bottom', fontsize=14)   
    return

