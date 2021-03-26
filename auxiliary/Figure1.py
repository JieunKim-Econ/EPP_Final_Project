#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns

import econtools
import econtools.metrics as mt


# In[2]:


##====================[Conceptions]========================
#Variables
#mesp: Month of birth
#year: Year of birth
#prem: Prematurity indicator
#semanas: Weeks of gestation at birth

df_con=pd.read_stata("data/data_births_20110196.dta")


# In[3]:


# Create month of birth variable: (0 = July 2007, 1 = August 2007, etc)

for i in range(0,11):
    df_con.loc[(df_con['year']==2000+i), 'm'] = df_con['mesp'] - 91 + 12*i


# In[4]:


# Create month of conception variable.
## Naive definition (9 months before the birth month) 

df_con['mc1'] = df_con['m'] - 9


# In[5]:


## Naivee plus prematures (9 months before the birth month, 8 if premature)

df_con['mc2'] = df_con['m'] - 9
df_con.loc[((df_con['semanas'] > 0) & (df_con['semanas'] < 38)) | (df_con['prem']==2), 'mc2'] =df_con['m'] - 8 


# In[6]:


## Sophisticated (calculated based on weeks of gestation)

df_con['mc3'] = df_con['m'] - 9
df_con.loc[((df_con['semanas'] > 43) & (df_con['semanas']!=np.nan)), 'mc3'] = df_con['m'] - 10 
df_con.loc[((df_con['semanas'] < 39) & (df_con['semanas']!=0)) | (df_con['prem']==2), 'mc3'] = df_con['m'] - 8


# In[7]:


# Group by month of conception

df_con=df_con.groupby('mc3', as_index=False)['mc3'].agg({'n':'count'})


# In[8]:


# Group by month of conception.

df_con= df_con.rename(columns={'mc3': 'mc'})


# In[11]:


# Calendar month of conception

df_con['month']=1 
for m in range(0,12):    
    for i in range(0,6):
        df_con.loc[(df_con['mc']== (-30+m)+(12*i)), 'month'] = 1+m
    for j in range(0,3):
        df_con.loc[(df_con['mc']== (-42+m)-(12*j)), 'month'] = 1+m
    for n in range(0,2):
        df_con.loc[(df_con['mc']== (-87+m)-(12*n)), 'month'] = 1+m


# In[12]:


# July indicator 

df_con.loc[df_con['month']==7, 'july'] = df_con['n']


# In[13]:


# Number of days in a month

df_con['days'] = 31
df_con.loc[(df_con['month']==2), 'days'] = 28
df_con.loc[(df_con['mc']==7), 'days'] = 29
df_con.loc[(df_con['month']==4)| (df_con['month']==6)|(df_con['month']==9)|(df_con['month']==11),'days'] = 30


# In[14]:


# A post indicator for post-policy conception

df_con['post']=0
df_con.loc[(df_con['mc']>=0), 'post']=1

df_con['mc2']=df_con['mc']*df_con['mc']
df_con['mc3']=df_con['mc']*df_con['mc']*df_con['mc']


# In[21]:


# Create Bimonthly number of conceptions
for i in range(1,16):
    df_con.loc[(df_con['mc'] == -(2*i-1)), 'bim_n'] = df_con.loc[df_con['mc']== -(2*i-1),'n'].sum() + df_con.loc[df_con['mc']== -2*i,'n'].sum()

for j in range(0,10):
    df_con.loc[(df_con['mc'] == 2*j+1), 'bim_n'] = df_con.loc[df_con['mc']== 2*j+1,'n'].sum() + df_con.loc[df_con['mc']== 2*j,'n'].sum()


# In[22]:


# Regressions Table 
# Create interaction dummies

df_con['ipost_1']=df_con['post']*df_con['mc']
df_con['ipost_2'] =df_con['post']*df_con['mc2']
df_con['ipost_3'] =df_con['post']*df_con['mc3']


# In[23]:


X =  ['mc','mc2','mc3','post', 'ipost_1', 'ipost_2', 'ipost_3', 'days']
results=mt.reg(df_con[(df_con['mc']> -30) & (df_con['mc']< 20)], 'bim_n',X, addcons=True)
predict_bim_n = results.yhat.round(0)


# In[24]:


predict_bim_n
predict_int=predict_bim_n.astype(int)


# In[25]:


predict_df = pd.DataFrame(predict_int)
predict_df.rename(columns={0:'predict_bim_n'}, inplace=True)


# In[26]:


for i in range (0,25):
    df_con.loc[df_con['mc']==-29+(2*i), 'predict_bim_n']= predict_df.iloc[i,0]


# In[27]:


def plot_RRD_curve_con(data):
    
    plt.pyplot.grid(True)
    df_p_c = df_con[['predict_bim_n','mc']]
    df_plot_c = df_p_c.dropna()
    
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(70000, 95000, 5000)
       
    df_untreat_c = df_plot_c[df_plot_c['mc'] < 0]
    m, b = np.polyfit(np.arange(-29,0,2),df_untreat_c['predict_bim_n'], 1)
    plt.pyplot.plot(np.arange(-29,0,2), m*np.arange(-29,0,2) + b, color='orange')
    
    df_treat_c = df_plot_c[df_plot_c['mc'] >= 0]
    m, b = np.polyfit(np.arange(1,20,2),df_treat_c['predict_bim_n'], 1)
    plt.pyplot.plot(np.arange(1,20,2), m*np.arange(1,20,2) + b, color='green')   
 
    return


# In[28]:


df_p = df_con[['predict_bim_n','mc']]
df_plot = df_p.dropna()
df_treat = df_plot[df_plot['mc'] >= 0]


# In[29]:


def plot_figure1_con(data):
    df_plot = df_con[['predict_bim_n','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(70000, 95000, 5000)
    plt.pyplot.axvline(x=0, color='r')
    plt.pyplot.xlabel('Month of conception (0=July 2007)', fontsize = 11)
    #plt.pyplot.ylabel('Number of conception')
    plt.pyplot.plot(df_con.mc, df_con.bim_n, 'o')
    plt.pyplot.grid(True)    
    plot_RRD_curve_con(data)

    plt.pyplot.title("Figure 1. Number of conceptions by month, 2005-2009")
    return


# In[30]:


##====================[Abortions]========================


# In[31]:


df_abo=pd.read_stata("data/data_abortions_20110196.dta")


# In[32]:


# Sum abortions across all regions
df_abo['n_tot']=pd.concat([df_abo.n_ive_and, df_abo.n_ive_val, df_abo.n_ive_rioja, df_abo.n_ive_cat, df_abo.n_ive_can,df_abo.n_ive_mad, df_abo.n_ive_gal, df_abo.n_ive_bal, df_abo.n_ive_pv, df_abo.n_ive_castlm, df_abo.n_ive_ast, df_abo.n_ive_arag],1).sum(1)


# In[33]:


# Create month variable that takes value 0 in July 2007.
df_abo['m']= df_abo.reset_index().index + 1 
df_abo['m'] = df_abo['m'] - 103


# In[34]:


# Generate a variable indicating number of days in a month

df_abo['days']=31
df_abo.loc[(df_abo['month']==4) | (df_abo['month']==6) | (df_abo['month']==9) | (df_abo['month']==11), 'days'] = 30
df_abo.loc[(df_abo['month']==2), 'days'] = 28
df_abo.loc[(df_abo['month']==2) & ((df_abo['year']==2000) | (df_abo['year']==2004) | (df_abo['year']==2008)), 'days'] = 29


# In[35]:


# Squared and cubed terms in m & Create Post dummy
df_abo['m2'] = df_abo['m']*df_abo['m']
df_abo['m3'] = df_abo['m']*df_abo['m']*df_abo['m']

df_abo['post']=0
df_abo.loc[(df_abo['m']>=0), 'post']=1


# In[36]:


# Restrict period

df_abo = df_abo[~(df_abo.m < -90) & ~(df_abo.m > 29)]


# In[38]:


# Create Bimonthly number of conceptions

df_abo['bim_ab']=np.nan    
for i in range (0,25):
    df_abo.loc[(df_abo['m'] ==-29+(2*i)), 'bim_ab'] = df_abo.loc[df_abo['m']==-29+(2*i),'n_tot'].sum() + df_abo.loc[df_abo['m']==-30+(2*i),'n_tot'].sum()


# In[39]:


# Regressions Table 
# Create interaction dummies
df_abo['ipost_1']=df_abo['post']*df_abo['m']
df_abo['ipost_2'] =df_abo['post']*df_abo['m2']
df_abo['ipost_3'] =df_abo['post']*df_abo['m3']

X_abo = ['m','m2','post','ipost_1', 'ipost_2', 'days']
result=mt.reg(df_abo[(df_abo['m']> -30)&(df_abo['m']<20)], 'bim_ab', X_abo, addcons=True)
predict_bim_ab = result.yhat.round(0)


# In[40]:


predict_bim_ab
predict_ab_int=predict_bim_ab.astype(int)


# In[41]:


predict_df_ab = pd.DataFrame(predict_ab_int)
predict_df_ab.rename(columns={0:'predict_bim_ab'}, inplace=True)


# In[43]:


for i in range (0,25):
    df_abo.loc[df_abo['m']==-29+(2*i), 'predict_bim_ab']= predict_df_ab.iloc[i,0]


# In[44]:


def plot_figure1_ab(data):
    df_plot = df_abo[['predict_bim_ab','m','m2','m3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(10000, 22000, 2000)
    plt.pyplot.axvline(x=0, color='r')
    plt.pyplot.xlabel('Month of abortion (0=July 2007)')
    plt.pyplot.plot(df_abo.m, df_abo.bim_ab, 'o')
    plt.pyplot.grid(True)

    plt.pyplot.title("Figure 1-2. Number of abortions by month, 2005-2009")
    return


# In[45]:


def plot_RRD_curve_ab(data):
    plt.pyplot.grid(True)
    
    df_p1 = df_abo[['predict_bim_ab','m']]
    df_plot = df_p1.dropna()
       
    df_untreat = df_plot[df_plot['m'] < 0]
    m, b = np.polyfit(np.arange(-29,0,2),df_untreat['predict_bim_ab'], 1)
    plt.pyplot.plot(np.arange(-29,0,2), m*np.arange(-29,0,2) + b, color='orange')

    
    df_treat = df_plot[df_plot['m'] >= 0]        
    m, b = np.polyfit(np.arange(1,20,2),df_treat['predict_bim_ab'], 1)
    plt.pyplot.plot(np.arange(1,20,2), m*np.arange(1,20,2) + b, color='green')
    
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(10000, 22000, 2000)
 
    return


# In[46]:


def plot_figure1_ab(data):
    df_plot = df_abo[['predict_bim_ab','m','m2','m3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(10000, 22000, 2000)
    plt.pyplot.axvline(x=0, color='r')
    plt.pyplot.xlabel('Month of abortion (0=July 2007)', fontsize = 11)
    
    plt.pyplot.plot(df_abo.m, df_abo.bim_ab, 'o')
    plt.pyplot.grid(True)
    plot_RRD_curve_ab(df_abo)

    plt.pyplot.title("Figure 1-2. Number of abortions by month, 2005-2009")
    return


# In[47]:


def plot_figure1(data, data2):   
    plt.pyplot.figure(figsize=(13, 4))
    plt.pyplot.subplot(1, 2, 1)
    plot_figure1_con(data)
    plot_RRD_curve_con(data)
    plt.pyplot.title('Number of conceptions by month 2005-2009',fontsize = 13)
    
    plt.pyplot.subplot(1, 2, 2)
    plot_figure1_ab(data2)
    plot_RRD_curve_ab(data2)
    plt.pyplot.title('Number of abortions by month 2005-2009',fontsize = 13)
    plt.pyplot.suptitle("Figure 1. Fertility Effect: Conceptions and Abortions by Month", verticalalignment='bottom', fontsize=14)   
    return

