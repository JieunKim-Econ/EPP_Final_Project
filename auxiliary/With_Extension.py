#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels as sm
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

import econtools
import econtools.metrics as mt

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load Female Labor Supply
et_ls=pd.read_stata("data/data_lfs_20110196.dta")


# In[3]:


# Control variables
et_ls['m2']=et_ls['m']*et_ls['m']

# No father present
et_ls['nodad']=0
et_ls.loc[et_ls['dadid']==0, 'nodad']=1

# Mother not married 
et_ls['smom']=0
et_ls.loc[et_ls['eciv']!=2, 'smom']=1

#Mother single
et_ls['single']=0
et_ls.loc[et_ls['eciv']==1, 'single']=1

# Mother separated or divorced
et_ls['sepdiv']=0
et_ls.loc[et_ls['eciv']==4, 'sepdiv']=1

# No partner in the household
et_ls['nopart']=0
et_ls.loc[et_ls['partner']==0, 'nopart']=1


# In[4]:


# Probability of the mother being in the maternity leave period at the time of the interview

et_ls['pleave']=0

et_ls.loc[(et_ls['q']==1) & (et_ls['m']==2)|(et_ls['q']==2) & (et_ls['m']==5)|(et_ls['q']==3) & (et_ls['m']==8)|(et_ls['q']==4) & (et_ls['m']==11) ,'pleave']=0.17
et_ls.loc[((et_ls['q']==1) & (et_ls['m']==3)) | ((et_ls['q']==2) & (et_ls['m']==6))  | ((et_ls['q']==3) & (et_ls['m']==9)) |((et_ls['q']==4) & (et_ls['m']==12)), 'pleave'] = 0.5
et_ls.loc[((et_ls['q']==1) & (et_ls['m']==4)) | ((et_ls['q']==2) & (et_ls['m']==7)) | ((et_ls['q']==3) & (et_ls['m']==10))  | ((et_ls['q']==4) & (et_ls['m']==13)), 'pleave'] = 0.83
et_ls.loc[((et_ls['q']==1) & (et_ls['m']>4) & (et_ls['m']<9)) | ((et_ls['q']==2) & (et_ls['m']>7) & (et_ls['m']<12)) | ((et_ls['q']==3) & (et_ls['m']>10) & (et_ls['m']<15))| ((et_ls['q']==4) & (et_ls['m']>13)), 'pleave'] = 1


# In[5]:


#=======================Regression & Figure Setup
# Create interaction dummies
et_ls['ipost_1']=et_ls['post']*et_ls['m']
et_ls['ipost_2'] =et_ls['post']*et_ls['m2']

# Create iq dummies: iq_1 for quarter 1, iq_2 for quarter 2, and so on
for j in range(1,5):
    et_ls['iq_'+str(j)] = 0
    for i in range(len(et_ls)):
        if et_ls.loc[i,'q'] == j:
            et_ls.loc[i, 'iq_'+str(j)] = 1


# In[6]:


def figure_ET6(data):
    features1=list(['primary','hsgrad','univ','primary_dad','hsgrad_dad','univ_dad','sib','nodad' ,'smom','single' ,'sepdiv', 'nopart'])
    et_ls[features1].corr()

    # Plot
    plt.figure(figsize=(12,9), dpi= 80)
    sns.heatmap(et_ls[features1].corr().corr(), xticklabels=et_ls[features1].corr(), yticklabels=et_ls[features1].corr().corr(), cmap='RdYlGn', center=0, annot=True)

    # Decorations
    plt.title('Correlogram of parents education level and marital status', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return


# In[7]:


def figure_ET3(data):
    x_var = 'm'
    groupby_var = 'eciv'
    df_agg = et_ls.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [et_ls[x_var].values.tolist() for i, et_ls in df_agg]

    # Draw
    plt.figure(figsize=(13,7), dpi= 80)
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, 50, stacked=True, density=False, color=colors[:len(vals)])

    # Decoration
    plt.legend({group:col for group, col in zip(np.unique(et_ls[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of marital status", fontsize=20)
    plt.xlabel("m: month of birth (0=July 2007)")
    plt.ylabel("Frequency")
    plt.ylim(0, 1000)
    plt.show()

    return


# In[8]:


def figure_ET4(data):
    plt.figure(figsize=(13,6), dpi= 80)
    sns.distplot(et_ls.loc[et_ls['univ'] == 1, "m"], color="dodgerblue", label="univ", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    sns.distplot(et_ls.loc[et_ls['primary'] == 1, "m"], color="orange", label="primary", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    sns.distplot(et_ls.loc[et_ls['hsgrad'] == 1, "m"], color="salmon", label="hsgrad", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    plt.ylim(0, 0.05)

    # Decoration
    plt.title('Density Plot of Mother Education Attainment', fontsize=20)
    plt.legend()
    plt.show()
    return


# In[9]:


def figure_ET5(data):
    # Draw Plot
    plt.figure(figsize=(13,6), dpi= 80)
    sns.distplot(et_ls.loc[et_ls['univ_dad'] == 1, "m"], color="g", label="univ_dad", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    sns.distplot(et_ls.loc[et_ls['primary_dad'] == 1, "m"], color="dodgerblue", label="primary_dad", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    sns.distplot(et_ls.loc[et_ls['hsgrad_dad'] == 1, "m"], color="orange", label="hsgrad_dad", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    plt.ylim(0, 0.05)

    # Decoration
    plt.title('Density Plot of Father Education Attainment', fontsize=20)
    plt.legend()
    plt.show()
    return


# In[10]:


#=======================Part 1
# 1-1. Primary Education

def P_WL_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']

    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True)
    return(result)


# In[11]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
pwl1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True).summary


# In[12]:


# 1-2. Highschool Education

def HS_WL_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True)
    return(result)


# In[13]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
hswl1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True).summary


# In[14]:


# 1-3. University Education
def UNI_WL_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True)
    return(result)


# In[15]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
uniwl1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True).summary


# In[16]:


# 1-4. No partner, No dad
def NPND_WL_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','nodad','nopart','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True)
    return(result)


# In[17]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','nodad','nopart','pleave','iq_2', 'iq_3', 'iq_4']
npndwl1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True).summary


# In[18]:


# 1-5. Single mom
def SG_WL_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','smom','single','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True)
    return(result)


# In[19]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','smom','single','pleave','iq_2', 'iq_3', 'iq_4']
sgwl1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True).summary


# In[20]:


#1-6. Smom, No partner
def NPS_WL_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','smom','nopart','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True)
    return(result)


# In[21]:


X_ls =  ['post','m','m2','ipost_1', 'ipost_2','smom','nopart','pleave','iq_2', 'iq_3', 'iq_4']
npswl1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work',X_ls, addcons=True).summary


# In[22]:


#1-7 No dad, smom
def NDS_WL_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True)
    return(result)


# In[23]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
ndswl1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True).summary


# In[24]:


#1-8 No partner, sepdiv
def NPDIV_WL_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', X_ls, addcons=True)
    return(result)


# In[25]:


X_ls =  ['post','m','m2','ipost_1', 'ipost_2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
npdivwl1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work',X_ls, addcons=True).summary


# In[26]:


#2-1. Primary Education
def P_WL_2(et_ls):
    X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True)
    return(result)


# In[27]:


X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
pwl2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True).summary


# In[28]:


#2-2. Highschool Education
def HS_WL_2(et_ls):
    X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True)
    return(result)


# In[29]:


X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
hswl2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True).summary


# In[30]:


#2-3. University Education
def UNI_WL_2(et_ls):
    X_ls =  ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work',X_ls, addcons=True)
    return(result)


# In[31]:


X_ls = ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
uniwl2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True).summary


# In[32]:


#2-4. nopart, nodad 
def NPND_WL_2(et_ls):
    X_ls = ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True)
    return(result)


# In[33]:


X_ls =  ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
npndwl2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True).summary


# In[34]:


#2-5. single, smom 
def SG_WL_2(et_ls):
    X_ls = ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True)
    return(result)


# In[35]:


X_ls = ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
sgwl2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True).summary


# In[36]:


#2-6. nopart, smom 
def NPS_WL_2(et_ls):
    X_ls =  ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work',X_ls, addcons=True)
    return(result)


# In[37]:


X_ls =  ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
npswl2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work',X_ls, addcons=True).summary


# In[38]:


#2-7. nodad, smom 
def NDS_WL_2(et_ls):
    X_ls = ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True)
    return(result)


# In[39]:


X_ls = ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
ndswl2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', X_ls, addcons=True).summary 


# In[40]:


#2-8. nopart, sepdiv 
def NPDIV_WL_2(et_ls):
    X_ls =  ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work',X_ls, addcons=True)
    return(result)


# In[41]:


X_ls =  ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
npdivwl2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work',X_ls, addcons=True).summary


# In[42]:


#3-1. Primary Education
def P_WL_3(et_ls):
    X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True)
    return(result)


# In[43]:


X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
pwl3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True).summary 


# In[44]:


#3-2. Highschool Education
def HS_WL_3(et_ls):
    X_ls =  ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work',X_ls, addcons=True)
    return(result)


# In[45]:


X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
hswl3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True).summary


# In[46]:


#3-3. University Education
def UNI_WL_3(et_ls):
    X_ls = ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True)
    return(result)


# In[47]:


X_ls =  ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
uniwl3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work',X_ls, addcons=True).summary


# In[48]:


#3-4. No partner, No dad 
def NPND_WL_3(et_ls):
    X_ls = ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True)
    return(result)


# In[49]:


X_ls = ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
npndwl3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True).summary


# In[50]:


#3-5. Single mom, smom 
def SG_WL_3(et_ls):
    X_ls = ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True)
    return(result)


# In[51]:


X_ls = ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
sgwl3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True).summary


# In[52]:


#3-6. No partner, smom
def NPS_WL_3(et_ls):
    X_ls = ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True)
    return(result)


# In[53]:


X_ls =  ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
npswl3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True).summary


# In[54]:


#3-7. No dad, smom 
def NDS_WL_3(et_ls):
    X_ls =  ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work',X_ls, addcons=True)
    return(result)


# In[55]:


X_ls = ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
ndswl3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True).summary 


# In[56]:


#3-8. No partner, sepdiv 
def NPDIV_WL_3(et_ls): 
    X_ls = ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', X_ls, addcons=True)
    return(result)


# In[57]:


npdivwl3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[58]:


#4-1. Primary Education
def P_WL_4(et_ls):
    X_ls = ['post','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True)
    return(result)


# In[59]:


X_ls =['post','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
pwl4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True).summary


# In[60]:


#4-2. Highschool Education
def HS_WL_4(et_ls):
    X_ls = ['post','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True)
    return(result)


# In[61]:


X_ls = ['post','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
hswl4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True).summary 


# In[62]:


#4-3. University Education
def UNI_WL_4(et_ls):
    X_ls =  ['post','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work',X_ls, addcons=True)
    return(result)


# In[63]:


uniwl4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[64]:


#4-4. No partner, No dad 
def NPND_WL_4(et_ls):
    X_ls = ['post','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True)
    return(result)


# In[65]:


npndwl4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[66]:


#4-5. Single mom, smom 
def SG_WL_4(et_ls):
    X_ls = ['post','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True)
    return(result)


# In[67]:


sgwl4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary 


# In[68]:


#4-6. No partner, smom 
def NPS_WL_4(et_ls):
    X_ls = ['post','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True)
    return(result)


# In[69]:


npswl4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[70]:


#4-7. No dad, smom 
def NDS_WL_4(et_ls):
    X_ls = ['post','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True)
    return(result)


# In[71]:


ndswl4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[72]:


#4-8. No part, sepdiv 
def NPDIV_WL_4(et_ls):
    X_ls = ['post','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', X_ls, addcons=True)
    return(result)


# In[73]:


npdivwl4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[74]:


#5-1. Primary Education
def P_WL_5(et_ls):
    X_ls = ['post','primary','primary_dad']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', X_ls, addcons=True)
    return(result)


# In[75]:


pwl5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','primary','primary_dad'], addcons=True).summary


# In[76]:


#5-2. Highschool Education
def HS_WL_5(et_ls):
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','hsgrad','hsgrad_dad'], addcons=True)
    return(result)


# In[77]:


hswl5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','hsgrad','hsgrad_dad'], addcons=True).summary


# In[78]:


#5-3. University Education
def UNI_WL_5(et_ls):
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','univ','univ_dad'], addcons=True)
    return(result)


# In[79]:


uniwl5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','univ','univ_dad'], addcons=True).summary


# In[80]:


#5-4. No partner, No dad
def NPND_WL_5(et_ls):
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','nodad'], addcons=True)
    return(result)


# In[81]:


npndwl5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','nodad'], addcons=True).summary


# In[82]:


#5-5. Single, smom 
def SG_WL_5(et_ls):
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','single','smom'], addcons=True)
    return(result)


# In[83]:


sgwl5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','single','smom'], addcons=True).summary


# In[84]:


#5-6. No partner, smom 
def NPS_WL_5(et_ls):
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom'], addcons=True)
    return(result)


# In[85]:


npswl5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom'], addcons=True).summary


# In[86]:


#5-7. No dad, smom 
def NDS_WL_5(et_ls):
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nodad','smom'], addcons=True)
    return(result)


# In[87]:


ndswl5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nodad','smom'], addcons=True).summary


# In[88]:


#5-8. No partner, sepdiv 
def NPDIV_WL_5(et_ls):
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','sepdiv'], addcons=True)
    return(result)


# In[89]:


npdivwl5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','sepdiv'], addcons=True).summary


# In[90]:


#6-1. Primary Education
def P_WL_6(et_ls):
    X_ls =  ['post','primary','primary_dad','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', X_ls, addcons=True)
    return(result)


# In[91]:


pwl6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','primary','primary_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[92]:


#6-2. Highschool Education
def HS_WL_6(et_ls):
    X_ls =  ['post','hsgrad','hsgrad_dad','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', X_ls, addcons=True)
    return(result)


# In[93]:


hswl6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','hsgrad','hsgrad_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[94]:


#6-3. University Education
def UNI_WL_6(et_ls):
    X_ls = ['post','univ','univ_dad','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', X_ls, addcons=True)
    return(result)


# In[95]:


uniwl6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','univ','univ_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[96]:


#6-4. No partner, No dad 
def NPND_WL_6(et_ls):
    X_ls = ['post','nopart','nodad','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', X_ls, addcons=True)
    return(result)


# In[97]:


npndwl6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','nodad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[98]:


#6-5. Single, smom 
def SG_WL_6(et_ls):
    X_ls = ['post','single','smom','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', X_ls, addcons=True)
    return(result)


# In[99]:


sgwl6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','single','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[100]:


#6-6. No partner, smom
def NPS_WL_6(et_ls):
    X_ls = ['post','nopart','smom','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work',X_ls, addcons=True)
    return(result)


# In[101]:


npswl6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[102]:


#6-7. No dad, smom 
def NDS_WL_6(et_ls):
    X_ls = ['post','nodad','smom','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', X_ls, addcons=True)
    return(result)


# In[103]:


ndswl6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nodad','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[104]:


#6-8. No partner, sepdiv 
def NPDIV_WL_6(et_ls):
    X_ls = ['post','nopart','sepdiv','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', X_ls, addcons=True)
    return(result)


# In[105]:


npdivwl6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','sepdiv','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[106]:


#=======================DID Setup
# Create i_n_month dummies: in_month_1 for January, in_month_2 for February, and so on
for j in range(1,13):
    et_ls['i_n_month_'+str(j)] = 0
    for i in range(len(et_ls)):
        if et_ls.loc[i,'n_month'] == j:
            et_ls.loc[i, 'i_n_month_'+str(j)] = 1


# In[108]:


#7-1. Primary Education
def P_WL_7(et_ls):
    X_ls_DID = ['post','m','m2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[111]:


X_ls_DID = ['post','m','m2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
pwl7=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True).summary


# In[112]:


#7-2. Highschool Education
def HS_WL_7(et_ls):
    X_ls_DID = ['post','m','m2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[114]:


X_ls_DID =['post','m','m2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
hswl7=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True).summary


# In[115]:


#7-3. Uni Education
def UNI_WL_7(et_ls):
    X_ls_DID = ['post','m','m2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[116]:


X_ls_DID = ['post','m','m2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
uniwl7=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True).summary


# In[117]:


#7-4. No partner, No dad 
def NPND_WL_7(et_ls):
    X_ls_DID =['post','m','m2','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[118]:


X_ls_DID = ['post','m','m2','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
npndwl7=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True).summary


# In[119]:


#7-5.  Single mom, smom
def SG_WL_7(et_ls):
    X_ls_DID = ['post','m','m2','single','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[121]:


X_ls_DID = ['post','m','m2','single','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
sgwl7=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True).summary


# In[122]:


#7-6. No partner, smom
def NPS_WL_7(et_ls):
    X_ls_DID = ['post','m','m2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[123]:


X_ls_DID=['post','m','m2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
npswl7=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True).summary


# In[124]:


#7-7. No dad, smom
def NDS_WL_7(et_ls):
    X_ls_DID = ['post','m','m2','smom','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[125]:


X_ls_DID = ['post','m','m2','smom','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
ndswl7=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True).summary


# In[126]:


#7-8.  No partner, sepdiv
def NPDIV_WL_7(et_ls):
    X_ls_DID = ['post','m','m2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[127]:


X_ls_DID = ['post','m','m2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
npdivwl7=mt.reg(et_ls, 'work', X_ls_DID, cluster='m', addcons=True).summary


# In[128]:


#=======================Part 2
#1-1. Primary Education
def P_EP_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True)
    return(result)


# In[129]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
pep1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True).summary


# In[131]:


#1-2. Highschool Education
def HS_EP_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True)
    return(result)


# In[132]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
hsep1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True).summary


# In[133]:


#1-3. Univ Education
def UNI_EP_1(et_ls): 
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True)
    return(result)


# In[134]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
uniep1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2',  X_ls, addcons=True).summary


# In[135]:


#1-4. No part, No dad
def NPND_EP_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','nodad','nopart','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True)
    return(result)


# In[136]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','nodad','nopart','pleave','iq_2', 'iq_3', 'iq_4']
npndep1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True).summary


# In[137]:


#1-5. Single,smom
def SG_EP_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','smom','single','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True)
    return(result)


# In[138]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','smom','single','pleave','iq_2', 'iq_3', 'iq_4']
sgep1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True).summary


# In[139]:


#1-6. smom, no partner
def NPS_EP_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True)
    return(result)


# In[142]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
npsep1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True).summary


# In[145]:


#1-7. No dad, smom
def NDS_EP_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True)
    return(result)


# In[144]:


X_ls = ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
ndsep1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[146]:


#1-8. No partner, sepdiv
def NPDIV_EP_1(et_ls):
    X_ls = ['post','m','m2','ipost_1', 'ipost_2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True)
    return(result)


# In[147]:


X_ls= ['post','m','m2','ipost_1', 'ipost_2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
npdivep1=mt.reg(et_ls[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', X_ls, addcons=True).summary


# In[148]:


#2-1. Primary
def P_EP_2(et_ls):
    X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True)
    return(result)


# In[149]:


X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
pep2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[150]:


#2-2. Highschool 
def HS_EP_2(et_ls):
    X_ls = ['post','m','ipost_1','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True)
    return(result)


# In[151]:


X_ls = ['post','m','ipost_1','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
hsep2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True).summary


# In[152]:


#2-3. University Education
def UNI_EP_2(et_ls):
    X_ls = ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True)
    return(result)


# In[153]:


X_ls = ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
uniep2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True).summary


# In[157]:


#2-4. no partner, no dad 
def NPND_EP_2(et_ls):
    X_ls = ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)],'work2', X_ls, addcons=True)
    return(result)


# In[158]:


X_ls = ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
npndep2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True).summary


# In[159]:


#2-5 single, smom 
def SG_EP_2(et_ls):
    X_ls = ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True)
    return(result)


# In[161]:


X_ls = ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
sgep2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True).summary


# In[162]:


#2-6.  nopart, smom 
def NPS_EP_2(et_ls):
    X_ls = ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True)
    return(result)


# In[163]:


X_ls = ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
npsep2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[164]:


#2-7. No dad, smom 
def NDS_EP_2(et_ls):
    X_ls = ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True)
    return(result)


# In[165]:


X_ls = ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
ndsep2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True).summary


# In[166]:


#2-8. nopart, sepdiv 
def NPDIV_EP_2(et_ls):
    X_ls = ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True)
    return(result)


# In[167]:


X_ls = ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
npdivep2=mt.reg(et_ls[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', X_ls, addcons=True).summary


# In[168]:


#3-1. primary
def P_EP_3(et_ls):
    X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True)
    return(result)


# In[169]:


X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
pep3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[170]:


#3-2.1 hsgard
def HS_EP_3(et_ls):
    X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2',X_ls, addcons=True)
    return(result)


# In[171]:


X_ls = ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4']
hsep3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True).summary


# In[174]:


#3-3. Uni Education
def UNI_EP_3(et_ls):
    X_ls = ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True)
    return(result)


# In[175]:


X_ls = ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
uniep3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True).summary


# In[176]:


#3-4. No partner, no dad
def NPND_EP_3(et_ls):
    X_ls = ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True)
    return(result)


# In[177]:


X_ls = ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
npndep3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True).summary


# In[178]:


#3-5. single, smom
def SG_EP_3(et_ls):
    X_ls= ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2',X_ls, addcons=True)
    return(result)


# In[179]:


X_ls= ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
sgep3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2',X_ls, addcons=True).summary


# In[180]:


#3-6.  nopart, smom
def NPS_EP_3(et_ls): 
    X_ls = ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True)
    return(result)


# In[181]:


X_ls = ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
npsep3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True).summary


# In[182]:


#3-7. No dad, smom 
def NDS_EP_3(et_ls):
    X_ls= ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True)
    return(result)


# In[183]:


X_ls= ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
ndsep3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True).summary


# In[186]:


#3-8. No partner, sepdiv 
def NPDIV_EP_3(et_ls):
    X_ls = ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True)
    return(result)


# In[187]:


X_ls = ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
npdivep3=mt.reg(et_ls[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', X_ls, addcons=True).summary


# In[188]:


#4-1. Primary
def P_EP_4(et_ls):
    X_ls=['post','primary','primary_dad','pleave','iq_2','iq_3','iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True)
    return(result)


# In[190]:


X_ls=['post','primary','primary_dad','pleave','iq_2','iq_3','iq_4']
pep4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True).summary


# In[191]:


#4-2. Highschool 
def HS_EP_4(et_ls):
    X_ls = ['post','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True)
    return(result)


# In[192]:


X_ls = ['post','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4']
hsep4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True).summary


# In[193]:


#4-3. Univ Education
def UNI_EP_4(et_ls):
    X_ls=['post','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True)
    return(result)


# In[194]:


X_ls=['post','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4']
uniep4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True).summary


# In[195]:


#4-4. nopart, nodad 
def NPND_EP_4(et_ls):
    X_ls=['post','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True)
    return(result)


# In[196]:


X_ls=['post','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4']
npndep4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True).summary


# In[198]:


#4-5. single, smom
def SG_EP_4(et_ls):
    X_ls=['post','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2',  X_ls, addcons=True)
    return(result)


# In[199]:


X_ls=['post','single','smom','pleave','iq_2', 'iq_3', 'iq_4']
sgep4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True).summary


# In[200]:


#4-6. no partner, smom 
def NPS_EP_4(et_ls):
    X_ls=['post','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True)
    return(result)


# In[201]:


X_ls=['post','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4']
npsep4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True).summary


# In[202]:


#4-7. no dad, smom 
def NDS_EP_4(et_ls):
    X_ls= ['post','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True)
    return(result)


# In[203]:


X_ls= ['post','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4']
ndsep4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True).summary


# In[204]:


#4-8 no part, sepdiv 
def NPDIV_EP_4(et_ls):
    X_ls=['post','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True)
    return(result)


# In[205]:


X_ls=['post','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4']
npdivep4=mt.reg(et_ls[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', X_ls, addcons=True).summary


# In[206]:


#5-1. Primary Education
def P_EP_5(et_ls):
    X_ls=['post','primary','primary_dad']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[207]:


X_ls=['post','primary','primary_dad']
pep5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','primary','primary_dad'], addcons=True).summary


# In[208]:


#5-2. Highschool Education
def HS_EP_5(et_ls):
    X_ls = ['post','hsgrad','hsgrad_dad']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[209]:


X_ls = ['post','hsgrad','hsgrad_dad']
hsep5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2',X_ls, addcons=True).summary


# In[210]:


#5-3. University Education
def UNI_EP_5(et_ls):
    X_ls=['post','univ','univ_dad']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[211]:


X_ls=['post','univ','univ_dad']
uniep5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[215]:


#5-4 nopart, nodad 
def NPND_EP_5(et_ls):
    X_ls = ['post','nopart','nodad']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[214]:


X_ls = ['post','nopart','nodad']
npndep5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[216]:


#5-5. single, smom 
def SG_EP_5(et_ls):
    X_ls = ['post','single','smom']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[217]:


X_ls = ['post','single','smom']
sgep5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[218]:


#5-6. no part, smom 
def NPS_EP_5(et_ls):
    X_ls=['post','nopart','smom']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2',  X_ls, addcons=True)
    return(result)


# In[219]:


X_ls=['post','nopart','smom']
npsep5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[220]:


#5-7. no part, smom
def NDS_EP_5(et_ls):
    X_ls= ['post','nopart','smom']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[221]:


X_ls= ['post','nopart','smom']
ndsep5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2',X_ls, addcons=True).summary


# In[222]:


#5-8. no part, sepdiv 
def NPDIV_EP_5(et_ls):
    X_ls=['post','nopart','sepdiv']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2',X_ls, addcons=True)
    return(result)


# In[223]:


X_ls=['post','nopart','sepdiv']
npdivep5=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[224]:


#6-1. Primary Education
def P_EP_6(et_ls):
    X_ls=['post','primary','primary_dad','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[225]:


X_ls=['post','primary','primary_dad','iq_2', 'iq_3', 'iq_4']
pep6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[226]:


#6-2. Highschool Education
def HS_EP_6(et_ls):
    X_ls= ['post','hsgrad','hsgrad_dad','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[227]:


X_ls= ['post','hsgrad','hsgrad_dad','iq_2', 'iq_3', 'iq_4']
hsep6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[228]:


#6-3. Univ Education
def UNI_EP_6(et_ls):
    X_ls=['post','univ','univ_dad','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[229]:


X_ls=['post','univ','univ_dad','iq_2', 'iq_3', 'iq_4']
uniep6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[230]:


#6-4. no part, no dad
def NPND_EP_6(et_ls):
    X_ls=['post','nopart','nodad','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[231]:


X_ls=['post','nopart','nodad','iq_2', 'iq_3', 'iq_4']
npndep6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[232]:


#6-5. single, smom 
def SG_EP_6(et_ls):
    X_ls= ['post','single','smom','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[233]:


X_ls= ['post','single','smom','iq_2', 'iq_3', 'iq_4']
sgep6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[234]:


#6-6 no part, smom 
def NPS_EP_6(et_ls):
    X_ls=['post','nopart','smom','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[235]:


X_ls=['post','nopart','smom','iq_2', 'iq_3', 'iq_4']
npsep6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[236]:


#6-7. no dad, smom 
def NDS_EP_6(et_ls):
    X_ls =['post','nodad','smom','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[237]:


X_ls =['post','nodad','smom','iq_2', 'iq_3', 'iq_4']
ndsep6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[238]:


#6-8. no part, sepdiv 
def NPDIV_EP_6(et_ls):
    X_ls=['post','nopart','sepdiv','iq_2', 'iq_3', 'iq_4']
    result=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True)
    return(result)


# In[239]:


X_ls=['post','nopart','sepdiv','iq_2', 'iq_3', 'iq_4']
npdivep6=mt.reg(et_ls[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', X_ls, addcons=True).summary


# In[240]:


#7-1. Primary Education
def P_EP_7(et_ls): 
    X_ls_DID = ['post','m','m2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[241]:


X_ls_DID = ['post','m','m2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
pep7=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True).summary


# In[242]:


#7-2. Highschool Education
def HS_EP_7(et_ls):
    X_ls_DID=['post','m','m2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[243]:


X_ls_DID=['post','m','m2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
hsep7=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True).summary


# In[244]:


#7-3. Uni Education
def UNI_EP_7(et_ls):
    X_ls_DID = ['post','m','m2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[245]:


X_ls_DID = ['post','m','m2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
uniep7=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True).summary


# In[246]:


#7-4. no part, no dad 
def NPND_EP_7(et_ls):
    X_ls_DID = ['post','m','m2','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work2', X_ls_DID , cluster='m', addcons=True)
    return(result)


# In[247]:


X_ls_DID = ['post','m','m2','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
npndep7=mt.reg(et_ls, 'work2', X_ls_DID , cluster='m', addcons=True).summary


# In[248]:


#7-5. single, smom
def SG_EP_7(et_ls):
    X_ls_DID = ['post','m','m2','single','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[249]:


X_ls_DID = ['post','m','m2','single','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
sgep7=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True).summary


# In[250]:


#7-6. No partner, smom
def NPS_EP_7(et_ls):
    X_ls_DID = ['post','m','m2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[251]:


X_ls_DID = ['post','m','m2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
npsep7=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True).summary


# In[254]:


#7-7. No dad, smom
def NDS_EP_7(et_ls):
    X_ls_DID=['post','m','m2','smom','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[255]:


X_ls_DID=['post','m','m2','smom','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
ndsep7=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True).summary


# In[256]:


#7-8.  No part, sepdiv
def NPDIV_EP_7(et_ls):
    X_ls_DID =['post','m','m2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    result=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True)
    return(result)


# In[257]:


X_ls_DID =['post','m','m2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
npdivep7=mt.reg(et_ls, 'work2', X_ls_DID, cluster='m', addcons=True).summary


# In[258]:


##================Create Table E1
def tableET1(et_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID(7)': []})
        result = ('Working Last Week_Primary', 'Primary_pvalue', 'Primary_SE', 'Working_Highschool', 'HS_pvalue','HS_SE',
                  'Working_Uni', 'Uni_pvalue', 'Uni_SE',
                  'Employed_Primary','E_Primary_pvalue', 'E_Primary_SE','Employed_Highschool',
                  'E_HS_pvalue','E_HS_SE', 'Employed_Uni', 'E_Uni_pvalue','E_Uni_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Education Level'] = result
        table = table.set_index('Education Level')

##=================Working Last Week 
        #1. Working Last Week under Primary educated parents 
        P_RD1 = P_WL_1(et_ls)
        P_RD2 = P_WL_2(et_ls)
        P_RD3 = P_WL_3(et_ls)
        P_RD4 = P_WL_4(et_ls)
        P_RD5 = P_WL_5(et_ls)
        P_RD6 = P_WL_6(et_ls) 
        P_DID7 = P_WL_7(et_ls)
        
        pr = [P_RD1.beta['post'], P_RD2.beta['post'], P_RD3.beta['post'], P_RD4.beta['post'], P_RD5.beta['post'],
                 P_RD6.beta['post'], P_DID7.beta['post']]
        table.loc["Working Last Week_Primary"] = pr
        
        #Primary_pvalue 
        p_pv = [pwl1['p>t']['post'], pwl2['p>t']['post'],pwl3['p>t']['post'], pwl4['p>t']['post'],pwl5['p>t']['post'],
                pwl6['p>t']['post'], pwl7['p>t']['post']]
        table.loc["Primary_pvalue"]=p_pv
        
        #Primary Standard Error
        p_se = [P_RD1.se['post'],P_RD2.se['post'], P_RD3.se['post'], P_RD4.se['post'], P_RD5.se['post'],
                 P_RD6.se['post'], P_DID7.se['post']]
        table.loc["Primary_SE"] = p_se
        
        #2. HS coefficient
        HS_RD1 = HS_WL_1(et_ls)
        HS_RD2 = HS_WL_2(et_ls)
        HS_RD3 = HS_WL_3(et_ls)
        HS_RD4 = HS_WL_4(et_ls)
        HS_RD5 = HS_WL_5(et_ls)
        HS_RD6 = HS_WL_6(et_ls) 
        HS_DID7 = HS_WL_7(et_ls)
        
        hs = [HS_RD1.beta['post'], HS_RD2.beta['post'], HS_RD3.beta['post'], HS_RD4.beta['post'], HS_RD5.beta['post'],
                 HS_RD6.beta['post'], P_DID7.beta['post']]
        table.loc["Working_Highschool"] = hs
        
        #Highschool_pvalue 
        hs_pv = [hswl1['p>t']['post'], hswl2['p>t']['post'],hswl3['p>t']['post'], hswl4['p>t']['post'],hswl5['p>t']['post'],
                hswl6['p>t']['post'], hswl7['p>t']['post']]
        table.loc["HS_pvalue"]=hs_pv
        
        #Highschool Standard Error
        hs_se = [HS_RD1.se['post'],HS_RD2.se['post'], HS_RD3.se['post'], HS_RD4.se['post'], HS_RD5.se['post'],
                 HS_RD6.se['post'], P_DID7.se['post']]
        table.loc["HS_SE"] = hs_se
        
        
        #3. Uni coefficient
        UNI_RD1 = UNI_WL_1(et_ls)
        UNI_RD2 = UNI_WL_2(et_ls)
        UNI_RD3 = UNI_WL_3(et_ls)
        UNI_RD4 = UNI_WL_4(et_ls)
        UNI_RD5 = UNI_WL_5(et_ls)
        UNI_RD6 = UNI_WL_6(et_ls) 
        UNI_DID7 = UNI_WL_7(et_ls)
        
        uni = [UNI_RD1.beta['post'], UNI_RD2.beta['post'], UNI_RD3.beta['post'], UNI_RD4.beta['post'], UNI_RD5.beta['post'],
                 UNI_RD6.beta['post'], P_DID7.beta['post']]
        table.loc["Working_Uni"] = uni
        
        #Uni_pvalue 
        uni_pv = [uniwl1['p>t']['post'], uniwl2['p>t']['post'],uniwl3['p>t']['post'], uniwl4['p>t']['post'],uniwl5['p>t']['post'],
                uniwl6['p>t']['post'], uniwl7['p>t']['post']]
        table.loc["Uni_pvalue"]=uni_pv
        
        #Uni SE 
        uni_se = [UNI_RD1.se['post'],UNI_RD2.se['post'], UNI_RD3.se['post'], UNI_RD4.se['post'], UNI_RD5.se['post'],
                 UNI_RD6.se['post'], P_DID7.se['post']]
        table.loc["Uni_SE"] = uni_se

###=============Employed
        #1. Employed under Primary educated parents 
        P_RD_EP1 = P_EP_1(et_ls)
        P_RD_EP2 = P_EP_2(et_ls)
        P_RD_EP3 = P_EP_3(et_ls)
        P_RD_EP4 = P_EP_4(et_ls)
        P_RD_EP5 = P_EP_5(et_ls)
        P_RD_EP6 = P_EP_6(et_ls) 
        P_DID7 = P_EP_7(et_ls)
        
        pr_ep = [P_RD_EP1.beta['post'], P_RD_EP2.beta['post'], P_RD_EP3.beta['post'], P_RD_EP4.beta['post'], P_RD_EP5.beta['post'],
                 P_RD_EP6.beta['post'], P_DID7.beta['post']]
        table.loc["Employed_Primary"] = pr_ep
        
        #Primary_pvalue 
        p_ep_pv = [pep1['p>t']['post'], pep2['p>t']['post'],pep3['p>t']['post'], pep4['p>t']['post'],pep5['p>t']['post'],
                pep6['p>t']['post'], pep7['p>t']['post']]
        table.loc["E_Primary_pvalue"]=p_ep_pv
        
        #Primary StandaRD_EP Error
        p_ep_se = [P_RD_EP1.se['post'],P_RD_EP2.se['post'], P_RD_EP3.se['post'], P_RD_EP4.se['post'], P_RD_EP5.se['post'],
                 P_RD_EP6.se['post'], P_DID7.se['post']]
        table.loc["E_Primary_SE"] = p_ep_se
        
        #2. HS coefficient
        HS_RD_EP1 = HS_EP_1(et_ls)
        HS_RD_EP2 = HS_EP_2(et_ls)
        HS_RD_EP3 = HS_EP_3(et_ls)
        HS_RD_EP4 = HS_EP_4(et_ls)
        HS_RD_EP5 = HS_EP_5(et_ls)
        HS_RD_EP6 = HS_EP_6(et_ls) 
        HS_DID7 = HS_EP_7(et_ls)
        
        hs_ep = [HS_RD_EP1.beta['post'], HS_RD_EP2.beta['post'], HS_RD_EP3.beta['post'], HS_RD_EP4.beta['post'], HS_RD_EP5.beta['post'],
                 HS_RD_EP6.beta['post'], P_DID7.beta['post']]
        table.loc["Employed_Highschool"] = hs_ep
        
        #Highschool_pvalue (Table5-copy)
        hs_ep_pv = [hsep1['p>t']['post'], hsep2['p>t']['post'],hsep3['p>t']['post'], hsep4['p>t']['post'],hsep5['p>t']['post'],
                hsep6['p>t']['post'], hsep7['p>t']['post']]
        table.loc["E_HS_pvalue"]=hs_ep_pv
        
        #Highschool StandaRD_EP Error
        hs_ep_se = [HS_RD_EP1.se['post'],HS_RD_EP2.se['post'], HS_RD_EP3.se['post'], HS_RD_EP4.se['post'], HS_RD_EP5.se['post'],
                 HS_RD_EP6.se['post'], P_DID7.se['post']]
        table.loc["E_HS_SE"] = hs_ep_se

        #Employed coefficient under college educated paretns
        UNI_EP_RD1 = UNI_EP_1(et_ls)
        UNI_EP_RD2 = UNI_EP_2(et_ls)
        UNI_EP_RD3 = UNI_EP_3(et_ls)
        UNI_EP_RD4 = UNI_EP_4(et_ls)
        UNI_EP_RD5 = UNI_EP_5(et_ls)
        UNI_EP_RD6 = UNI_EP_6(et_ls) 
        UNI_DID7 = UNI_EP_7(et_ls)
        
        uni_ep = [UNI_EP_RD1.beta['post'], UNI_EP_RD2.beta['post'], UNI_EP_RD3.beta['post'], UNI_EP_RD4.beta['post'], UNI_EP_RD5.beta['post'],
                 UNI_EP_RD6.beta['post'], P_DID7.beta['post']]
        table.loc["Employed_Uni"] = uni_ep
        
        #Uni_pvalue 
        uni_ep_pv = [uniep1['p>t']['post'], uniep2['p>t']['post'],uniep3['p>t']['post'], uniep4['p>t']['post'],uniep5['p>t']['post'],
                     uniep6['p>t']['post'], uniep7['p>t']['post']]
        table.loc["E_Uni_pvalue"]=uni_ep_pv
        
        #Uni SE 
        uni_ep_se = [UNI_EP_RD1.se['post'],UNI_EP_RD2.se['post'], UNI_EP_RD3.se['post'], UNI_EP_RD4.se['post'], UNI_EP_RD5.se['post'],
                 UNI_EP_RD6.se['post'], P_DID7.se['post']]
        table.loc["E_Uni_SE"] = uni_ep_se

    
    
    
 #####================================================================================       
        #Observations
        table=table.astype(float).round(4)
        obs =[P_RD1.N, P_RD2.N, P_RD3.N, P_RD4.N, P_RD5.N, P_RD6.N, P_DID7.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48]
        table.loc["Number of months"] = months

        
        return(table)


# In[259]:


##================Create Table E2
def tableET2(et_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID(7)': []})
        result = ('1.NoPartner_NoDad', 'NoPartner_NoDad_pvalue', 'NoPartner_NoDad_SE', 
                  '2.Singlemom', 'Singlemom_pvalue','Singlemom_SE',
                  '3.NoPartner_NotMarried', 'NoPartner_NotMarried_pvalue', 'NoPartner_NotMarried_SE',
                  '4.NoDad_NotMarried', 'NoDad_NotMarried_pvalue', 'NoDad_NotMarried_SE',
                  '5.Divorced_NoPartner','Divorced_NoPartner_pvalue','Divorced_NoPartner_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Marital Status_WLS'] = result
        table = table.set_index('Marital Status_WLS')

##=================Working Last Week 
        #1. Working Last Week under No Part & No Dad
        NPND_RD1 = NPND_WL_1(et_ls)
        NPND_RD2 = NPND_WL_2(et_ls)
        NPND_RD3 = NPND_WL_3(et_ls)
        NPND_RD4 = NPND_WL_4(et_ls)
        NPND_RD5 = NPND_WL_5(et_ls)
        NPND_RD6 = NPND_WL_6(et_ls) 
        NPND_DID7 = NPND_WL_7(et_ls)
        
        npnd = [NPND_RD1.beta['post'], NPND_RD2.beta['post'], NPND_RD3.beta['post'], NPND_RD4.beta['post'], NPND_RD5.beta['post'],
                 NPND_RD6.beta['post'], NPND_DID7.beta['post']]
        table.loc["1.NoPartner_NoDad"] = npnd
        
        #NoPartner_NoDad_pvalue 
        npnd_pv = [npndwl1['p>t']['post'], npndwl2['p>t']['post'],npndwl3['p>t']['post'], npndwl4['p>t']['post'],npndwl5['p>t']['post'],
                npndwl6['p>t']['post'], npndwl7['p>t']['post']]
        table.loc["NoPartner_NoDad_pvalue"]=npnd_pv
        
        #NoPartner_NoDad Standard Error
        npnd_se = [NPND_RD1.se['post'],NPND_RD2.se['post'], NPND_RD3.se['post'], NPND_RD4.se['post'], NPND_RD5.se['post'],
                 NPND_RD6.se['post'], NPND_DID7.se['post']]
        table.loc["NoPartner_NoDad_SE"] = npnd_se
        
        #1-2 Working Last Week under Singlemom
        SG_RD1 = SG_WL_1(et_ls)
        SG_RD2 = SG_WL_2(et_ls)
        SG_RD3 = SG_WL_3(et_ls)
        SG_RD4 = SG_WL_4(et_ls)
        SG_RD5 = SG_WL_5(et_ls)
        SG_RD6 = SG_WL_6(et_ls) 
        SG_DID7 = SG_WL_7(et_ls)
        
        
        SG = [SG_RD1.beta['post'], SG_RD2.beta['post'], SG_RD3.beta['post'], SG_RD4.beta['post'], SG_RD5.beta['post'],
                 SG_RD6.beta['post'], SG_DID7.beta['post']]
        table.loc["2.Singlemom"] = SG
        
        #Singlemom_pvalue 
        SG_pv = [sgwl1['p>t']['post'], sgwl2['p>t']['post'],sgwl3['p>t']['post'], sgwl4['p>t']['post'],sgwl5['p>t']['post'],
                sgwl6['p>t']['post'], sgwl7['p>t']['post']]

        table.loc["Singlemom_pvalue"]=SG_pv
        
        #Singlemom Standard Error
        SG_se = [SG_RD1.se['post'],SG_RD2.se['post'], SG_RD3.se['post'], SG_RD4.se['post'], SG_RD5.se['post'],
                 SG_RD6.se['post'], SG_DID7.se['post']]
        table.loc["Singlemom_SE"] = SG_se
        
        #1-3 Working Last Week under NoPartner_NotMarried
        NPS_RD1 = NPS_WL_1(et_ls)
        NPS_RD2 = NPS_WL_2(et_ls)
        NPS_RD3 = NPS_WL_3(et_ls)
        NPS_RD4 = NPS_WL_4(et_ls)
        NPS_RD5 = NPS_WL_5(et_ls)
        NPS_RD6 = NPS_WL_6(et_ls) 
        NPS_DID7 = NPS_WL_7(et_ls)
        
        NPS = [NPS_RD1.beta['post'], NPS_RD2.beta['post'], NPS_RD3.beta['post'], NPS_RD4.beta['post'], NPS_RD5.beta['post'],
                 NPS_RD6.beta['post'], NPS_DID7.beta['post']]
        table.loc["3.NoPartner_NotMarried"] = NPS
        
        #NoPartner_NotMarriedmom_pvalue 
        NPS_pv = [npswl1['p>t']['post'], npswl2['p>t']['post'],npswl3['p>t']['post'], npswl4['p>t']['post'],npswl5['p>t']['post'],
                npswl6['p>t']['post'], npswl7['p>t']['post']]

        table.loc["NoPartner_NotMarried_pvalue"]=NPS_pv
        
        #NoPartner_NotMarriedmom Standard Error
        NPS_se = [NPS_RD1.se['post'],NPS_RD2.se['post'], NPS_RD3.se['post'], NPS_RD4.se['post'], NPS_RD5.se['post'],
                 NPS_RD6.se['post'], NPS_DID7.se['post']]
        table.loc["NoPartner_NotMarried_SE"] = NPS_se
        
        
        #1-4 Working Last Week under Nodad_NotMarried
        NDS_RD1 = NDS_WL_1(et_ls)
        NDS_RD2 = NDS_WL_2(et_ls)
        NDS_RD3 = NDS_WL_3(et_ls)
        NDS_RD4 = NDS_WL_4(et_ls)
        NDS_RD5 = NDS_WL_5(et_ls)
        NDS_RD6 = NDS_WL_6(et_ls) 
        NDS_DID7 = NDS_WL_7(et_ls)
        
        NDS = [NDS_RD1.beta['post'], NDS_RD2.beta['post'], NDS_RD3.beta['post'], NDS_RD4.beta['post'], NDS_RD5.beta['post'],
                 NDS_RD6.beta['post'], NDS_DID7.beta['post']]
        table.loc["4.NoDad_NotMarried"] = NDS
        
        #NoDad_NotMarried_pvalue 
        NDS_pv = [ndswl1['p>t']['post'], ndswl2['p>t']['post'],ndswl3['p>t']['post'], ndswl4['p>t']['post'],ndswl5['p>t']['post'],
                ndswl6['p>t']['post'], ndswl7['p>t']['post']]

        table.loc["NoDad_NotMarried_pvalue"]=NDS_pv
        
        #NoDad_NotMarried Standard Error
        NDS_se = [NDS_RD1.se['post'],NDS_RD2.se['post'], NDS_RD3.se['post'], NDS_RD4.se['post'], NDS_RD5.se['post'],
                 NDS_RD6.se['post'], NDS_DID7.se['post']]
        table.loc["NoDad_NotMarried_SE"] = NDS_se
        
        #1-5 Working Last Week under Divorced_NoPartner
        NPDIV_RD1 = NPDIV_WL_1(et_ls)
        NPDIV_RD2 = NPDIV_WL_2(et_ls)
        NPDIV_RD3 = NPDIV_WL_3(et_ls)
        NPDIV_RD4 = NPDIV_WL_4(et_ls)
        NPDIV_RD5 = NPDIV_WL_5(et_ls)
        NPDIV_RD6 = NPDIV_WL_6(et_ls) 
        NPDIV_DID7 = NPDIV_WL_7(et_ls)
        
        NPDIV = [NPDIV_RD1.beta['post'], NPDIV_RD2.beta['post'], NPDIV_RD3.beta['post'], NPDIV_RD4.beta['post'], NPDIV_RD5.beta['post'],
                 NPDIV_RD6.beta['post'], NPDIV_DID7.beta['post']]
        table.loc["5.Divorced_NoPartner"] = NPDIV
        
        #Divorced_NoPartnermom_pvalue 
        NPDIV_pv = [npdivwl1['p>t']['post'], npdivwl2['p>t']['post'],npdivwl3['p>t']['post'], npdivwl4['p>t']['post'],npdivwl5['p>t']['post'],
                npdivwl6['p>t']['post'], npdivwl7['p>t']['post']]

        table.loc["Divorced_NoPartner_pvalue"]=NPDIV_pv
        
        #Divorced_NoPartnermom Standard Error
        NPDIV_se = [NPDIV_RD1.se['post'],NPDIV_RD2.se['post'], NPDIV_RD3.se['post'], NPDIV_RD4.se['post'], NPDIV_RD5.se['post'],
                 NPDIV_RD6.se['post'], NPDIV_DID7.se['post']]
        table.loc["Divorced_NoPartner_SE"] = NPDIV_se




##=============================================================================    
        #Observations
        table=table.astype(float).round(4)
        obs =[NPND_RD1.N, NPND_RD2.N, NPND_RD3.N, NPND_RD4.N, NPND_RD5.N, NPND_RD6.N, NPND_DID7.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48]
        table.loc["Number of months"] = months

        
        return(table)


# In[260]:


##================Create Table E2
def tableET3(et_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID(7)': []})
        result = ( '1.NoPartner_NoDad', 'NoPartner_NoDad_pvalue', 'NoPartner_NoDad_SE', 
                  '2.Singlemom', 'Singlemom_pvalue','Singlemom_SE',
                  '3.NoPartner_NotMarried', 'NoPartner_NotMarried_pvalue', 'NoPartner_NotMarried_SE',
                  '4.NoDad_NotMarried', 'NoDad_NotMarried_pvalue', 'NoDad_NotMarried_SE',
                  '5.Divorced_NoPartner','Divorced_NoPartner_pvalue','Divorced_NoPartner_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Marital Status_Employed'] = result
        table = table.set_index('Marital Status_Employed')

##=================Employed 
        #1. Employed under No Part & No Dad
        NPND_RD_EP1 = NPND_EP_1(et_ls)
        NPND_RD_EP2 = NPND_EP_2(et_ls)
        NPND_RD_EP3 = NPND_EP_3(et_ls)
        NPND_RD_EP4 = NPND_EP_4(et_ls)
        NPND_RD_EP5 = NPND_EP_5(et_ls)
        NPND_RD_EP6 = NPND_EP_6(et_ls) 
        NPND_DID_EP7 = NPND_EP_7(et_ls)
        
        ep_npnd = [NPND_RD_EP1.beta['post'], NPND_RD_EP2.beta['post'], NPND_RD_EP3.beta['post'], NPND_RD_EP4.beta['post'], NPND_RD_EP5.beta['post'],
                 NPND_RD_EP6.beta['post'], NPND_DID_EP7.beta['post']]
        table.loc["1.NoPartner_NoDad"] = ep_npnd
        
        #EP_NoPartner_NoDad_pvalue 
        ep_npnd_pv = [npndep1['p>t']['post'], npndep2['p>t']['post'],npndep3['p>t']['post'], npndep4['p>t']['post'],npndep5['p>t']['post'],
                npndep6['p>t']['post'], npndep7['p>t']['post']]

        table.loc["NoPartner_NoDad_pvalue"]=ep_npnd_pv
        
        #EP_NoPartner_NoDad Standard Error
        ep_npnd_se = [NPND_RD_EP1.se['post'],NPND_RD_EP2.se['post'], NPND_RD_EP3.se['post'], NPND_RD_EP4.se['post'], NPND_RD_EP5.se['post'],
                 NPND_RD_EP6.se['post'], NPND_DID_EP7.se['post']]
        table.loc["NoPartner_NoDad_SE"] = ep_npnd_se

        
        #1-2 Employed under EP_Singlemom
        SG_RD_EP1 = SG_EP_1(et_ls)
        SG_RD_EP2 = SG_EP_2(et_ls)
        SG_RD_EP3 = SG_EP_3(et_ls)
        SG_RD_EP4 = SG_EP_4(et_ls)
        SG_RD_EP5 = SG_EP_5(et_ls)
        SG_RD_EP6 = SG_EP_6(et_ls) 
        SG_DID_EP7 = SG_EP_7(et_ls)
        
        
        EP_SG = [SG_RD_EP1.beta['post'], SG_RD_EP2.beta['post'], SG_RD_EP3.beta['post'], SG_RD_EP4.beta['post'], SG_RD_EP5.beta['post'],
                 SG_RD_EP6.beta['post'], SG_DID_EP7.beta['post']]
        table.loc["2.Singlemom"] = EP_SG
        
        #EP_Singlemom_pvalue 
        EP_SG_pv = [sgep1['p>t']['post'], sgep2['p>t']['post'],sgep3['p>t']['post'], sgep4['p>t']['post'],sgep5['p>t']['post'],
                sgep6['p>t']['post'], sgep7['p>t']['post']]

        table.loc["Singlemom_pvalue"]=EP_SG_pv
        
        #EP_Singlemom StandaRD_EP Error
        EP_SG_se = [SG_RD_EP1.se['post'],SG_RD_EP2.se['post'], SG_RD_EP3.se['post'], SG_RD_EP4.se['post'], SG_RD_EP5.se['post'],
                 SG_RD_EP6.se['post'], SG_DID_EP7.se['post']]
        table.loc["Singlemom_SE"] = EP_SG_se

        
        #1-3 Employed under EP_NoPartner_NotMarried
        NPS_RD_EP1 = NPS_EP_1(et_ls)
        NPS_RD_EP2 = NPS_EP_2(et_ls)
        NPS_RD_EP3 = NPS_EP_3(et_ls)
        NPS_RD_EP4 = NPS_EP_4(et_ls)
        NPS_RD_EP5 = NPS_EP_5(et_ls)
        NPS_RD_EP6 = NPS_EP_6(et_ls) 
        NPS_DID_EP7 = NPS_EP_7(et_ls)
        
        EP_NPS = [NPS_RD_EP1.beta['post'], NPS_RD_EP2.beta['post'], NPS_RD_EP3.beta['post'], NPS_RD_EP4.beta['post'], NPS_RD_EP5.beta['post'],
                 NPS_RD_EP6.beta['post'], NPS_DID_EP7.beta['post']]
        table.loc["3.NoPartner_NotMarried"] = EP_NPS
        
        #EP_NoPartner_NotMarriedmom_pvalue 
        EP_NPS_pv = [npsep1['p>t']['post'], npsep2['p>t']['post'],npsep3['p>t']['post'], npsep4['p>t']['post'],npsep5['p>t']['post'],
                npsep6['p>t']['post'], npsep7['p>t']['post']]

        table.loc["NoPartner_NotMarried_pvalue"]= EP_NPS_pv
        
        #EP_NoPartner_NotMarriedmom Standard Error
        EP_NPS_se = [NPS_RD_EP1.se['post'],NPS_RD_EP2.se['post'], NPS_RD_EP3.se['post'], NPS_RD_EP4.se['post'], NPS_RD_EP5.se['post'],
                 NPS_RD_EP6.se['post'], NPS_DID_EP7.se['post']]
        table.loc["NoPartner_NotMarried_SE"] = EP_NPS_se

        
       #1-4 Employed Last Week under EP_NoDad_NotMarried
        NDS_RD_EP1 = NDS_EP_1(et_ls)
        NDS_RD_EP2 = NDS_EP_2(et_ls)
        NDS_RD_EP3 = NDS_EP_3(et_ls)
        NDS_RD_EP4 = NDS_EP_4(et_ls)
        NDS_RD_EP5 = NDS_EP_5(et_ls)
        NDS_RD_EP6 = NDS_EP_6(et_ls) 
        NDS_DID_EP7 = NDS_EP_7(et_ls)
        
        EP_NDS = [NDS_RD_EP1.beta['post'], NDS_RD_EP2.beta['post'], NDS_RD_EP3.beta['post'], NDS_RD_EP4.beta['post'], NDS_RD_EP5.beta['post'],
                 NDS_RD_EP6.beta['post'], NDS_DID_EP7.beta['post']]
        table.loc["4.NoDad_NotMarried"] = EP_NDS
        
        #EP_NoDad_NotMarried_pvalue 
        EP_NDS_pv = [ndsep1['p>t']['post'], ndsep2['p>t']['post'],ndsep3['p>t']['post'], ndsep4['p>t']['post'],ndsep5['p>t']['post'],
                ndsep6['p>t']['post'], ndsep7['p>t']['post']]

        table.loc["NoDad_NotMarried_pvalue"]=EP_NDS_pv
        
        #EP_NoDad_NotMarried Standard Error
        EP_NDS_se = [NDS_RD_EP1.se['post'],NDS_RD_EP2.se['post'], NDS_RD_EP3.se['post'], NDS_RD_EP4.se['post'], NDS_RD_EP5.se['post'],
                 NDS_RD_EP6.se['post'], NDS_DID_EP7.se['post']]
        table.loc["NoDad_NotMarried_SE"] = EP_NDS_se

        #1-5 Employed under EP_Divorced_NoPartner
        NPDIV_RD_EP1 = NPDIV_EP_1(et_ls)
        NPDIV_RD_EP2 = NPDIV_EP_2(et_ls)
        NPDIV_RD_EP3 = NPDIV_EP_3(et_ls)
        NPDIV_RD_EP4 = NPDIV_EP_4(et_ls)
        NPDIV_RD_EP5 = NPDIV_EP_5(et_ls)
        NPDIV_RD_EP6 = NPDIV_EP_6(et_ls) 
        NPDIV_DID_EP7 = NPDIV_EP_7(et_ls)
        
        EP_NPDIV = [NPDIV_RD_EP1.beta['post'], NPDIV_RD_EP2.beta['post'], NPDIV_RD_EP3.beta['post'], NPDIV_RD_EP4.beta['post'], NPDIV_RD_EP5.beta['post'],
                 NPDIV_RD_EP6.beta['post'], NPDIV_DID_EP7.beta['post']]
        table.loc["5.Divorced_NoPartner"] = EP_NPDIV
        
        #EP_Divorced_NoPartnermom_pvalue 
        EP_NPDIV_pv = [npdivep1['p>t']['post'], npdivep2['p>t']['post'],npdivep3['p>t']['post'], npdivep4['p>t']['post'],npdivep5['p>t']['post'],
                npdivep6['p>t']['post'], npdivep7['p>t']['post']]

        table.loc["Divorced_NoPartner_pvalue"]=EP_NPDIV_pv
        
        #EP_Divorced_NoPartnermom Standard Error
        EP_NPDIV_se = [NPDIV_RD_EP1.se['post'],NPDIV_RD_EP2.se['post'], NPDIV_RD_EP3.se['post'], NPDIV_RD_EP4.se['post'], NPDIV_RD_EP5.se['post'],
                 NPDIV_RD_EP6.se['post'], NPDIV_DID_EP7.se['post']]
        table.loc["Divorced_NoPartner_SE"] = EP_NPDIV_se





##=============================================================================    
        #Observations
        table=table.astype(float).round(4)
        obs =[NPND_RD_EP1.N, NPND_RD_EP2.N, NPND_RD_EP3.N, NPND_RD_EP4.N, NPND_RD_EP5.N, NPND_RD_EP6.N, NPND_DID_EP7.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48]
        table.loc["Number of months"] = months

        
        return(table)

