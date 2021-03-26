#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm

import unittest
import econtools
import econtools.metrics as mt


# In[2]:


# Load Household Expenditure

co_he=pd.read_stata("data/data_hbs_20110196.dta")


# In[3]:


# Age of mom and dad

co_he['agemom'].fillna(0, inplace=True)
co_he['agedad'].fillna(0, inplace=True)


# In[4]:


# Mom not present

del co_he['nomom']
co_he['nomom'] = 0
co_he.loc[co_he['agemom']==0, 'nomom'] = 1


# In[5]:


# Dad not present

del co_he['nodad']
co_he['nodad'] = 0
co_he.loc[co_he['agedad']==0, 'nodad'] = 1


# In[6]:


# Education of mom and dad

co_he['sec1mom']=0
co_he['sec1dad']=0
co_he['sec2mom']=0
co_he['sec2dad']=0
co_he['unimom']=0
co_he['unidad']=0

co_he.loc[co_he['educmom']==3, 'sec1mom'] = 1
co_he.loc[co_he['educdad']==3, 'sec1dad'] = 1

co_he.loc[(co_he['educmom']>3)&(co_he['educmom']<7), 'sec2mom'] = 1
co_he.loc[(co_he['educdad']>3)&(co_he['educdad']<7), 'sec2dad'] = 1

co_he.loc[(co_he['educmom']==7)|(co_he['educmom']==8), 'unimom'] = 1
co_he.loc[(co_he['educdad']==7)|(co_he['educdad']==8), 'unidad'] = 1


# In[7]:


# Immigrant

co_he['immig']=0
co_he.loc[(co_he['nacmom']==2) | (co_he['nacmom']==3), 'immig'] = 1


# In[8]:


# Mom not married

co_he['smom'] = 0
co_he.loc[co_he['ecivmom']!=2, 'smom'] = 1


# In[9]:


# Siblings and Daycare

co_he['sib']=0
co_he.loc[co_he['nmiem2']>1, 'sib'] = 1

co_he['age2']=co_he['agemom']*co_he['agemom']
co_he['age3']=co_he['agemom']*co_he['agemom']*co_he['agemom']

co_he['daycare_bin']=0
co_he.loc[(co_he['m_exp12312']>0) &(co_he['m_exp12312']!=np.nan), 'daycare_bin'] = 1


# In[10]:


# Create interaction dummies
co_he['ipost_1']=co_he['post']*co_he['month']
co_he['ipost_2'] =co_he['post']*co_he['month2']

# Create imes_enc dummies: imes_enc_1 for January, imes_enc_2 for February, and so on
for j in range(1,13):
    co_he['imes_enc_'+str(j)] = 0
    for i in range(len(co_he)):
        if co_he.loc[i,'mes_enc'] == j:
            co_he.loc[i, 'imes_enc_'+str(j)] = 1


# In[11]:


#======================= TABLE 3-A. Covariate Regression
#======================= 3A-1. [Age of Mother] Regression Discontinuity Design 
def HBS_Agemom1(co_he):
    X_co = ['post','month','month2','ipost_1','ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<10)], 'agemom', X_co, addcons=True)
    return(result)


# In[12]:


def HBS_Agemom2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']>-6) & (co_he['month']<5)], 'agemom', X_co, addcons=True)
    return(result)


# In[13]:


def HBS_Agemom3(co_he):    
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'agemom', X_co, addcons=True)
    return(result)


# In[14]:


def HBS_Agemom4(co_he):    
    result=mt.reg(co_he[(co_he['month']>-4) & (co_he['month']<3)], 'agemom', ['post'],addcons=True)
    return(result)


# In[15]:


def HBS_Agemom5(co_he):  
    result=mt.reg(co_he[(co_he['month']>-3) & (co_he['month']<2)], 'agemom', ['post'],addcons=True)
    return(result)


# In[16]:


#======================= 3A-2. [Age of Father] Regression Discontinuity Design 
def HBS_Agedad1(co_he):
    X_co =  ['post','month','month2','month3','ipost_1','ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'agedad',X_co, addcons=True)
    return(result)


# In[17]:


def HBS_Agedad2(co_he):
    X_co =  ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'agedad', X_co, addcons=True)
    return(result)


# In[18]:


def HBS_Agedad3(co_he):
    X_co =  ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'agedad', X_co, addcons=True)
    return(result)


# In[19]:


def HBS_Agedad4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'agedad', ['post'],addcons=True)
    return(result)


# In[20]:


def HBS_Agedad5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'agedad', ['post'],addcons=True)
    return(result)


# In[21]:


#======================= 3A-3. [Mother with Secondary Education] Regression Discontinuity Design 
def HBS_secmom1(co_he):
    X_co = ['post','month','month2','ipost_1','ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'sec1mom', X_co, addcons=True)
    return(result)


# In[22]:


def HBS_secmom2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'sec1mom', X_co,addcons=True)
    return(result)


# In[23]:


def HBS_secmom3(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'sec1mom', X_co, addcons=True)
    return(result)


# In[24]:


def HBS_secmom4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'sec1mom', ['post'],addcons=True)
    return(result)


# In[25]:


def HBS_secmom5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'sec1mom', ['post'],addcons=True)
    return(result)


# In[26]:


#======================= 3A-4. [Mother with Highschool Education] Regression Discontinuity Design 
def HBS_hsmom1(co_he):
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'sec2mom', X_co, addcons=True)
    return(result)


# In[27]:


def HBS_hsmom2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'sec2mom', X_co, addcons=True)
    return(result)


# In[28]:


def HBS_hsmom3(co_he):
    X_co =  ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'sec2mom', X_co, addcons=True)
    return(result)


# In[29]:


def HBS_hsmom4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'sec2mom', ['post'],addcons=True)
    return(result)


# In[30]:


def HBS_hsmom5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'sec2mom', ['post'],addcons=True)
    return(result)


# In[31]:


#======================= 3A-5. [Mother with College Education] Regression Discontinuity Design 
def HBS_cm1(co_he):
    X_co =  ['post','month','month2','ipost_1','ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'unimom', X_co, addcons=True)
    return(result)


# In[32]:


def HBS_cm2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'unimom', X_co, addcons=True)
    return(result)


# In[33]:


def HBS_cm3(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'unimom', X_co,addcons=True)
    return(result)


# In[34]:


def HBS_cm4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'unimom', ['post'],addcons=True)
    return(result)


# In[35]:


def HBS_cm5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'unimom', ['post'],addcons=True)
    return(result)


# In[36]:


#======================= 3A-6. [Father with Secondary Education] Regression Discontinuity Design 
def HBS_secdad1(co_he):
    X_co = ['post','month','month2','ipost_1','ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'sec1dad', X_co, addcons=True)
    return(result)


# In[37]:


def HBS_secdad2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'sec1dad', X_co, addcons=True)
    return(result)


# In[38]:


def HBS_secdad3(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'sec1dad', X_co, addcons=True)
    return(result)


# In[39]:


def HBS_secdad4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'sec1dad', ['post'],addcons=True)
    return(result)


# In[40]:


def HBS_secdad5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'sec1dad', ['post'],addcons=True)
    return(result)


# In[41]:


#======================= 3A-7. [Father with Highschool Education] Regression Discontinuity Design 
def HBS_hsdad1(co_he):
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'sec2dad', X_co, addcons=True)
    return(result)


# In[42]:


def HBS_hsdad2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'sec2dad', X_co, addcons=True)
    return(result)


# In[43]:


def HBS_hsdad3(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'sec2dad',  X_co, addcons=True)
    return(result)


# In[44]:


def HBS_hsdad4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'sec2dad', ['post'],addcons=True)
    return(result)


# In[45]:


def HBS_hsdad5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'sec2dad', ['post'],addcons=True)
    return(result)


# In[46]:


#======================= 3A-8. [Father with College Education] Regression Discontinuity Design 
def HBS_cd1(co_he):
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'unidad', X_co, addcons=True)
    return(result)


# In[47]:


def HBS_cd2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'unidad', X_co, addcons=True)
    return(result)


# In[48]:


def HBS_cd3(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'unidad', X_co, addcons=True)
    return(result)


# In[49]:


def HBS_cd4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'unidad', ['post'],addcons=True)
    return(result)


# In[50]:


def HBS_cd5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'unidad', ['post'],addcons=True)
    return(result)


# In[51]:


#======================= 3A-9. [MotherImmigrant] Regression Discontinuity Design 
def HBS_immig1(co_he):
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'immig', X_co ,addcons=True)
    return(result)


# In[52]:


def HBS_immig2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'immig', X_co, addcons=True)
    return(result)


# In[53]:


def HBS_immig3(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'immig', X_co, addcons=True)
    return(result)


# In[54]:


def HBS_immig4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'immig', ['post'],addcons=True)
    return(result)


# In[55]:


def HBS_immig5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'immig', ['post'],addcons=True)
    return(result)


# In[56]:


#======================= 3A-10. [Not first born (Sibling)] Regression Discontinuity Design 
def HBS_sib1(co_he):
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    result=mt.reg(co_he[(co_he['month']> -10) & (co_he['month']<9)], 'sib', X_co, addcons=True)
    return(result)


# In[57]:



#RDD9m(2)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_sib2(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -7) & (co_he['month']<6)], 'sib', X_co, addcons=True)
    return(result)


# In[58]:


def HBS_sib3(co_he):
    X_co = ['post','month','ipost_1']
    result=mt.reg(co_he[(co_he['month']> -5) & (co_he['month']<4)], 'sib', X_co, addcons=True)
    return(result)


# In[59]:


def HBS_sib4(co_he):
    result=mt.reg(co_he[(co_he['month']> -4) & (co_he['month']<3)], 'sib', ['post'],addcons=True)
    return(result)


# In[60]:


def HBS_sib5(co_he):
    result=mt.reg(co_he[(co_he['month']> -3) & (co_he['month']<2)], 'sib', ['post'],addcons=True)
    return(result)


# In[61]:


#========================= Female Labor Supply =========================#
co_ls=pd.read_stata("data/data_lfs_20110196.dta")


# In[62]:


# Control variables
co_ls['m2']=co_ls['m']*co_ls['m']

# No father present
co_ls['nodad']=0
co_ls.loc[co_ls['dadid']==0, 'nodad']=1

# Mother not married 
co_ls['smom']=0
co_ls.loc[co_ls['eciv']!=2, 'smom']=1

# Mother single
co_ls['single']=0
co_ls.loc[co_ls['eciv']==1, 'single']=1

# Mother separated or divorced
co_ls['sepdiv']=0
co_ls.loc[co_ls['eciv']==4, 'sepdiv']=1

# No partner in the household
co_ls['nopart']=0
co_ls.loc[co_ls['partner']==0, 'nopart']=1


# In[63]:


# Probability of the mother being in the maternity leave period at the time of the interview

co_ls['pleave']=0

co_ls.loc[(co_ls['q']==1) & (co_ls['m']==2)|(co_ls['q']==2) & (co_ls['m']==5)|(co_ls['q']==3) & (co_ls['m']==8)|(co_ls['q']==4) & (co_ls['m']==11) ,'pleave']=0.17
co_ls.loc[((co_ls['q']==1) & (co_ls['m']==3)) | ((co_ls['q']==2) & (co_ls['m']==6))  | ((co_ls['q']==3) & (co_ls['m']==9)) |((co_ls['q']==4) & (co_ls['m']==12)), 'pleave'] = 0.5
co_ls.loc[((co_ls['q']==1) & (co_ls['m']==4)) | ((co_ls['q']==2) & (co_ls['m']==7)) | ((co_ls['q']==3) & (co_ls['m']==10))  | ((co_ls['q']==4) & (co_ls['m']==13)), 'pleave'] = 0.83
co_ls.loc[((co_ls['q']==1) & (co_ls['m']>4) & (co_ls['m']<9)) | ((co_ls['q']==2) & (co_ls['m']>7) & (co_ls['m']<12)) | ((co_ls['q']==3) & (co_ls['m']>10) & (co_ls['m']<15))| ((co_ls['q']==4) & (co_ls['m']>13)), 'pleave'] = 1


# In[64]:


# Create interaction dummies
co_ls['ipost_1']=co_ls['post']*co_ls['m']
co_ls['ipost_2'] =co_ls['post']*co_ls['m2']

# Create iq dummies: iq_1 for quarter 1, iq_2 for quarter 2, and so on
for j in range(1,5):
    co_ls['iq_'+str(j)] = 0
    for i in range(len(co_ls)):
        if co_ls.loc[i,'q'] == j:
            co_ls.loc[i, 'iq_'+str(j)] = 1


# In[65]:


#======================= TABLE 3-B. Covariate Regression
#======================= 3B-1. [Age of Mother] Regression Discontinuity Design 

def LFS_Agemom1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'age', X_co_ls, addcons=True)
    return(result)


# In[66]:


def LFS_Agemom2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']<6)], 'age', X_co_ls,addcons=True)
    return(result)  


# In[67]:


def LFS_Agemom3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']<4)], 'age', X_co_ls, addcons=True)
    return(result)  


# In[68]:


def LFS_Agemom4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']<3)], 'age', ['post'],addcons=True)
    return(result)  


# In[69]:


def LFS_Agemom5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']<2)], 'age', ['post'],addcons=True)
    return(result)  


# In[70]:


#======================= 3B-2. [Age of Father] Regression Discontinuity Design 
def LFS_Agedad1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'agedad', X_co_ls, addcons=True)
    return(result)


# In[71]:


def LFS_Agedad2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']< 6)], 'agedad', X_co_ls, addcons=True)
    return(result)


# In[72]:


def LFS_Agedad3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']< 4)], 'agedad', X_co_ls, addcons=True)
    return(result)


# In[73]:


def LFS_Agedad4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']< 3)], 'agedad', ['post'], addcons=True)
    return(result)


# In[74]:


def LFS_Agedad5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']< 2)], 'agedad', ['post'], addcons=True)
    return(result)


# In[75]:


#======================= 3B-3. [Mother with Secondary Education] Regression Discontinuity Design 
def LFS_secmom1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']<9)], 'primary', X_co_ls, addcons=True)
    return(result)


# In[76]:


def LFS_secmom2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']<6)], 'primary', X_co_ls, addcons=True)
    return(result)


# In[77]:


def LFS_secmom3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']<4)], 'primary', X_co_ls, addcons=True)
    return(result)


# In[78]:


def LFS_secmom4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']<3)], 'primary', ['post'], addcons=True)
    return(result)


# In[79]:


def LFS_secmom5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']<2)], 'primary', ['post'], addcons=True)
    return(result)


# In[80]:


#======================= 3B-4. [Mother with Highschool Education] Regression Discontinuity Design 
def LFS_hsmom1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'hsgrad', X_co_ls, addcons=True)
    return(result)


# In[81]:


def LFS_hsmom2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']< 6)], 'hsgrad', X_co_ls, addcons=True)
    return(result)


# In[82]:


def LFS_hsmom3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']< 4)], 'hsgrad',  X_co_ls, addcons=True)
    return(result)


# In[83]:


def LFS_hsmom4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']<3)], 'hsgrad', ['post'], addcons=True)
    return(result)


# In[84]:


def LFS_hsmom5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']<2)], 'hsgrad', ['post'], addcons=True)
    return(result)


# In[85]:


#======================= 3B-5. [Mother with College Education] Regression Discontinuity Design 
def LFS_cm1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'univ', X_co_ls, addcons=True)
    return(result)


# In[86]:


def LFS_cm2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']< 6)], 'univ', X_co_ls, addcons=True)
    return(result)


# In[87]:


def LFS_cm3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']<4)], 'univ', X_co_ls, addcons=True)
    return(result)


# In[88]:


def LFS_cm4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']< 3)], 'univ', ['post'], addcons=True)
    return(result)


# In[89]:


def LFS_cm5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']< 2)], 'univ', ['post'], addcons=True)
    return(result)


# In[90]:


#======================= 3B-6. [Father with Secondary Education] Regression Discontinuity Design 
def LFS_secdad1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'primary_dad', X_co_ls, addcons=True)
    return(result)


# In[91]:


def LFS_secdad2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']< 6)], 'primary_dad', X_co_ls, addcons=True)
    return(result)


# In[92]:


def LFS_secdad3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -6) & (co_ls['m']< 4)], 'primary_dad', X_co_ls, addcons=True)
    return(result)


# In[93]:


def LFS_secdad4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']<3)], 'primary_dad', ['post'], addcons=True)
    return(result)


# In[94]:


def LFS_secdad5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']< 2)], 'primary_dad', ['post'], addcons=True)
    return(result)


# In[95]:


#======================= 3B-7. [Father with Highschool Education] Regression Discontinuity Design 
def LFS_hsdad1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'hsgrad_dad',X_co_ls, addcons=True)
    return(result)


# In[96]:


def LFS_hsdad2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']<6)], 'hsgrad_dad', X_co_ls, addcons=True)
    return(result)


# In[97]:


def LFS_hsdad3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']<4)], 'hsgrad_dad', X_co_ls, addcons=True)
    return(result)


# In[98]:


def LFS_hsdad4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']< 3)], 'hsgrad_dad', ['post'], addcons=True)
    return(result)


# In[99]:


def LFS_hsdad5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']< 2)], 'hsgrad_dad', ['post'], addcons=True)
    return(result)


# In[100]:


#======================= 3B-8. [Father with College Education] Regression Discontinuity Design 
def LFS_cd1(co_ls):
    X_co_ls =  ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'univ_dad', X_co_ls, addcons=True)
    return(result)


# In[101]:


def LFS_cd2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']< 6)], 'univ_dad', X_co_ls, addcons=True)
    return(result)


# In[102]:


def LFS_cd3(co_ls):
    X_co_ls = ['post','m','ipost_1']    
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']<4)], 'univ_dad', X_co_ls, addcons=True)
    return(result)


# In[103]:


def LFS_cd4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']< 3)], 'univ_dad', ['post'], addcons=True)
    return(result)


# In[104]:


def LFS_cd5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']<2)], 'univ_dad', ['post'], addcons=True)
    return(result)


# In[105]:


#======================= 3B-9. [MotherImmigrant] Regression Discontinuity Design 
def LFS_immig1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'immig', X_co_ls, addcons=True)
    return(result)


# In[106]:


def LFS_immig2(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']<6)], 'immig', X_co_ls, addcons=True)
    return(result)


# In[107]:


def LFS_immig3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']< 4)], 'immig', X_co_ls, addcons=True)
    return(result)


# In[108]:


def LFS_immig4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']< 3)], 'immig', ['post'], addcons=True)
    return(result)


# In[109]:


def LFS_immig5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']< 2)], 'immig', ['post'], addcons=True)
    return(result)


# In[110]:


#======================= 3B-10. [Not first born (Sibling)] Regression Discontinuity Design 
def LFS_sib1(co_ls):
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2']
    result=mt.reg(co_ls[(co_ls['m']> -10) & (co_ls['m']< 9)], 'sib', X_co_ls, addcons=True)
    return(result)


# In[111]:


def LFS_sib2(co_ls):
    X_co_ls =  ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -7) & (co_ls['m']<6)], 'sib', X_co_ls, addcons=True)
    return(result)


# In[112]:


def LFS_sib3(co_ls):
    X_co_ls = ['post','m','ipost_1']
    result=mt.reg(co_ls[(co_ls['m']> -5) & (co_ls['m']< 4)], 'sib', X_co_ls, addcons=True)
    return(result)


# In[113]:


def LFS_sib4(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -4) & (co_ls['m']< 3)], 'sib', ['post'], addcons=True)
    return(result)


# In[114]:


def LFS_sib5(co_ls):
    result=mt.reg(co_ls[(co_ls['m']> -3) & (co_ls['m']< 2)], 'sib', ['post'], addcons=True)
    return(result)


# In[115]:


#================================= Create Table 3
def table3_HBS(co_he):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': []})
        result = ('Age of mother', 'AoM_SE', 'Age of father', 'AoF_SE',
                  'Mother secondary','MS_SE','Mother highschool graduate','MHG_SE',
                  'Mother college graduate','MCG_SE','Father seconary','FS_SE',
                  'Father highschool graduate', 'FHG_SE',
                  'Father college graduate','FCG_SE','Mother immigrant','MI_SE',
                  'Not first born','NFB_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Number of months')
        
        table['HBS: Balance in Covariate'] = result
        table = table.set_index('HBS: Balance in Covariate')

#============================== Panel A. HBS
        
        #Age of mother 
        HBS_AOM1 = HBS_Agemom1(co_he)
        HBS_AOM2 = HBS_Agemom2(co_he)
        HBS_AOM3 = HBS_Agemom3(co_he)
        HBS_AOM4 = HBS_Agemom4(co_he)
        HBS_AOM5= HBS_Agemom5(co_he)
        
        hbs_aom = [HBS_AOM1.beta['post'],HBS_AOM2.beta['post'],HBS_AOM3.beta['post'], HBS_AOM4.beta['post'],HBS_AOM5.beta['post']]
        table.loc["Age of mother"] = hbs_aom
        
        #AoM Standard Error
        hbs_aom_se = [HBS_AOM1.se['post'],HBS_AOM2.se['post'],HBS_AOM3.se['post'], HBS_AOM4.se['post'], HBS_AOM5.se['post']]
        table.loc["AoM_SE"]=hbs_aom_se
     
        #Age of father 
        HBS_AOF1 = HBS_Agedad1(co_he)
        HBS_AOF2 = HBS_Agedad2(co_he)
        HBS_AOF3 = HBS_Agedad3(co_he)
        HBS_AOF4 = HBS_Agedad4(co_he)
        HBS_AOF5= HBS_Agedad5(co_he)
        
        hbs_aod = [HBS_AOF1.beta['post'],HBS_AOF2.beta['post'],HBS_AOF3.beta['post'], HBS_AOF4.beta['post'],HBS_AOF5.beta['post']]
        table.loc["Age of father"] = hbs_aod
        
        #AoF Standard Error
        hbs_aod_se = [HBS_AOF1.se['post'],HBS_AOF2.se['post'],HBS_AOF3.se['post'], HBS_AOF4.se['post'], HBS_AOF5.se['post']]
        table.loc["AoF_SE"]=hbs_aod_se
            
        #Mother Secondary 
        HBS_MS1 = HBS_secmom1(co_he)
        HBS_MS2 = HBS_secmom2(co_he)
        HBS_MS3 = HBS_secmom3(co_he)
        HBS_MS4 = HBS_secmom4(co_he)
        HBS_MS5 = HBS_secmom5(co_he)
        hbs_ms = [HBS_MS1.beta['post'],HBS_MS2.beta['post'],HBS_MS3.beta['post'], HBS_MS4.beta['post'], HBS_MS5.beta['post']]
        table.loc["Mother secondary"] = hbs_ms
        
        #MS Standard Error
        hbs_ms_se = [HBS_MS1.se['post'],HBS_MS2.se['post'],HBS_MS3.se['post'], HBS_MS4.se['post'], HBS_MS5.se['post']]
        table.loc["MS_SE"]=hbs_ms_se
 
        #Mother highschool graduate
        HBS_MHG1 = HBS_hsmom1(co_he)
        HBS_MHG2 = HBS_hsmom2(co_he)
        HBS_MHG3 = HBS_hsmom3(co_he)
        HBS_MHG4 = HBS_hsmom4(co_he)
        HBS_MHG5 = HBS_hsmom5(co_he)
        hbs_mhg = [ HBS_MHG1.beta['post'], HBS_MHG2.beta['post'], HBS_MHG3.beta['post'],  HBS_MHG4.beta['post'],  HBS_MHG5.beta['post']]
        table.loc["Mother highschool graduate"] = hbs_mhg
        
        #MHG Standard Error
        hbs_mhg_se = [ HBS_MHG1.se['post'],HBS_MHG2.se['post'],HBS_MHG3.se['post'], HBS_MHG4.se['post'], HBS_MHG5.se['post']]
        table.loc["MHG_SE"]= hbs_mhg_se
        
        #Mother college graduate
        HBS_MCG1 = HBS_cm1(co_he)
        HBS_MCG2 = HBS_cm2(co_he)
        HBS_MCG3 = HBS_cm3(co_he)
        HBS_MCG4 = HBS_cm4(co_he)
        HBS_MCG5 = HBS_cm5(co_he)
        hbs_mcg = [ HBS_MCG1.beta['post'],HBS_MCG2.beta['post'],HBS_MCG3.beta['post'], HBS_MCG4.beta['post'], HBS_MCG5.beta['post']]
        table.loc["Mother college graduate"] = hbs_mcg
        
        #MCG_SE
        hbs_mcg_se = [HBS_MCG1.se['post'],HBS_MCG2.se['post'],HBS_MCG3.se['post'], HBS_MCG4.se['post'], HBS_MCG5.se['post']]
        table.loc["MCG_SE"]=hbs_mcg_se
 
        #Father seconary
        HBS_FS1 = HBS_secdad1(co_he)
        HBS_FS2 = HBS_secdad2(co_he)
        HBS_FS3 = HBS_secdad3(co_he)
        HBS_FS4 = HBS_secdad4(co_he)
        HBS_FS5 = HBS_secdad5(co_he)
        hbs_fs = [HBS_FS1.beta['post'],HBS_FS2.beta['post'],HBS_FS3.beta['post'], HBS_FS4.beta['post'], HBS_FS5.beta['post']]
        table.loc["Father seconary"] = hbs_fs
         
        #FS_SE
        hbs_fs_se =[HBS_FS1.se['post'],HBS_FS2.se['post'],HBS_FS3.se['post'], HBS_FS4.se['post'], HBS_FS5.se['post']]
        table.loc["FS_SE"]= hbs_fs_se
        
        #'Father highschool graduate'
        HBS_FHG1 = HBS_hsdad1(co_he)
        HBS_FHG2 = HBS_hsdad2(co_he)
        HBS_FHG3 = HBS_hsdad3(co_he)
        HBS_FHG4 = HBS_hsdad4(co_he)
        HBS_FHG5 = HBS_hsdad5(co_he)
        hbs_fhg = [HBS_FHG1.beta['post'], HBS_FHG2.beta['post'], HBS_FHG3.beta['post'],  HBS_FHG4.beta['post'],  HBS_FHG5.beta['post']]
        table.loc["Father highschool graduate"] = hbs_fhg
        
        #FHG Standard Error
        hbs_fhg_se = [ HBS_FHG1.se['post'],HBS_FHG2.se['post'],HBS_FHG3.se['post'], HBS_FHG4.se['post'], HBS_FHG5.se['post']]
        table.loc["FHG_SE"]= hbs_fhg_se
        
        #Father college graduate
        HBS_FCG1 = HBS_cd1(co_he)
        HBS_FCG2 = HBS_cd2(co_he)
        HBS_FCG3 = HBS_cd3(co_he)
        HBS_FCG4 = HBS_cd4(co_he)
        HBS_FCG5 = HBS_cd5(co_he)
        hbs_FCG = [ HBS_FCG1.beta['post'],HBS_FCG2.beta['post'],HBS_FCG3.beta['post'], HBS_FCG4.beta['post'], HBS_FCG5.beta['post']]
        table.loc["Father college graduate"] = hbs_FCG
        
        #FCG_SE
        hbs_fcg_se = [HBS_FCG1.se['post'],HBS_FCG2.se['post'],HBS_FCG3.se['post'], HBS_FCG4.se['post'], HBS_FCG5.se['post']]
        table.loc["FCG_SE"]=hbs_fcg_se

        #Mother immigrant
        HBS_MI1 = HBS_immig1(co_he) 
        HBS_MI2 = HBS_immig2(co_he) 
        HBS_MI3 = HBS_immig3(co_he) 
        HBS_MI4 = HBS_immig4(co_he) 
        HBS_MI5 =HBS_immig5(co_he)

        hbs_MI = [ HBS_MI1.beta['post'],HBS_MI2.beta['post'],HBS_MI3.beta['post'], HBS_MI4.beta['post'], HBS_MI5.beta['post']]
        table.loc["Mother immigrant"] = hbs_MI
        
        #MI_SE
        hbs_MI_se = [HBS_MI1.se['post'],HBS_MI2.se['post'],HBS_MI3.se['post'], HBS_MI4.se['post'], HBS_MI5.se['post']]
        table.loc['MI_SE']=hbs_MI_se
        
        #Not first born
        HBS_NFB1 = HBS_sib1(co_he) 
        HBS_NFB2 = HBS_sib2(co_he) 
        HBS_NFB3 = HBS_sib3(co_he) 
        HBS_NFB4 = HBS_sib4(co_he) 
        HBS_NFB5 = HBS_sib5(co_he)

        hbs_NFB = [HBS_NFB1.beta['post'],HBS_NFB2.beta['post'],HBS_NFB3.beta['post'], HBS_NFB4.beta['post'], HBS_NFB5.beta['post']]
        table.loc["Not first born"] = hbs_NFB

        #'NFB_SE'
        hbs_NFB_se = [HBS_NFB1.se['post'],HBS_NFB2.se['post'],HBS_NFB3.se['post'], HBS_NFB4.se['post'], HBS_NFB5.se['post']]
        table.loc['NFB_SE']=hbs_NFB_se

        #Observations
        table=table.astype(float).round(3)
        obs =[HBS_NFB1.N, HBS_NFB2.N, HBS_NFB3.N, HBS_NFB4.N, HBS_NFB5.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N"]
        table.loc["Quadric trend in m"] = quadric
        
        #Number of months
        months = [18,12,8,6,4]
        table.loc["Number of months"] = months
        return(table)


# In[116]:


def table3_LFS(co_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': []})
        result = ('Age of mother', 'AoM_SE', 'Age of father', 'AoF_SE',
                  'Mother secondary','MS_SE','Mother highschool graduate','MHG_SE',
                  'Mother college graduate','MCG_SE','Father seconary','FS_SE',
                  'Father highschool graduate','FHG_SE',
                  'Father college graduate','FCG_SE','Mother immigrant','MI_SE',
                  'Not first born','NFB_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Number of months')

        
        table['LFS: Balance in Covariate'] = result
        table = table.set_index('LFS: Balance in Covariate')

#===================================Panel B.LFS
        #Age of mother 
        LFS_AOM1 = LFS_Agemom1(co_ls)
        LFS_AOM2 = LFS_Agemom2(co_ls)
        LFS_AOM3 = LFS_Agemom3(co_ls)
        LFS_AOM4 = LFS_Agemom4(co_ls)
        LFS_AOM5= LFS_Agemom5(co_ls)
        
        LFS_aom = [LFS_AOM1.beta['post'],LFS_AOM2.beta['post'],LFS_AOM3.beta['post'], LFS_AOM4.beta['post'],LFS_AOM5.beta['post']]
        table.loc["Age of mother"] = LFS_aom
        
        #AoM Standard Error
        LFS_aom_se = [LFS_AOM1.se['post'],LFS_AOM2.se['post'],LFS_AOM3.se['post'], LFS_AOM4.se['post'], LFS_AOM5.se['post']]
        table.loc["AoM_SE"]=LFS_aom_se
     
        #Age of father 
        LFS_AOF1 = LFS_Agedad1(co_ls)
        LFS_AOF2 = LFS_Agedad2(co_ls)
        LFS_AOF3 = LFS_Agedad3(co_ls)
        LFS_AOF4 = LFS_Agedad4(co_ls)
        LFS_AOF5= LFS_Agedad5(co_ls)
        
        LFS_aod = [LFS_AOF1.beta['post'],LFS_AOF2.beta['post'],LFS_AOF3.beta['post'], LFS_AOF4.beta['post'],LFS_AOF5.beta['post']]
        table.loc["Age of father"] = LFS_aod
        
        #AoF Standard Error
        LFS_aod_se = [LFS_AOF1.se['post'],LFS_AOF2.se['post'],LFS_AOF3.se['post'], LFS_AOF4.se['post'], LFS_AOF5.se['post']]
        table.loc["AoF_SE"]=LFS_aod_se
            
        #Mother Secondary 
        LFS_MS1 = LFS_secmom1(co_ls)
        LFS_MS2 = LFS_secmom2(co_ls)
        LFS_MS3 = LFS_secmom3(co_ls)
        LFS_MS4 = LFS_secmom4(co_ls)
        LFS_MS5 = LFS_secmom5(co_ls)
        LFS_ms = [LFS_MS1.beta['post'],LFS_MS2.beta['post'],LFS_MS3.beta['post'], LFS_MS4.beta['post'], LFS_MS5.beta['post']]
        table.loc["Mother secondary"] = LFS_ms
        
        #MS Standard Error
        LFS_ms_se = [LFS_MS1.se['post'],LFS_MS2.se['post'],LFS_MS3.se['post'], LFS_MS4.se['post'], LFS_MS5.se['post']]
        table.loc["MS_SE"]=LFS_ms_se
 
        #Mother highschool graduate
        LFS_MHG1 = LFS_hsmom1(co_ls)
        LFS_MHG2 = LFS_hsmom2(co_ls)
        LFS_MHG3 = LFS_hsmom3(co_ls)
        LFS_MHG4 = LFS_hsmom4(co_ls)
        LFS_MHG5 = LFS_hsmom5(co_ls)
        LFS_mhg = [ LFS_MHG1.beta['post'], LFS_MHG2.beta['post'], LFS_MHG3.beta['post'],  LFS_MHG4.beta['post'],  LFS_MHG5.beta['post']]
        table.loc["Mother highschool graduate"] = LFS_mhg
        
        #MHG Standard Error
        LFS_mhg_se = [ LFS_MHG1.se['post'],LFS_MHG2.se['post'],LFS_MHG3.se['post'], LFS_MHG4.se['post'], LFS_MHG5.se['post']]
        table.loc["MHG_SE"]= LFS_mhg_se
        
        #Mother college graduate
        LFS_MCG1 = LFS_cm1(co_ls)
        LFS_MCG2 = LFS_cm2(co_ls)
        LFS_MCG3 = LFS_cm3(co_ls)
        LFS_MCG4 = LFS_cm4(co_ls)
        LFS_MCG5 = LFS_cm5(co_ls)
        LFS_mcg = [ LFS_MCG1.beta['post'],LFS_MCG2.beta['post'],LFS_MCG3.beta['post'], LFS_MCG4.beta['post'], LFS_MCG5.beta['post']]
        table.loc["Mother college graduate"] = LFS_mcg
        
        #MCG_SE
        LFS_mcg_se = [LFS_MCG1.se['post'],LFS_MCG2.se['post'],LFS_MCG3.se['post'], LFS_MCG4.se['post'], LFS_MCG5.se['post']]
        table.loc["MCG_SE"]=LFS_mcg_se
 
        #Father seconary
        LFS_FS1 = LFS_secdad1(co_ls)
        LFS_FS2 = LFS_secdad2(co_ls)
        LFS_FS3 = LFS_secdad3(co_ls)
        LFS_FS4 = LFS_secdad4(co_ls)
        LFS_FS5 = LFS_secdad5(co_ls)
        LFS_fs = [LFS_FS1.beta['post'],LFS_FS2.beta['post'],LFS_FS3.beta['post'], LFS_FS4.beta['post'], LFS_FS5.beta['post']]
        table.loc["Father seconary"] = LFS_fs
         
        #FS_SE
        LFS_fs_se =[LFS_FS1.se['post'],LFS_FS2.se['post'],LFS_FS3.se['post'], LFS_FS4.se['post'], LFS_FS5.se['post']]
        table.loc["FS_SE"]= LFS_fs_se
        
        #'Father highschool graduate'
        LFS_FHG1 = LFS_hsdad1(co_ls)
        LFS_FHG2 = LFS_hsdad2(co_ls)
        LFS_FHG3 = LFS_hsdad3(co_ls)
        LFS_FHG4 = LFS_hsdad4(co_ls)
        LFS_FHG5 = LFS_hsdad5(co_ls)
        LFS_fhg = [LFS_FHG1.beta['post'], LFS_FHG2.beta['post'], LFS_FHG3.beta['post'],  LFS_FHG4.beta['post'],  LFS_FHG5.beta['post']]
        table.loc["Father highschool graduate"] = LFS_fhg
        
        #FHG Standard Error
        LFS_fhg_se = [ LFS_FHG1.se['post'],LFS_FHG2.se['post'],LFS_FHG3.se['post'], LFS_FHG4.se['post'], LFS_FHG5.se['post']]
        table.loc["FHG_SE"]= LFS_fhg_se
        
        #Father college graduate
        LFS_FCG1 = LFS_cd1(co_ls)
        LFS_FCG2 = LFS_cd2(co_ls)
        LFS_FCG3 = LFS_cd3(co_ls)
        LFS_FCG4 = LFS_cd4(co_ls)
        LFS_FCG5 = LFS_cd5(co_ls)
        LFS_FCG = [ LFS_FCG1.beta['post'],LFS_FCG2.beta['post'],LFS_FCG3.beta['post'], LFS_FCG4.beta['post'], LFS_FCG5.beta['post']]
        table.loc["Father college graduate"] = LFS_FCG
        
        #FCG_SE
        LFS_fcg_se = [LFS_FCG1.se['post'],LFS_FCG2.se['post'],LFS_FCG3.se['post'], LFS_FCG4.se['post'], LFS_FCG5.se['post']]
        table.loc["FCG_SE"]=LFS_fcg_se

        #Mother immigrant
        LFS_MI1 = LFS_immig1(co_ls) 
        LFS_MI2 = LFS_immig2(co_ls) 
        LFS_MI3 = LFS_immig3(co_ls) 
        LFS_MI4 = LFS_immig4(co_ls) 
        LFS_MI5 =LFS_immig5(co_ls)

        LFS_MI = [ LFS_MI1.beta['post'],LFS_MI2.beta['post'],LFS_MI3.beta['post'], LFS_MI4.beta['post'], LFS_MI5.beta['post']]
        table.loc["Mother immigrant"] = LFS_MI
        
        #MI_SE
        LFS_MI_se = [LFS_MI1.se['post'],LFS_MI2.se['post'],LFS_MI3.se['post'], LFS_MI4.se['post'], LFS_MI5.se['post']]
        table.loc['MI_SE']=LFS_MI_se
        
        #Not first born
        LFS_NFB1 = LFS_sib1(co_ls) 
        LFS_NFB2 = LFS_sib2(co_ls) 
        LFS_NFB3 = LFS_sib3(co_ls) 
        LFS_NFB4 = LFS_sib4(co_ls) 
        LFS_NFB5 = LFS_sib5(co_ls)

        LFS_NFB = [LFS_NFB1.beta['post'],LFS_NFB2.beta['post'],LFS_NFB3.beta['post'], LFS_NFB4.beta['post'], LFS_NFB5.beta['post']]
        table.loc["Not first born"] = LFS_NFB

        #'NFB_SE'
        LFS_NFB_se = [LFS_NFB1.se['post'],LFS_NFB2.se['post'],LFS_NFB3.se['post'], LFS_NFB4.se['post'], LFS_NFB5.se['post']]
        table.loc['NFB_SE']=LFS_NFB_se

        #Observations
        table=table.astype(float).round(3)
        LFS_obs = [LFS_NFB1.N, LFS_NFB2.N, LFS_NFB3.N, LFS_NFB4.N, LFS_NFB5.N]
        table.loc["Observations"] = LFS_obs        
               
        #Linar trend in m
        LFS_linear = ["Y","Y","Y","N","N"]
        table.loc["Linear trend in m"] = LFS_linear
        
        #Quadric trend in m
        LFS_quadric = ["Y","N","N","N","N"]
        table.loc["Quadric trend in m"] = LFS_quadric
        
        #Number of months
        LFS_months = [18,12,8,6,4]
        table.loc["Number of months"] = LFS_months
        
        return(table)


# In[128]:


# Check the accuracy of the above replicated results via unittest

# Table 3.PanelA.House Budget Survey (HBS): (1) Mother highschool graduate
class test_table_3HBSa(unittest.TestCase):
    def setUp(self):
        HBS_MHG1 = HBS_hsmom1(co_he)
        HBS_MHG2 = HBS_hsmom2(co_he)
        HBS_MHG3 = HBS_hsmom3(co_he)
        HBS_MHG4 = HBS_hsmom4(co_he)
        HBS_MHG5 = HBS_hsmom5(co_he)
        hbs_mhg = [ HBS_MHG1.beta['post'], HBS_MHG2.beta['post'], HBS_MHG3.beta['post'],  HBS_MHG4.beta['post'],  HBS_MHG5.beta['post']]
        self.result = [round(num, 3) for num in hbs_mhg]
        self.expected = [-0.003, -0.055, -0.043, 0.01, 0.001]

    def test_count_eq_table_3HBSa(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3HBSa(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[129]:


# Table 3.PanelA.HBS: (2) Father highschool graduate
class test_table_3HBSb(unittest.TestCase):
    def setUp(self):
        HBS_FHG1 = HBS_hsdad1(co_he)
        HBS_FHG2 = HBS_hsdad2(co_he)
        HBS_FHG3 = HBS_hsdad3(co_he)
        HBS_FHG4 = HBS_hsdad4(co_he)
        HBS_FHG5 = HBS_hsdad5(co_he)
        hbs_fhg = [HBS_FHG1.beta['post'], HBS_FHG2.beta['post'], HBS_FHG3.beta['post'],  HBS_FHG4.beta['post'],  HBS_FHG5.beta['post']]
        self.result = [round(num, 3) for num in hbs_fhg]
        self.expected = [-0.026, -0.032, -0.005, -0.007, 0.005]

    def test_count_eq_table_3HBSb(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3HBSb(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[130]:


# Table 3.PanelA.HBS: (3) Mother immigrant
class test_table_3c(unittest.TestCase):
    def setUp(self):
        HBS_MI1 = HBS_immig1(co_he) 
        HBS_MI2 = HBS_immig2(co_he) 
        HBS_MI3 = HBS_immig3(co_he) 
        HBS_MI4 = HBS_immig4(co_he) 
        HBS_MI5 =HBS_immig5(co_he)
        hbs_MI = [ HBS_MI1.beta['post'],HBS_MI2.beta['post'],HBS_MI3.beta['post'], HBS_MI4.beta['post'], HBS_MI5.beta['post']]       
        self.result = [round(num, 3) for num in hbs_MI]
        self.expected = [-0.038, -0.016, 0.013, 0.011, 0.058]

    def test_count_eq_table_3HBSc(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3HBSc(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[131]:


# Table 3.PanelA.HBS: (4) Not first born
class test_table_3d(unittest.TestCase):
    def setUp(self):
        HBS_NFB1 = HBS_sib1(co_he) 
        HBS_NFB2 = HBS_sib2(co_he) 
        HBS_NFB3 = HBS_sib3(co_he) 
        HBS_NFB4 = HBS_sib4(co_he) 
        HBS_NFB5 = HBS_sib5(co_he)
        hbs_NFB = [HBS_NFB1.beta['post'],HBS_NFB2.beta['post'],HBS_NFB3.beta['post'], HBS_NFB4.beta['post'], HBS_NFB5.beta['post']]
        self.result = [round(num, 3) for num in hbs_NFB]
        self.expected = [-0.017, 0.003, -0.036, -0.017, -0.045]
        
    def test_count_eq_table_3HBSd(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3HBSd(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[132]:


# Table 3.PanelB.Labor Force Survey (LFS): (1) Age of mother
class test_table_3LFSa(unittest.TestCase):
    def setUp(self):
        LFS_AOM1 = LFS_Agemom1(co_ls)
        LFS_AOM2 = LFS_Agemom2(co_ls)
        LFS_AOM3 = LFS_Agemom3(co_ls)
        LFS_AOM4 = LFS_Agemom4(co_ls)
        LFS_AOM5= LFS_Agemom5(co_ls)   
        LFS_aom = [LFS_AOM1.beta['post'],LFS_AOM2.beta['post'],LFS_AOM3.beta['post'], LFS_AOM4.beta['post'],LFS_AOM5.beta['post']]
        self.result = [round(num, 3) for num in LFS_aom]
        self.expected = [-0.307,-0.283,-0.292,-0.692,-0.164]
        
    def test_count_eq_table_3LFSa(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3LFSa(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[133]:


# Table 3.PanelB.LFS: (2) Age of father
class test_table_3LFSb(unittest.TestCase):
    def setUp(self):
        LFS_AOF1 = LFS_Agedad1(co_ls)
        LFS_AOF2 = LFS_Agedad2(co_ls)
        LFS_AOF3 = LFS_Agedad3(co_ls)
        LFS_AOF4 = LFS_Agedad4(co_ls)
        LFS_AOF5= LFS_Agedad5(co_ls)     
        LFS_aod = [LFS_AOF1.beta['post'],LFS_AOF2.beta['post'],LFS_AOF3.beta['post'], LFS_AOF4.beta['post'],LFS_AOF5.beta['post']]
        self.result = [round(num, 3) for num in LFS_aod]
        self.expected = [-0.43,-0.405,-0.728,-0.912,-0.605]
        
    def test_count_eq_table_3LFSb(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3LFSb(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[134]:


# Table 3.PanelB.LFS: (3) Mother secondary education
class test_table_3LFSc(unittest.TestCase):
    def setUp(self):
        LFS_MS1 = LFS_secmom1(co_ls)
        LFS_MS2 = LFS_secmom2(co_ls)
        LFS_MS3 = LFS_secmom3(co_ls)
        LFS_MS4 = LFS_secmom4(co_ls)
        LFS_MS5 = LFS_secmom5(co_ls)
        LFS_ms = [LFS_MS1.beta['post'],LFS_MS2.beta['post'],LFS_MS3.beta['post'], LFS_MS4.beta['post'], LFS_MS5.beta['post']]
        self.result = [round(num, 3) for num in LFS_ms]
        self.expected = [0.009, 0.013, 0.05, 0.024, 0.04]
        
    def test_count_eq_table_3LFSc(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3LFSc(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[135]:


# Table 3.PanelB.LFS: (4) Father seconary education
class test_table_3LFSd(unittest.TestCase):
    def setUp(self):
        #Father seconary
        LFS_FS1 = LFS_secdad1(co_ls)
        LFS_FS2 = LFS_secdad2(co_ls)
        LFS_FS3 = LFS_secdad3(co_ls)
        LFS_FS4 = LFS_secdad4(co_ls)
        LFS_FS5 = LFS_secdad5(co_ls)
        LFS_fs = [LFS_FS1.beta['post'],LFS_FS2.beta['post'],LFS_FS3.beta['post'], LFS_FS4.beta['post'], LFS_FS5.beta['post']]
        self.result = [round(num, 3) for num in LFS_fs]
        self.expected = [-0.015,-0.023,-0.008,-0.001,-0.012]
        
    def test_count_eq_table_3LFSd(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3LFSd(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[136]:


# Table 3.PanelB.LFS: (5) Mother immigrant
class test_table_3LFSe(unittest.TestCase):
    def setUp(self):
        #Father seconary
        LFS_MI1 = LFS_immig1(co_ls) 
        LFS_MI2 = LFS_immig2(co_ls) 
        LFS_MI3 = LFS_immig3(co_ls) 
        LFS_MI4 = LFS_immig4(co_ls) 
        LFS_MI5 =LFS_immig5(co_ls)
        LFS_MI = [ LFS_MI1.beta['post'],LFS_MI2.beta['post'],LFS_MI3.beta['post'], LFS_MI4.beta['post'], LFS_MI5.beta['post']]
        self.result = [round(num, 3) for num in LFS_MI]
        self.expected = [0.022,0.009,0.015,0.009,0.008]
        
    def test_count_eq_table_3LFSe(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3LFSe(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[137]:


# Table 3.PanelB.LFS: (6) Not first born
class test_table_3LFSf(unittest.TestCase):
    def setUp(self):
        LFS_NFB1 = LFS_sib1(co_ls) 
        LFS_NFB2 = LFS_sib2(co_ls) 
        LFS_NFB3 = LFS_sib3(co_ls) 
        LFS_NFB4 = LFS_sib4(co_ls) 
        LFS_NFB5 = LFS_sib5(co_ls)
        LFS_NFB = [LFS_NFB1.beta['post'],LFS_NFB2.beta['post'],LFS_NFB3.beta['post'], LFS_NFB4.beta['post'], LFS_NFB5.beta['post']]
        self.result = [round(num, 3) for num in LFS_NFB]
        self.expected = [0.046,0.007,0.045,-0.017,0.018]
        
    def test_count_eq_table_3LFSf(self):
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq_table_3LFSf(self):
        self.assertListEqual(self.result, self.expected)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

