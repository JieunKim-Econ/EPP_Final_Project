#!/usr/bin/env python
# coding: utf-8

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


##============[Fertility]
##====================Conceptions========================
# Variables
# mesp: Month of birth
# year: Year of birth
# prem: Prematurity indicator
# semanas: Weeks of gestation at birth

etdf=pd.read_stata("data/data_births_20110196.dta")


# Create month of birth variable: (0 = July 2007, 1 = August 2007, etc)

etdf.loc[(etdf['year']==2010), 'm'] = etdf['mesp'] +29
etdf.loc[(etdf['year']==2009), 'm'] = etdf['mesp'] + 17 
etdf.loc[(etdf['year']==2008), 'm'] = etdf['mesp'] + 5  
etdf.loc[(etdf['year']==2007), 'm'] = etdf['mesp'] - 7  
etdf.loc[(etdf['year']==2006), 'm'] = etdf['mesp'] - 19
etdf.loc[(etdf['year']==2005), 'm'] = etdf['mesp'] - 31 
etdf.loc[(etdf['year']==2004), 'm'] = etdf['mesp'] - 43 
etdf.loc[(etdf['year']==2003), 'm'] = etdf['mesp'] - 55 
etdf.loc[(etdf['year']==2002), 'm'] = etdf['mesp'] - 67
etdf.loc[(etdf['year']==2001), 'm'] = etdf['mesp'] - 79 
etdf.loc[(etdf['year']==2000), 'm'] = etdf['mesp'] - 91 

getdf=etdf.groupby('m',as_index=False)['m'].agg({'n':'count'})

def figure_ET2(data):
    
    fig = plt.figure(figsize = (10, 5)) 
    plt.grid(True)
    plt.ylim(0, 52000) 
    birth = list(getdf.n) 
    month = list(getdf.m) 

    # creating the bar plot 
    plt.bar(month, birth, color ='orange',  width = 1) 
    plt.axvline(x=0, color='salmon')

    
    plt.fill_betweenx(y=range(44000), x1=-30,x2=30, alpha=0.2, facecolor='c', label = '60 months around the cutoff')
    plt.fill_betweenx(y=range(44000), x1=-20,x2=20, alpha=0.4, facecolor='c', label = '40 months')
    plt.fill_betweenx(y=range(44000), x1=-9,x2=9, alpha=0.6, facecolor='c', label = '18 motnhs ')
    plt.legend(loc='best')
    
    plt.xlabel("m=month of birth (0=July 2007)") 
    plt.ylabel("No. of Births") 
    plt.title("Figure ET2. Discontinuity check in births") 
    plt.show() 
    
    return  
