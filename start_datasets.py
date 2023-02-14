import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns
#import warning
%matplotlib inline

#warning.ignorewarning(...)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

import os, sys
from google.colab import drive

drive.mount('/content/gdrive')

%cd '/content/gdrive/My Drive/'
!ls

#default_dir = "/content/gdrive/My Drive"

app_train = pd.read_csv('application_train.csv')
app_test = pd.read_csv('application_test.csv')
app_train.head()

app_train.shape, app_test.shape

####

app_train['AMT_INCOME_TOTAL'].hist()
plt.hist(app_train['AMT_INCOME_TOTAL'])
sns.distplot(app_train['AMT_INCOME_TOTAL'])
sns.boxplot(app_train['AMT_INCOME_TOTAL'])

####

cond_1 = app_train['AMT_INCOME_TOTAL'] < 1000000
app_train[cond_1]['AMT_INCOME_TOTAL'].hist()
sns.distplot(app_train[cond_1]['AMT_INCOME_TOTAL'])

####

cond1 = (app_train['TARGET'] ==1)
cond0 = (app_train['TARGET'] ==0)

cond_amt = (app_train['AMT_INCOME_TOTAL'] < 500000)
sns.distplot(app_train[cond0 & cond_amt]["AMT_INCOME_TOTAL"], label = '0', color = 'blue')
sns.distplot(app_train[cond1 & cond_amt]["AMT_INCOME_TOTAL"], label = '1', color = 'red')

####

sns.violinplot(x = 'TARGET', y = 'AMT_INCOME_TOTAL', data = app_train[cond_amt])

fig, axs = plt.subplots(figsize = (12, 4), nrows = 1, ncols = 2, squeeze = False)
cond1 = (app_train['TARGET'] ==1)
cond0 = (app_train['TARGET'] ==0)


sns.violinplot(x = 'TARGET', y = 'AMT_INCOME_TOTAL', data = app_train[cond_amt], ax = axs[0][0])

cond_amt = (app_train['AMT_INCOME_TOTAL'] < 500000)
sns.distplot(app_train[cond0 & cond_amt]["AMT_INCOME_TOTAL"], label = '0', color = 'blue', ax = axs[0][1])
sns.distplot(app_train[cond1 & cond_amt]["AMT_INCOME_TOTAL"], label = '1', color = 'red', ax = axs[0][1])

####

def show_column_hist_by_target(df, column, is_amt = False):

  cond1 = (df['TARGET'] ==1)
  cond0 = (df['TARGET'] ==0)

  fig, axs = plt.subplots(figsize = (12, 4), nrows = 1, ncols = 2, squeeze = False)
 
  cond_amt = True
  if is_amt:
    cond_amt = df[column] < 500000

  sns.violinplot(x = 'TARGET', y = column, data = df[cond_amt], ax = axs[0][0])
  
  cond_amt = (app_train['AMT_INCOME_TOTAL'] < 500000)
  sns.distplot(df[cond0 & cond_amt][column], label = '0', color = 'blue', ax = axs[0][1])
  sns.distplot(df[cond1 & cond_amt][column], label = '1', color = 'red', ax = axs[0][1])

show_column_hist_by_target(app_train, 'AMT_INCOME_TOTAL', is_amt = True)
