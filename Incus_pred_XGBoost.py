#!/usr/bin/env python
# coding: utf-8

# # Environment Setup

# ## Once

# In[1]:


# Change output format
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#InteractiveShell.ast_node_interactivity = "last_expr"

# Import packages
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.cluster import KMeans
import json
import requests
#!pip install folium
import folium
import random
import imageio
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
import matplotlib.colors as colors
import math
from io import BytesIO
#from tabulate import tabulate
import time
from chinese_calendar import is_workday, is_holiday
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import plot_importance
import xgboost as xgb


# Other settings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows', None)
#sns.set(font='SimHei')

# -*- coding: utf-8 -*- 
#import cx_Oracle
import os
#os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
from collections import Counter
# df.to_pickle('./Cache/cache_df_nd_2019.pkl')
# pd.read_pickle('samples')


# ## Control

# In[2]:


#设置展示形式
#InteractiveShell.ast_node_interactivity = "last_expr"
InteractiveShell.ast_node_interactivity = "all"
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 20)

#设置目录
os.chdir('/Users/bryan/Documents/MetersBonwe/Data')
#os.chdir('/Users/bryan/Documents/Python/Data')
#weather = pd.read_csv('./Data/weather.csv', encoding = 'utf-16', delimiter = '\t')

#设置科学计数法
#np.set_printoptions(suppress=False)
np.set_printoptions(suppress=True)


# # Read Data

# ## 客流

# In[3]:


## inout data (BI数据)
# def load_bi_data(bi_name):
#     url = 'http://192.168.103.63:7777/analytics/saw.dll?GO&path=/users/hq01uh618/' + bi_name + '&Action=Extract&format=csv&NQUser=HQ01UH618&NQPassword=304558'
#     response = requests.get(url)
#     r = BytesIO(response.content)
#     return pd.read_csv(r, encoding = 'utf-16', delimiter = ',')

raw_cus_stm = pd.read_pickle('./spb_cus_inout_2018_now.pkl')
raw_cus_stm


# ## 天气

# In[4]:


raw_weather = pd.read_pickle('./weather_2018_now.pkl')
raw_weather


# In[5]:


#os.chdir('/project/datadir/wnb')
#weather = pd.read_csv('./Data/weather.csv', encoding = 'utf-16', delimiter = '\t')
#weather.shape

def clean_weather(df):
    #筛选城市
    df = df[df['县区'] == '重庆']
    #转换日期格式
    df['Date'] = pd.to_datetime(df['日历日(YYYY-MM-DD)'])
    #针对区县去重
    df = df.drop_duplicates(subset=['Date','城市'], keep='first')
    #筛选时间
    df = df[df['Date'] >= pd.to_datetime('2017-01-01')]
    df = df[df['Date'] <= pd.to_datetime('2021-01-06')]
    #dropna
    df = df.dropna()
    #天气情况
    weather_cons = ['雷','雨','云','阴','晴','雪']
    for i in weather_cons:
        df[i] = df.天气.map(lambda x: i in x)
        
    return df

weather = clean_weather(raw_weather)
weather.shape
weather.head(2)


# ## 进店人数客流分布

# In[529]:


os.chdir('/project/datadir/wnb')
in_dis = pd.read_csv('./Data/进店人数记录情况202006.csv', encoding = 'utf-16', delimiter = '\t')
in_dis.shape
in_dis.head()


# # EDA

# ## 天气

# In[526]:


sns.set(font = 'Simhei',style = 'ticks')
f = plt.figure(figsize = (20,6))
sns.countplot(weather.天气, order = weather['天气'].value_counts().index)
plt.xticks(rotation=270)
plt.title('沙坪坝旗舰店天气情况 - (20170101-20201231)', fontsize = 15)


# ## 人数分布

# In[447]:


f = plt.figure(figsize = (15,8))
sns.distplot(raw_cus_stm.进店人数, color = sns.color_palette("husl")[3])
plt.title('沙坪坝旗舰店进店人数分布情况')


# ## 气温变化

# In[525]:


sns.set(style = 'whitegrid', font = 'Simhei')
f = plt.figure(figsize = (15,5))
plt.title('沙坪坝旗舰店气温时间趋势')
sns.lineplot(x = 'Date', y = '平均温度', data = weather, color = sns.color_palette("husl")[3])


# ## 人数-时间趋势

# In[466]:


plot_data = raw_cus_stm
plot_data['Date'] = pd.to_datetime(plot_data['日历日(YYYY-MM-DD)'])
f = plt.figure(figsize = (15,7))
sns.lineplot(x = 'Date', y = '进店人数', data = plot_data, color = sns.color_palette("husl")[4])
plt.title('沙坪坝旗舰店进店人数 - 时间 趋势', fontsize = 15)


# # Feature Engineering

# In[6]:


## feature engineering:
def get_dist(date):
    frt_day = pd.to_datetime(date.year*10000+101, format = '%Y%m%d')
    dist = (date - frt_day).days
    return dist

def feat_eng(input_df):
    # 加入天气
    df = input_df.merge(weather, how = 'left', on = '日历日(YYYY-MM-DD)')
    # 去除气温为空
    df = df[df['平均温度'].notnull()]
    # 加入是否是工作日
    df['is_workday'] = df.Date.apply(is_workday)
    # 年
    df['year'] = df.Date.map(lambda x: x.year)
    # 月
    df['month'] = df.Date.map(lambda x: x.month)
    # 日
    df['day'] = df.Date.map(lambda x: x.day)
    # 一日气温变化
    df['one_day_var'] = abs(df['平均温度'] - df['平均温度'].shift(1))
    # 三日气温变化
    df['three_day_var'] = abs(df['平均温度'] - df['平均温度'].shift(3))
    # 去除空值
    df = df.dropna()
    # 计算距离
    df['dist'] = df.Date.apply(get_dist)
    #df = df.drop(['日历日(YYYY-MM-DD)','门店ID','出店人数','Date','省份','城市','县区','天气'], axis=1)
    #筛选时间
    #df = df[df['Date'] >= pd.to_datetime('2017-01-01')]
    df = df[(df['Date'] <= pd.to_datetime('2019-11-01'))|(df['Date'] >= pd.to_datetime('2020-04-01'))]
    df = df[df['Date'] <= pd.to_datetime('2021-01-06')]    
    #df = df[df.进店人数<=8000]
    
    return df

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

cus_stm = feat_eng(raw_cus_stm)
cus_stm.head(2)
cus_stm.shape


# In[7]:


# # 划分数据并建立预测模型
features = cus_stm.drop(['日历日(YYYY-MM-DD)','门店ID','出店人数','Date','省份','城市','县区','天气','进店人数'], axis=1)
target = cus_stm['进店人数']

from sklearn.model_selection import train_test_split

# 将数据切分成训练集和测试集
x_train, x_test,y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 0)


# # Training Model

# ## Model_1

# In[8]:


# model_1
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

# training
Model_1 = XGBRegressor(learning_rate=0.1,n_estimators=100,max_depth=8)
start = time.clock()
Model_1.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)])
end = time.clock()
print('Model_1 Running time: %s Seconds'%(end-start))


# #### Model 1 result & eval

# In[9]:


y_pred = Model_1.predict(x_test.values)
Test_Pred = Model_1.predict(features.values)
print("Evaluation Metrics:" )
print('R square: '+str(r2_score(y_test,y_pred)))
print('RMSE After Gradient Desent: 738\n')


print('Whole DF with pred:')
result = cus_stm.drop(['日历日(YYYY-MM-DD)','门店ID','出店人数','省份','城市','县区'], axis=1)
result['f3_model1'] = Test_Pred
result['f3_model1_var'] = result['f3_model1'] - result['进店人数']
result[result['f3_model1_var']>500]


print('Test set with pred')
test_set_result = pd.DataFrame(y_pred)
test_set_result['pred'] = list(y_test)
test_set_result.head(30)


# In[342]:


# Feature Importance
f = plt.figure(figsize = (15,5))
plt.title('F3M1 Feature Importance Chart', fontsize = 15)
sns.barplot(x = list(features.columns), y = Model_1.feature_importances_)


from xgboost import plot_importance
plot_importance(Model_1)


# ## Model_2

# In[334]:


# Model_2
Model_2 = XGBRegressor(learning_rate=0.1,n_estimators=100,max_depth=6,min_child_weight = 1,
                     subsample=0.8,colsample_btree=0.8,objective='reg:linear',
                     scale_pos_weight=1,random_state=27)
start=time.clock()
Model_2.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
          early_stopping_rounds = 10)
end=time.clock()
print('Model_2 Running time: %s Seconds'%(end-start))

### make prediction for test data
y_pred = Model_2.predict(x_test.values)
error = rmspe(y_test,y_pred)
print("accuarcy: %.2f%%" % (error*100.0))


# #### Model 2 result & eval

# In[335]:


from sklearn.metrics import r2_score
y_pred = Model_2.predict(x_test.values)
Test_Pred = Model_2.predict(features.values)
print("Evaluation Metrics:" )
print('R square: '+str(r2_score(y_test,y_pred)))
print('RMSE After Gradient Desent: 611\n')


print('Whole DF with pred:')
result = cus_stm.drop(['日历日(YYYY-MM-DD)','门店ID','出店人数','省份','城市','县区'], axis=1)
result['f3_model1'] = Test_Pred
result['f3_model1_var'] = result['f3_model1'] - result['进店人数']
result.head()
#result[result['f3_model1_var']< -500]


print('Test set with pred:')
test_set_result = pd.DataFrame(y_pred)
test_set_result['test'] = list(y_test)
test_set_result.columns = ['Pred','Actual']
test_set_result.head()


# In[343]:


# Feature Importance
f = plt.figure(figsize = (15,5))
plt.title('F3M2 Feature Importance Chart', fontsize = 15)
sns.barplot(x = list(features.columns), y = Model_2.feature_importances_)

from xgboost import plot_importance
plot_importance(Model_2)


# ## Model_3

# In[345]:


# # 网格搜索方法调整模型的最佳参数
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

parameters = {
    'min_child_weight':[2,3,4],   
    'colsample_bytree':[0.5,0.7,1],
    'scale_pos_weight':[0.6,0.7,0.8]
}

xlf = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=6, min_child_weight = 1,
                     subsample=0.8, colsample_btree=0.8, objective='reg:linear',
                     scale_pos_weight=1,random_state=27)
n_iter_search = 5

gsearch = RandomizedSearchCV(xlf, param_distributions=parameters, n_iter=n_iter_search, cv=2, iid=False)

start=time.clock()
gsearch.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 20)
end=time.clock()
print('RandomSearch Running time: %s Seconds'%(end-start))

print("Best score: %0.3f" % gsearch.best_score_)
best_estimator = gsearch.best_estimator_
print("Best parameters set" )
best_estimator


# In[347]:


Model_3=XGBRegressor(base_score=0.5, booster='gbtree', colsample_btree=0.8,
       colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
       gamma=0, importance_type='gain', learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=4, missing=None,
       n_estimators=100, n_jobs=40, nthread=None, objective='reg:linear',
       random_state=27, reg_alpha=0, reg_lambda=1, scale_pos_weight=0.6,
       seed=None, silent=None, subsample=0.8, verbosity=None)

start=time.clock()
Model_3.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 100)
end=time.clock()
print('Model_3 Running time: %s Seconds'%(end-start))



### make prediction for test data
y_pred = Model_3.predict(x_test.values)
error = rmspe(y_test,y_pred)
print("accuarcy: %.2f%%" % (error*100.0))


# In[349]:


from sklearn.metrics import r2_score
y_pred = Model_3.predict(x_test.values)
Test_Pred = Model_3.predict(features.values)
print("Evaluation Metrics:" )
print('R square: '+str(r2_score(y_test,y_pred)))
print('RMSE After Gradient Desent: 586\n')


print('Whole DF with pred:')
result = cus_stm.drop(['日历日(YYYY-MM-DD)','门店ID','出店人数','省份','城市','县区'], axis=1)
result['f3_model3'] = Test_Pred
result['f3_model3_var'] = result['f3_model3'] - result['进店人数']
result.head()
#result[result['f3_model1_var']< -500]


print('Test set with pred:')
test_set_result = pd.DataFrame(y_pred)
test_set_result['test'] = list(y_test)
test_set_result.columns = ['Pred','Actual']
test_set_result.head()


# ## Model_4

# ### 调参

# In[350]:


# 网格搜索法得到的模型精度不高，说明模型精度和n_estimators和max_depth有很紧密的联系，再次选用网格搜索法仅对这两个参数进行选择
parameters = {
    'n_estimators':[100,200,300,400],   
    'max_depth':[5,7,9]  
}

xlf = XGBRegressor(learning_rate=0.1,n_estimators=100,max_depth=6,min_child_weight = 1,
                     subsample=0.8,colsample_btree=0.8,objective='reg:linear',
                     scale_pos_weight=1,random_state=27)
n_iter_search = 5

gsearch = RandomizedSearchCV(xlf,param_distributions=parameters,n_iter=n_iter_search, cv=2, iid=False)

start=time.clock()
gsearch.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 20)
end=time.clock()
print('RandomSearch Running time: %s Seconds'%(end-start))

print("Best score: %0.3f" % gsearch.best_score_)
best_estimator = gsearch.best_estimator_
print("Best parameters set" )
best_estimator


# ### run模型

# In[351]:


Model_4=XGBRegressor(base_score=0.5, booster='gbtree', colsample_btree=0.8,
       colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
       gamma=0, importance_type='gain', learning_rate=0.1,
       max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
       n_estimators=400, n_jobs=40, nthread=None, objective='reg:linear',
       random_state=27, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=None, subsample=0.8, verbosity=1)

start=time.clock()
Model_4.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 100)
end=time.clock()
print('Model4 Running time: %s Seconds'%(end-start))

y_pred = Model_4.predict(x_test.values)
error = rmspe(y_test,y_pred)
print("accuarcy: %.2f%%" % (error*100.0))

print("Make predictions on the test set")
Test_Pred = Model_4.predict(test_new_final.values)

#result = pd.DataFrame({"Id": test_new['Id'], 'Sales':np.expm1(Test_Pred)})
#result.to_csv("Rossmann_submission_Model_4.csv", index=False)


# ### 评估

# In[352]:


from sklearn.metrics import r2_score
y_pred = Model_4.predict(x_test.values)
Test_Pred = Model_4.predict(features.values)
print("Evaluation Metrics:" )
print('R square: '+str(r2_score(y_test,y_pred)))
print('RMSE After Gradient Desent: 562\n')


print('Whole DF with pred:')
result = cus_stm.drop(['日历日(YYYY-MM-DD)','门店ID','出店人数','省份','城市','县区'], axis=1)
result['f3_model4'] = Test_Pred
result['f3_model4_var'] = result['f3_model4'] - result['进店人数']
result.head()
#result[result['f3_model1_var']< -500]


print('Test set with pred:')
test_set_result = pd.DataFrame(y_pred)
test_set_result['test'] = list(y_test)
test_set_result.columns = ['Pred','Actual']
test_set_result.head()


# ## Model_5

# ### 调参

# ### 调树深和剪枝

# In[356]:


param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

xlf = XGBRegressor(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight = 1,
                     gamma=0, subsample=0.8, colsample_btree=0.8,objective='reg:linear',
                     scale_pos_weight=1,random_state=27)
n_iter_search = 5

gsearch = RandomizedSearchCV(xlf,param_distributions=param_test2,n_iter=n_iter_search, cv=2, iid=False)

start=time.clock()
gsearch.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 20)
end=time.clock()
print('RandomSearch Running time: %s Seconds'%(end-start))

print("Best score: %0.3f" % gsearch.best_score_)
best_estimator = gsearch.best_estimator_
print("Best parameters set" )
best_estimator


# In[359]:


param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

xlf = XGBRegressor(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight = 1,
                     gamma=0, subsample=0.8, colsample_btree=0.8,objective='reg:linear',
                     scale_pos_weight=1,random_state=27)
n_iter_search = 5

gsearch = RandomizedSearchCV(xlf,param_distributions=param_test1,n_iter=n_iter_search, cv=2, iid=False)

start=time.clock()
gsearch.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 20)
end=time.clock()
print('RandomSearch Running time: %s Seconds'%(end-start))

print("Best score: %0.3f" % gsearch.best_score_)
best_estimator = gsearch.best_estimator_
print("Best parameters set" )
best_estimator


# ### 调gamma

# In[361]:


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

xlf = XGBRegressor(learning_rate=0.1,n_estimators=1000,max_depth=7,min_child_weight = 5,
                     gamma=0, subsample=0.8, colsample_btree=0.8,objective='reg:linear',
                     scale_pos_weight=1,random_state=27)
n_iter_search = 5

gsearch = RandomizedSearchCV(xlf,param_distributions=param_test3,n_iter=n_iter_search, cv=5, iid=False)

start=time.clock()
gsearch.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 20)
end=time.clock()
print('RandomSearch Running time: %s Seconds'%(end-start))

print("Best score: %0.3f" % gsearch.best_score_)
best_estimator = gsearch.best_estimator_
print("Best parameters set" )
best_estimator


# ### 调subsample和colsample_bytree

# In[362]:


param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

xlf = XGBRegressor(learning_rate=0.1,n_estimators=1000,max_depth=7,min_child_weight = 5,
                     gamma=0, subsample=0.8, colsample_btree=0.8,objective='reg:linear',
                     scale_pos_weight=1,random_state=27)
n_iter_search = 5

gsearch = RandomizedSearchCV(xlf,param_distributions=param_test4,n_iter=n_iter_search, cv=5, iid=False)

start=time.clock()
gsearch.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 20)
end=time.clock()
print('RandomSearch Running time: %s Seconds'%(end-start))

print("Best score: %0.3f" % gsearch.best_score_)
best_estimator = gsearch.best_estimator_
print("Best parameters set" )
best_estimator


# ### 调正则项

# In[363]:


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

xlf = XGBRegressor(learning_rate=0.1,n_estimators=1000,max_depth=7,min_child_weight = 5,
                     gamma=0, subsample=0.8, colsample_btree=0.8,objective='reg:linear',
                     scale_pos_weight=1,random_state=27)
n_iter_search = 5

gsearch = RandomizedSearchCV(xlf,param_distributions=param_test6,n_iter=n_iter_search, cv=5, iid=False)

start=time.clock()
gsearch.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 20)
end=time.clock()
print('RandomSearch Running time: %s Seconds'%(end-start))

print("Best score: %0.3f" % gsearch.best_score_)
best_estimator = gsearch.best_estimator_
print("Best parameters set" )
best_estimator


# ## Model_S

# ### 模型调参

# In[33]:


Model_S = XGBRegressor(base_score=0.5, booster='gbtree', colsample_btree=0.8,
       colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.8,
       gamma=0, importance_type='gain', learning_rate=0.01,
       max_delta_step=0, max_depth=7, min_child_weight=5, missing=None,
       n_estimators=5000, n_jobs=40, nthread=None, objective='reg:linear',
       random_state=27, reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=None, subsample=0.8, verbosity=1)

start = time.clock()
Model_S.fit(x_train.values,y_train.values,eval_set = [(x_test.values,y_test.values)],eval_metric = "rmse",
            early_stopping_rounds = 100)
end=time.clock()
print('Model4 Running time: %s Seconds'%(end-start))

y_pred = Model_S.predict(x_test.values)
error = rmspe(y_test,y_pred)
print("accuarcy: %.2f%%" % (error*100.0))

print("Make predictions on the test set")
Test_Pred = Model_S.predict(features.values)

#result = pd.DataFrame({"Id": test_new['Id'], 'Sales':np.expm1(Test_Pred)})
#result.to_csv("Rossmann_submission_Model_4.csv", index=False)


# In[34]:


from sklearn.metrics import r2_score
y_pred = Model_S.predict(x_test.values)
Test_Pred = Model_S.predict(features.values)
print("Evaluation Metrics:" )
print('R square: '+str(r2_score(y_test,y_pred)))
print('RMSE After Gradient Desent: 553\n')


print('Whole DF with pred:')
result = cus_stm.drop(['日历日(YYYY-MM-DD)','门店ID','出店人数','省份','城市','县区'], axis=1)
result['f3_modelS'] = Test_Pred
result['f3_modelS_var'] = result['f3_modelS'] - result['进店人数']
result.head()
#result[result['f3_model1_var']< -500]


print('Test set with pred:')
test_set_result = pd.DataFrame(y_pred)
test_set_result['test'] = list(y_test)
test_set_result.columns = ['Pred','Actual']
test_set_result.head()


# In[382]:


test_set = pd.concat([x_test, y_test], axis = 1)
test_set['pred'] = y_pred
test_set['var'] = test_set['pred'] - test_set['进店人数']
test_set


# ### 模型评估

# In[396]:


sns.set(style = 'whitegrid')
f = plt.figure(figsize = (20,5))
plt.title('Prediction Result On Test Set', fontsize = 15)
sns.lineplot(x = range(166), y = 'pred', data = test_set, color = sns.color_palette("Set2")[0])
sns.lineplot(x = range(166), y = '进店人数', data = test_set, color = sns.color_palette("Set2")[1])


# In[522]:


f = plt.figure(figsize = (20,6))
#fig, axes = plt.subplots(nrows = 1,ncols=2)
#sns.set_theme(style = 'whitegrid')

ax = sns.catplot(x = 'is_workday', y = 'var', hue = 'year', data = test_set)
plt.title('预测结果variance分布情况 - 1', font = 'Simhei', fontsize = 15)
#ax = sns.violinplot(x = 'is_workday', y = 'var', data = test_set)
plt.show()


# In[512]:


sns.set(style = 'whitegrid')
f = plt.figure(figsize = (8,6))
#fig, axes = plt.subplots(nrows = 1,ncols=2)
#sns.set_theme(style = 'whitegrid')

#ax = sns.catplot(x = 'is_workday', y = 'var', hue = 'year', data = test_set)
plt.title('预测结果variance分布情况 - 2', font = 'Simhei', fontsize = 15)

ax = sns.violinplot(x = 'is_workday', y = 'var', data = test_set, palette = sns.color_palette("Set2"))
plt.show()


# In[367]:


# Feature Importance
f = plt.figure(figsize = (15,5))
plt.title('F3MS Feature Importance Chart', fontsize = 15)
sns.barplot(x = list(features.columns), y = Model_S.feature_importances_)

from xgboost import plot_importance
plot_importance(Model_2)

