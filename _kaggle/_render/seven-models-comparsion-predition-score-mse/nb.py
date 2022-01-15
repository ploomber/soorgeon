# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns


# %% [markdown]
# # **1. Read the Data**

# %%
Train_data = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/train.csv')
Test_data = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/test.csv')
train_data = Train_data.copy()
test_data = Test_data.copy()


# %% [markdown]
# # **2. Data Proprecessing**
#
# * I Split the 'date' column to 'Month' column and 'Day' column, I didn't consider the Year.
#
# * I define the 'weekday' and 'weekend' columns from 'date' columns.
#
# * I dummies the 'country', 'store', 'product' columns.

# %%
#Month
def split_month(date):
  return date.split('-')[1]

#Day
def split_day(date):
  return date.split('-')[2]

#Weekend
def weekend(date):
  import datetime
  weekend = []
  a = pd.to_datetime(date)
  for i in range(len(a)):
    if a.iloc[i].weekday() >= 5 :
      weekend.append(1)
    else:
      weekend.append(0)
  return weekend

#Weekday
def weekday(date):
    import datetime
    weekday = []
    a = pd.to_datetime(date)
    for i in range(len(a)):
        weekday.append(a.iloc[i].weekday())
    return weekday


# %%
train_data['Month'] = train_data['date'].apply(split_month)
train_data['Day'] = train_data['date'].apply(split_day)
train_data['Weekend'] = weekend(train_data['date'])
train_data['Weekday'] = weekday(train_data['date'])
train_data = train_data.drop(columns = ['row_id', 'date'])

test_data['Month'] = test_data['date'].apply(split_month)
test_data['Day'] = test_data['date'].apply(split_day)
test_data['Weekend'] = weekend(test_data['date'])
test_data['Weekday'] = weekday(test_data['date'])
test_data = test_data.drop(columns = ['row_id', 'date'])

# %%
#Dummies the 'country', 'store', 'product'
train_data_dum = pd.get_dummies(train_data[['country', 'store', 'product']])
test_data_dum = pd.get_dummies(test_data[['country', 'store', 'product']])

train_data = pd.concat([train_data, train_data_dum],axis = 1)
test_data = pd.concat([test_data, test_data_dum],axis = 1)

train_data = train_data.drop(columns = ['country', 'store', 'product'])
test_data = test_data.drop(columns = ['country', 'store', 'product'])

# %% [markdown]
# ### Define the training data and training target

# %%
data = train_data.drop(columns = 'num_sold')
target = train_data['num_sold']

from sklearn.preprocessing import StandardScaler
Normalize = StandardScaler()
target = np.log(target)

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.8, random_state = 5)
x_train = Normalize.fit_transform(x_train)
x_test = Normalize.transform(x_test)


# %% [markdown]
# # **3. Seven Model Performance**
#

# %% [markdown]
# ## 3-1. DecisionTreeRegressor

# %%
from sklearn.tree import DecisionTreeRegressor
sns.set()
DTR = DecisionTreeRegressor(max_depth = 12, min_samples_leaf = 2).fit(x_train, y_train)
y_pred_DTR = DTR.predict(x_test)
plt.scatter(y_test, y_pred_DTR)
plt.plot([x for x in range(4, 10)], [x for x in range(4, 10)], color = 'r')
plt.xlabel("Reality")
plt.ylabel("Predicted")
plt.title('DecisionTreeRegressor')
plt.show()
plt.clf()

# %% [markdown]
# ## 3-2. RandomForestRegressor

# %%
from sklearn.ensemble import RandomForestRegressor
sns.set()
RFR = RandomForestRegressor(max_depth = 15, random_state = 2).fit(x_train, y_train)
y_pred_RFR = RFR.predict(x_test)
plt.scatter(y_test, y_pred_RFR)
plt.plot([x for x in range(4, 10)], [x for x in range(4, 10)], color = 'r')
plt.xlabel("Reality")
plt.ylabel("Predicted")
plt.title('RandomFroestRegressor')
plt.show()
plt.clf()

# %% [markdown]
# ## 3-3. GradientBoostingRegressor

# %%
from sklearn.ensemble import GradientBoostingRegressor
sns.set()
GBR = GradientBoostingRegressor(learning_rate=0.10, max_depth= 6,
                                min_samples_leaf = 5,n_estimators = 500, random_state = 40,subsample = 0.3).fit(x_train, y_train)
y_pred_GBR = GBR.predict(x_test)
plt.scatter(y_test, y_pred_GBR)
plt.plot([x for x in range(4, 10)], [x for x in range(4, 10)], color = 'r')
plt.xlabel("Reality")
plt.ylabel("Predicted")
plt.title('GradientBoostingRegressor')
plt.show()
plt.clf()
print(GBR.score(x_test, y_test))

# %% [markdown]
# ## 3-4. SVR rbf

# %%
from sklearn.svm import SVR
sns.set()
svr_rbf = SVR(kernel = 'rbf', gamma = 0.2 , C = 0.15, degree = 2, epsilon=0.1).fit(x_train, y_train)
y_pred_svr = svr_rbf.predict(x_test)
plt.scatter(y_test, y_pred_svr)
plt.plot([x for x in range(4, 10)], [x for x in range(4, 10)], color = 'r')
plt.xlabel("Reality")
plt.ylabel("Predicted")
plt.title('SVM_RBF')
plt.show()
plt.clf()

# %% [markdown]
# ## 3-5. SVR Linear

# %%
from sklearn.svm import SVR
sns.set()
svr_linear = SVR(kernel = 'linear').fit(x_train, y_train)
y_pred_svr = svr_linear.predict(x_test)
plt.scatter(y_test, y_pred_svr)
plt.plot([x for x in range(4, 10)], [x for x in range(4, 10)], color = 'r')
plt.xlabel("Reality")
plt.ylabel("Predicted")
plt.title('SVM_Linear')
plt.show()
plt.clf()

# %% [markdown]
# ## 3-6. Deep Learning

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
data = Normalize.fit_transform(data)
model = Sequential()
model.add(Dense(512, input_shape = (data.shape[1], ), activation = 'sigmoid'))
model.add(Dense(64,activation = 'sigmoid'))
model.add(Dense(8))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam', metrics= 'mse')
history = model.fit(data, target, batch_size = 128, epochs = 100 , validation_split= 0.2, verbose = 1)

# %%
import seaborn as sns
sns.set()
df_history = pd.DataFrame(history.history)
sns.lineplot(x = df_history.index, y = df_history.loss)

# %%
y_pred = model.predict(data)
plt.scatter(target, y_pred)
plt.plot([x for x in range(4, 10)], [x for x in range(4, 10)], color = 'r')
plt.xlabel("Reality")
plt.ylabel("Predicted")
plt.show()
plt.clf()

# %% [markdown]
# ## 3-7. KNeighborsRegressor

# %%
from sklearn.neighbors import KNeighborsRegressor
sns.set()
KNN = KNeighborsRegressor(n_neighbors = 3, weights = 'distance').fit(x_train, y_train)
y_pred_KNN = KNN.predict(x_test)
plt.scatter(y_test, y_pred_KNN)
plt.plot([x for x in range(4, 10)], [x for x in range(4, 10)], color = 'r')
plt.xlabel("Reality")
plt.ylabel("Predicted")
plt.title('KNeighborsRegressor')
plt.show()
plt.clf()

# %% [markdown]
# # **4. Six Model Comparsion(R2_score and Mean squared error)**

# %%
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
def model_fit(x_train, x_test, y_train, y_test):

  DTR = DecisionTreeRegressor(max_depth = 10, 
                              min_samples_leaf = 8).fit(x_train, y_train)

  RFR = RandomForestRegressor(max_depth = 30).fit(x_train, y_train)
    
  GBR = GradientBoostingRegressor(learning_rate=0.10, max_depth= 6,
                                min_samples_leaf = 5,n_estimators = 500, random_state = 40,subsample = 0.3).fit(x_train, y_train)

  svr_rbf = SVR(kernel = 'rbf', 
                gamma = 0.2 , 
                C = 0.15, 
                degree = 2, 
                epsilon=0.1).fit(x_train, y_train)  

  svr_linear = SVR(kernel = 'linear').fit(x_train, y_train)

  KNN = KNeighborsRegressor(n_neighbors = 10).fit(x_train, y_train)

  return DTR, RFR, GBR, svr_rbf, svr_linear, KNN


# %% [markdown]
# ## 4-1. R square score Comparsion

# %%
Model = model_fit(x_train, x_test, y_train, y_test)
ML_model = ['DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor', 'svr_rbf', 'svr_linear', 'KNeighborsRegressor', 'DeepLearning']
sns.set()
from sklearn.metrics import r2_score
R_square_num = []
for i in range(6):
  R_square = r2_score(y_test, Model[i].predict(x_test))
  R_square_num.append(R_square)
R_square_num.append(r2_score(y_test, model.predict(x_test)))
plt.figure(figsize = (10, 10))
plt.xlabel('R Square Score')
plt.ylabel('Model Type')
plt.title('The R Square Score Comparsion')
sns.barplot(x = R_square_num, y = ML_model)

# %% [markdown]
# ## 4-2. Mean Square Root Comparsion

# %%
Model = model_fit(x_train, x_test, y_train, y_test)
ML_model = ['DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor', 'svr_rbf', 'svr_linear', 'KNeighborsRegressor', 'DeepLearning']
sns.set()
from sklearn.metrics import mean_squared_error
mse_num = []
for i in range(6):
  mse = mean_squared_error(y_test, Model[i].predict(x_test))
  mse_num.append(mse)
mse_num.append(mean_squared_error(y_test, model.predict(x_test)))
plt.figure(figsize = (10, 10))
plt.xlabel('mean_square_error')
plt.ylabel('Model Type')
plt.title('The mean_square_error Comparsion')
sns.barplot(x = mse_num, y = ML_model)

# %% [markdown]
# # **5. Predict the Test_data**

# %%
test_data = Normalize.transform(test_data)
submission_target = np.exp(Model[2].predict(test_data))

# %% [markdown]
# # **6. Submission**

# %%
submission = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/sample_submission.csv')
submission['num_sold'] = submission_target
submission.to_csv('submission.csv', index=False)

# %%
submission

# %% [markdown]
# # **7. Thanks for your view!**
