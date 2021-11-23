# -*- coding: utf-8 -*-
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

# %% [markdown] papermill={"duration": 0.049112, "end_time": "2021-01-17T15:24:40.020719", "exception": false, "start_time": "2021-01-17T15:24:39.971607", "status": "completed"} tags=[]
# # Introduction
# In this notebook, we will learn how to work with and predict time series. Time series are a collection of **time-dependent data** points. That means that each data point is assigned to a specific timestamp. Ideally, these data points are in chronological order and in contant time intervals (e.g. every minute or everyday). The time series forecasting problem **analyzes patterns in the past data to make predictions about the future**. The most popular example is probably stock price prediction. Other examples are sales of seasonal clothing or weather forecasts. In contrast to regression problems, time series are time-dependent and show specific characteristics, such as **trend and seasonality**.
#
# **Overview**
# * [Problem Definition](#Problem-Definition)<br>
# * [Data Collection](#Data-Collection)<br>
# * [Data Preprocessing](#Data-Preprocessing)<br>
#     * [Chronological Order and Equidistant Timestamps](#[Chronological-Order-and-Equidistant-Timestamps])<br>
#     * [Handling Missing Values](#Handling-Missing-Values)<br>
#     * [Resampling](#Resampling)<br>
#     * [Stationarity](#Stationarity)<br>
# * [Feature Engineering](#Feature-Engineering)<br>
#     * [Time Features](#Time-Features)<br>
#     * [Decomposition](#Decomposition)<br>
#     * [Lag](#Lag)<br>
# * [Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
#     * [Autocorrelation Analysis](#Autocorrelation-Analysis)<br>
# * [Cross Validation](#Cross-Validation)<br>
# * [Models](#Models)<br>
#     * [Models for Univariate Time Series](#Models-for-Univariate-Time-Series)<br>
#         * [Naive Approach](#Naive-Approach)<br>
#         * [Moving Average](#Moving-Average)<br>
#         * [Exponential Smoothing  (IN WORK)](#MExponential-Smoothing)<br>
#         * [ARIMA](#ARIMA)<br>
#     * [Models for Multivariate Time Series](#Models-for-Multivariate-Time-Series)<br>
#         * [Vector Autoregression (VAR)](#Vector-Autoregression)<br>

# %% [markdown] papermill={"duration": 0.045927, "end_time": "2021-01-17T15:24:40.113569", "exception": false, "start_time": "2021-01-17T15:24:40.067642", "status": "completed"} tags=[]
# # Problem Definition
# For this tutorial, we will build a model to predict the depth to groundwater of an aquifer located in Petrignano, Italy. The question we want to answer is
# > What is the future depth to groundwater of a well belonging to the aquifier in Petrigrano over the next quarter?
#
# > The wells field of the alluvial plain between Ospedalicchio di Bastia Umbra and Petrignano is fed by three underground aquifers separated by low permeability septa. The aquifer can be considered a water table groundwater and is also fed by the Chiascio river. The groundwater levels are influenced by the following parameters: rainfall, depth to groundwater, temperatures and drainage volumes, level of the Chiascio river.
#
# > Indeed, both rainfall and temperature affect features like level, flow, depth to groundwater and hydrometry some time after it fell down.
#
# # Data Collection
# In a typical workflow for time series, this would be the time for data collection. In this example, we will skip the data collection step and use data from the [Acea Smart Water Analytics challenge](https://www.kaggle.com/c/acea-water-prediction/). Therefore, this section will be a dataset overview.
#
# Although the dataset contains multiple waterbodies, we will only be looking at the Aquifer_Petrignano.csv file.
#
# Time series data usually comes in **tabular** format (e.g. csv files).

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _kg_hide-input=true _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 1.178488, "end_time": "2021-01-17T15:24:41.339596", "exception": false, "start_time": "2021-01-17T15:24:40.161108", "status": "completed"} tags=[]
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # Visualization
import matplotlib.pyplot as plt  # Visualization

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

import warnings  # Supress warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("input/Aquifer_Petrignano.csv")

### Simplifications for the sake of the tutorial ###
# Drop data before 2009 for the purpose of this tutorial
df = df[df.Rainfall_Bastia_Umbra.notna()].reset_index(drop=True)
# Drop one of the target columns, so we can focus on only one target
df = df.drop(['Depth_to_Groundwater_P24', 'Temperature_Petrignano'], axis=1)

# Simplify column names
df.columns = [
    'Date', 'Rainfall', 'Depth_to_Groundwater', 'Temperature',
    'Drainage_Volume', 'River_Hydrometry'
]

targets = ['Depth_to_Groundwater']
features = [feature for feature in df.columns if feature not in targets]
df.head()

# %% [markdown] papermill={"duration": 0.04978, "end_time": "2021-01-17T15:24:41.437658", "exception": false, "start_time": "2021-01-17T15:24:41.387878", "status": "completed"} tags=[]
# Since we are working with time series, the most essential features are the time related feature. In this example, we have the column `Date` which  uniquely identifies a day. Ideally, the data is already in chronological order and the time stamps are equidistant in time series. This is already the case in our data: The time interval is one day and the data is already in chronological order. Therefore, we do not have to do this additional data preparation step.
#
#
# This column is provided in string format. Let's convert it to the `datetime64[ns]` data type.

# %% papermill={"duration": 0.232976, "end_time": "2021-01-17T15:24:41.719795", "exception": false, "start_time": "2021-01-17T15:24:41.486819", "status": "completed"} tags=[]
from datetime import datetime, date

df['Date'] = pd.to_datetime(df.Date, format='%d/%m/%Y')
df.head().style.set_properties(subset=['Date'],
                               **{'background-color': 'dodgerblue'})

# %% [markdown] papermill={"duration": 0.049347, "end_time": "2021-01-17T15:24:41.818199", "exception": false, "start_time": "2021-01-17T15:24:41.768852", "status": "completed"} tags=[]
# Features:
# * **Rainfall** indicates the quantity of rain falling (mm)
# * **Temperature** indicates the temperature (°C)
# * **Volume** indicates the volume of water taken from the drinking water treatment plant (m$^3$)
# * **Hydrometry** indicates the groundwater level (m)
#
# Target:
# * **Depth to Groundwater** indicates the groundwater level (m from the ground floor)
#

# %% _kg_hide-input=true _kg_hide-output=false papermill={"duration": 2.442321, "end_time": "2021-01-17T15:24:44.309917", "exception": false, "start_time": "2021-01-17T15:24:41.867596", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=5, ncols=1, figsize=(15, 25))

sns.lineplot(x=df.Date,
             y=df.Rainfall.fillna(np.inf),
             ax=ax[0],
             color='dodgerblue')
ax[0].set_title('Feature: Rainfall', fontsize=14)
ax[0].set_ylabel(ylabel='Rainfall', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.Temperature.fillna(np.inf),
             ax=ax[1],
             color='dodgerblue',
             label='Bastia Umbra')
ax[1].set_title('Feature: Temperature', fontsize=14)
ax[1].set_ylabel(ylabel='Temperature', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.fillna(np.inf),
             ax=ax[2],
             color='dodgerblue')
ax[2].set_title('Feature: Volume', fontsize=14)
ax[2].set_ylabel(ylabel='Volume', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.River_Hydrometry.fillna(np.inf),
             ax=ax[3],
             color='dodgerblue')
ax[3].set_title('Feature: Hydrometry', fontsize=14)
ax[3].set_ylabel(ylabel='Hydrometry', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.Depth_to_Groundwater.fillna(np.inf),
             ax=ax[4],
             color='dodgerblue')
ax[4].set_title('Target: Depth to Groundwater', fontsize=14)
ax[4].set_ylabel(ylabel='Depth to Groundwater', fontsize=14)

for i in range(5):
    ax[i].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])

plt.show()

# %% [markdown] papermill={"duration": 0.053201, "end_time": "2021-01-17T15:24:44.417274", "exception": false, "start_time": "2021-01-17T15:24:44.364073", "status": "completed"} tags=[]
# # Data Preprocessing
#
# ## Chronological Order and Equidistant Timestamps
# The data should be in **chronological order** and the **timestamps should be equidistant** in time series. The chronological order can be achieved by sorting the dataframe by the timestamps. Equidisant timestamps indicates constant time intervals. To check this, the difference between each timestamp can be taken. If this is not the case, you can decide on a constant time interval and resample the data (see [Resampling](#Resampling)).
#
# This is already the case in our data: The time interval is one day and the data is already in chronological order. Therefore, we do not have to do this additional data preparation step.

# %% papermill={"duration": 0.087239, "end_time": "2021-01-17T15:24:44.567105", "exception": false, "start_time": "2021-01-17T15:24:44.479866", "status": "completed"} tags=[]
# Sort values by timestamp (not necessary in this case)
df = df.sort_values(by='Date')

# Check time intervals
df['Time_Interval'] = df.Date - df.Date.shift(1)

df[['Date', 'Time_Interval']].head()

# %% _kg_hide-input=true papermill={"duration": 0.069336, "end_time": "2021-01-17T15:24:44.701883", "exception": false, "start_time": "2021-01-17T15:24:44.632547", "status": "completed"} tags=[]
print(f"{df['Time_Interval'].value_counts()}")
df = df.drop('Time_Interval', axis=1)

# %% [markdown] papermill={"duration": 0.055138, "end_time": "2021-01-17T15:24:44.814942", "exception": false, "start_time": "2021-01-17T15:24:44.759804", "status": "completed"} tags=[]
# ## Handling Missing Values
#
# We can see that `Depth_to_Groundwater` has missing values.
#
# Furthermore, plotting the time series reveals that there seem to be some **implausible zero values** for `Drainage_Volume`, and `River_Hydrometry`. We will have to clean them by replacing them by `nan` values and filling them afterwards.

# %% _kg_hide-input=true _kg_hide-output=false papermill={"duration": 1.668274, "end_time": "2021-01-17T15:24:46.538039", "exception": false, "start_time": "2021-01-17T15:24:44.869765", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
old = df.River_Hydrometry.copy()
df['River_Hydrometry'] = np.where((df.River_Hydrometry == 0), np.nan,
                                  df.River_Hydrometry)

sns.lineplot(x=df.Date,
             y=old.fillna(np.inf),
             ax=ax[0],
             color='darkorange',
             label='original')
sns.lineplot(x=df.Date,
             y=df.River_Hydrometry.fillna(np.inf),
             ax=ax[0],
             color='dodgerblue',
             label='modified')
ax[0].set_title('Feature: Hydrometry', fontsize=14)
ax[0].set_ylabel(ylabel='Hydrometry', fontsize=14)

old = df.Drainage_Volume.copy()
df['Drainage_Volume'] = np.where((df.Drainage_Volume == 0), np.nan,
                                 df.Drainage_Volume)

sns.lineplot(x=df.Date,
             y=old.fillna(np.inf),
             ax=ax[1],
             color='darkorange',
             label='original')
sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.fillna(np.inf),
             ax=ax[1],
             color='dodgerblue',
             label='modified')
ax[1].set_title('Feature: Volume', fontsize=14)
ax[1].set_ylabel(ylabel='Volume', fontsize=14)

for i in range(2):
    ax[i].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])

plt.show()

# %% [markdown] papermill={"duration": 0.058049, "end_time": "2021-01-17T15:24:46.655445", "exception": false, "start_time": "2021-01-17T15:24:46.597396", "status": "completed"} tags=[]
# Now we have to think about what to do with these missing values.

# %% _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _kg_hide-input=true _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" papermill={"duration": 1.40932, "end_time": "2021-01-17T15:24:48.123363", "exception": false, "start_time": "2021-01-17T15:24:46.714043", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
sns.heatmap(df.T.isna(), cmap='Blues')
ax.set_title('Fields with Missing Values', fontsize=16)
#for tick in ax.xaxis.get_major_ticks():
#    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
plt.show()

# %% [markdown] papermill={"duration": 0.059385, "end_time": "2021-01-17T15:24:48.243078", "exception": false, "start_time": "2021-01-17T15:24:48.183693", "status": "completed"} tags=[]
# * **Option 1: Fill NaN with Outlier or Zero**
#
#     In this specific example filling the missing value with an outlier value such as -999 is not a good idea. However, many notebooks in this challenge have been using -999.
#
# * **Option 2: Fill NaN with Mean Value**
#
#     Also in this example, we can see that filling NaNs with the mean value is also not sufficient.
#
# * **Option 3: Fill NaN with Last Value with `.ffill()`**
#
#     Filling NaNs with the last value is already a little bit better in this case.
#
# * **Option 4: Fill NaN with Linearly Interpolated Value with `.interpolate()`**
#
#     Filling NaNs with the interpolated values is the best option in this small examlple but it requires knowledge of the neighouring values.
#

# %% _kg_hide-input=true papermill={"duration": 3.1715, "end_time": "2021-01-17T15:24:51.474321", "exception": false, "start_time": "2021-01-17T15:24:48.302821", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 12))

sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.fillna(0),
             ax=ax[0],
             color='darkorange',
             label='modified')
sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.fillna(np.inf),
             ax=ax[0],
             color='dodgerblue',
             label='original')
ax[0].set_title('Fill NaN with 0', fontsize=14)
ax[0].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

mean_val = df.Drainage_Volume.mean()
sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.fillna(mean_val),
             ax=ax[1],
             color='darkorange',
             label='modified')
sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.fillna(np.inf),
             ax=ax[1],
             color='dodgerblue',
             label='original')
ax[1].set_title(f'Fill NaN with Mean Value ({mean_val:.0f})', fontsize=14)
ax[1].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.ffill(),
             ax=ax[2],
             color='darkorange',
             label='modified')
sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.fillna(np.inf),
             ax=ax[2],
             color='dodgerblue',
             label='original')
ax[2].set_title(f'FFill', fontsize=14)
ax[2].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.interpolate(),
             ax=ax[3],
             color='darkorange',
             label='modified')
sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.fillna(np.inf),
             ax=ax[3],
             color='dodgerblue',
             label='original')
ax[3].set_title(f'Interpolate', fontsize=14)
ax[3].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

for i in range(4):
    ax[i].set_xlim([date(2019, 5, 1), date(2019, 10, 1)])
plt.tight_layout()
plt.show()

# %% papermill={"duration": 0.076282, "end_time": "2021-01-17T15:24:51.615549", "exception": false, "start_time": "2021-01-17T15:24:51.539267", "status": "completed"} tags=[]
df['Drainage_Volume'] = df['Drainage_Volume'].interpolate()
df['River_Hydrometry'] = df['River_Hydrometry'].interpolate()
df['Depth_to_Groundwater'] = df['Depth_to_Groundwater'].interpolate()

# %% [markdown] papermill={"duration": 0.063411, "end_time": "2021-01-17T15:24:51.744017", "exception": false, "start_time": "2021-01-17T15:24:51.680606", "status": "completed"} tags=[]
# ## Resampling
#
# Resampling can provide additional information on the data. There are two types of resampling:
# * **Upsampling** is when the frequency of samples is increased (e.g. days to hours)
# * **Downsampling** is when the frequency of samples is decreased (e.g. days to weeks)
#
# In this example, we will do some downsampling with the `.resample()` function.

# %% _kg_hide-input=true papermill={"duration": 11.3081, "end_time": "2021-01-17T15:25:03.116975", "exception": false, "start_time": "2021-01-17T15:24:51.808875", "status": "completed"} tags=[]
fig, ax = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(16, 12))

ax[0, 0].bar(df.Date, df.Rainfall, width=5, color='dodgerblue')
ax[0, 0].set_title('Daily Rainfall (Acc.)', fontsize=14)

resampled_df = df[['Date', 'Rainfall'
                   ]].resample('7D', on='Date').sum().reset_index(drop=False)
ax[1, 0].bar(resampled_df.Date,
             resampled_df.Rainfall,
             width=10,
             color='dodgerblue')
ax[1, 0].set_title('Weekly Rainfall (Acc.)', fontsize=14)

resampled_df = df[['Date', 'Rainfall'
                   ]].resample('M', on='Date').sum().reset_index(drop=False)
ax[2, 0].bar(resampled_df.Date,
             resampled_df.Rainfall,
             width=15,
             color='dodgerblue')
ax[2, 0].set_title('Monthly Rainfall (Acc.)', fontsize=14)

resampled_df = df[['Date', 'Rainfall'
                   ]].resample('12M', on='Date').sum().reset_index(drop=False)
ax[3, 0].bar(resampled_df.Date,
             resampled_df.Rainfall,
             width=20,
             color='dodgerblue')
ax[3, 0].set_title('Annual Rainfall (Acc.)', fontsize=14)

for i in range(4):
    ax[i, 0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])

sns.lineplot(df.Date, df.Temperature, color='dodgerblue', ax=ax[0, 1])
ax[0, 1].set_title('Daily Temperature (Acc.)', fontsize=14)

resampled_df = df[['Date', 'Temperature'
                   ]].resample('7D', on='Date').mean().reset_index(drop=False)
sns.lineplot(resampled_df.Date,
             resampled_df.Temperature,
             color='dodgerblue',
             ax=ax[1, 1])
ax[1, 1].set_title('Weekly Temperature (Acc.)', fontsize=14)

resampled_df = df[['Date', 'Temperature'
                   ]].resample('M', on='Date').mean().reset_index(drop=False)
sns.lineplot(resampled_df.Date,
             resampled_df.Temperature,
             color='dodgerblue',
             ax=ax[2, 1])
ax[2, 1].set_title('Monthly Temperature (Acc.)', fontsize=14)

resampled_df = df[['Date', 'Temperature'
                   ]].resample('365D',
                               on='Date').mean().reset_index(drop=False)
sns.lineplot(resampled_df.Date,
             resampled_df.Temperature,
             color='dodgerblue',
             ax=ax[3, 1])
ax[3, 1].set_title('Annual Temperature (Acc.)', fontsize=14)

for i in range(4):
    ax[i, 1].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
    ax[i, 1].set_ylim([-5, 35])
plt.show()

# %% [markdown] papermill={"duration": 0.066262, "end_time": "2021-01-17T15:25:03.251035", "exception": false, "start_time": "2021-01-17T15:25:03.184773", "status": "completed"} tags=[]
# In this example, resampling would not be necessary. On the other hand, there is no necessity to look at the daily data. Considering weekly data seems to be sufficient as well. Therefore, we will **downsample the data to a weekly basis**.

# %% papermill={"duration": 0.090569, "end_time": "2021-01-17T15:25:03.408281", "exception": false, "start_time": "2021-01-17T15:25:03.317712", "status": "completed"} tags=[]
df_downsampled = df[[
    'Date', 'Depth_to_Groundwater', 'Temperature', 'Drainage_Volume',
    'River_Hydrometry'
]].resample('7D', on='Date').mean().reset_index(drop=False)

df_downsampled['Rainfall'] = df[['Date', 'Rainfall']].resample(
    '7D', on='Date').sum().reset_index(drop=False)[['Rainfall']]

df = df_downsampled

# %% [markdown] papermill={"duration": 0.06652, "end_time": "2021-01-17T15:25:03.542412", "exception": false, "start_time": "2021-01-17T15:25:03.475892", "status": "completed"} tags=[]
# ## Stationarity
#
# Some time-series models, such as such as [ARIMA](#ARIMA), assume that the underlying data is stationary.
# Stationarity describes that the time-series has
# * constant mean and mean is not time-dependent
# * constant variance and variance is not time-dependent
# * constant covariance and covariance is not time-dependent
#
# > If a time series has a specific (stationary) behavior over a given time interval, then it can be assumed that the time series will behave the same at a later time.
#
# Time series **with trend and/or seasonality are not stationary**. Trend indicates that the mean is not constant over time and seasonality indicates that the variance is not constant over time.

# %% _kg_hide-input=true papermill={"duration": 0.865994, "end_time": "2021-01-17T15:25:04.474960", "exception": false, "start_time": "2021-01-17T15:25:03.608966", "status": "completed"} tags=[]
t = np.linspace(0, 19, 20)

fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 4))
stationary = [
    5,
    4,
    5,
    6,
    5,
    4,
    5,
    6,
    5,
    4,
    5,
    6,
    5,
    4,
    5,
    6,
    5,
    4,
    5,
    6,
]
sns.lineplot(x=t, y=stationary, ax=ax[0], color='forestgreen')
sns.lineplot(x=t, y=5, ax=ax[0], color='grey')
sns.lineplot(x=t, y=6, ax=ax[0], color='grey')
sns.lineplot(x=t, y=4, ax=ax[0], color='grey')
ax[0].lines[2].set_linestyle("--")
ax[0].lines[3].set_linestyle("--")
ax[0].set_title(
    f'Stationary \nconstant mean \nconstant variance \nconstant covariance',
    fontsize=14)

nonstationary1 = [9, 0, 1, 10, 8, 1, 2, 9, 7, 2, 3, 8, 6, 3, 4, 7, 5, 4, 5, 6]
sns.lineplot(x=t, y=nonstationary1, ax=ax[1], color='indianred')
sns.lineplot(x=t, y=5, ax=ax[1], color='grey')
sns.lineplot(x=t, y=t * 0.25 - 0.5, ax=ax[1], color='grey')
sns.lineplot(x=t, y=t * (-0.25) + 11, ax=ax[1], color='grey')
ax[1].lines[2].set_linestyle("--")
ax[1].lines[3].set_linestyle("--")
ax[1].set_title(
    f'Non Stationary \nconstant mean \n non-constant variance\nnconstant covariance',
    fontsize=14)

nonstationary2 = [
    0,
    2,
    1,
    3,
    2,
    4,
    3,
    5,
    4,
    6,
    5,
    7,
    6,
    8,
    7,
    9,
    8,
    10,
    9,
    11,
]
sns.lineplot(x=t, y=nonstationary2, ax=ax[2], color='indianred')
sns.lineplot(x=t, y=t * 0.5 + 0.7, ax=ax[2], color='grey')
sns.lineplot(x=t, y=t * 0.5, ax=ax[2], color='grey')
sns.lineplot(x=t, y=t * 0.5 + 1.5, ax=ax[2], color='grey')
ax[2].lines[2].set_linestyle("--")
ax[2].lines[3].set_linestyle("--")
ax[2].set_title(
    f'Non Stationary \n non-constant mean\nconstant variance\nnconstant covariance',
    fontsize=14)

nonstationary3 = [
    5,
    4.5,
    4,
    4.5,
    5,
    5.5,
    6,
    5.5,
    5,
    4.5,
    4,
    5,
    6,
    5,
    4,
    6,
    4,
    6,
    4,
    6,
]
sns.lineplot(x=t, y=nonstationary3, ax=ax[3], color='indianred')
sns.lineplot(x=t, y=5, ax=ax[3], color='grey')
sns.lineplot(x=t, y=6, ax=ax[3], color='grey')
sns.lineplot(x=t, y=4, ax=ax[3], color='grey')
ax[3].lines[2].set_linestyle("--")
ax[3].lines[3].set_linestyle("--")
ax[3].set_title(
    f'Stationary \nconstant mean \nconstant variance \nnon-constant covariance',
    fontsize=14)

for i in range(4):
    ax[i].set_ylim([-1, 12])
    ax[i].set_xlabel('Time', fontsize=14)

# %% [markdown] papermill={"duration": 0.069159, "end_time": "2021-01-17T15:25:04.613281", "exception": false, "start_time": "2021-01-17T15:25:04.544122", "status": "completed"} tags=[]
# The check for stationarity can be done via three different approaches:
# 1. **visually**: plot time series and check for trends or seasonality
# 2. **basic statistics**: split time series and compare the mean and variance of each partition
# 3. **statistical test**: Augmented Dickey Fuller test
#
# Let's do the **visual check** first. We can see that all features except `Temperature` have non-constant mean and non-constant variance. Therefore, **none of these seem to be stationary**. However, `Temperature` shows strong seasonality (hot in summer, cold in winter) and therefore it is not stationary either.

# %% _kg_hide-input=true papermill={"duration": 2.012837, "end_time": "2021-01-17T15:25:06.695661", "exception": false, "start_time": "2021-01-17T15:25:04.682824", "status": "completed"} tags=[]
rolling_window = 52
f, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

sns.lineplot(x=df.Date, y=df.Rainfall, ax=ax[0, 0], color='indianred')
sns.lineplot(x=df.Date,
             y=df.Rainfall.rolling(rolling_window).mean(),
             ax=ax[0, 0],
             color='black',
             label='rolling mean')
sns.lineplot(x=df.Date,
             y=df.Rainfall.rolling(rolling_window).std(),
             ax=ax[0, 0],
             color='blue',
             label='rolling std')
ax[0, 0].set_title(
    'Rainfall: Non-stationary \nnon-constant mean & non-constant variance',
    fontsize=14)
ax[0, 0].set_ylabel(ylabel='Rainfall', fontsize=14)

sns.lineplot(x=df.Date, y=df.Temperature, ax=ax[1, 0], color='indianred')
sns.lineplot(x=df.Date,
             y=df.Temperature.rolling(rolling_window).mean(),
             ax=ax[1, 0],
             color='black',
             label='rolling mean')
sns.lineplot(x=df.Date,
             y=df.Temperature.rolling(rolling_window).std(),
             ax=ax[1, 0],
             color='blue',
             label='rolling std')
ax[1, 0].set_title(
    'Temperature: Non-stationary \nvariance is time-dependent (seasonality)',
    fontsize=14)
ax[1, 0].set_ylabel(ylabel='Temperature', fontsize=14)

sns.lineplot(x=df.Date, y=df.River_Hydrometry, ax=ax[0, 1], color='indianred')
sns.lineplot(x=df.Date,
             y=df.River_Hydrometry.rolling(rolling_window).mean(),
             ax=ax[0, 1],
             color='black',
             label='rolling mean')
sns.lineplot(x=df.Date,
             y=df.River_Hydrometry.rolling(rolling_window).std(),
             ax=ax[0, 1],
             color='blue',
             label='rolling std')
ax[0, 1].set_title(
    'Hydrometry: Non-stationary \nnon-constant mean & non-constant variance',
    fontsize=14)
ax[0, 1].set_ylabel(ylabel='Hydrometry', fontsize=14)

sns.lineplot(x=df.Date, y=df.Drainage_Volume, ax=ax[1, 1], color='indianred')
sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.rolling(rolling_window).mean(),
             ax=ax[1, 1],
             color='black',
             label='rolling mean')
sns.lineplot(x=df.Date,
             y=df.Drainage_Volume.rolling(rolling_window).std(),
             ax=ax[1, 1],
             color='blue',
             label='rolling std')
ax[1, 1].set_title(
    'Volume: Non-stationary \nnon-constant mean & non-constant variance',
    fontsize=14)
ax[1, 1].set_ylabel(ylabel='Volume', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.Depth_to_Groundwater,
             ax=ax[2, 0],
             color='indianred')
sns.lineplot(x=df.Date,
             y=df.Depth_to_Groundwater.rolling(rolling_window).mean(),
             ax=ax[2, 0],
             color='black',
             label='rolling mean')
sns.lineplot(x=df.Date,
             y=df.Depth_to_Groundwater.rolling(rolling_window).std(),
             ax=ax[2, 0],
             color='blue',
             label='rolling std')
ax[2, 0].set_title(
    'Depth to Groundwater: Non-stationary \nnon-constant mean & non-constant variance',
    fontsize=14)
ax[2, 0].set_ylabel(ylabel='Depth to Groundwater', fontsize=14)

for i in range(3):
    ax[i, 0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
    ax[i, 1].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])

f.delaxes(ax[2, 1])
plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.076945, "end_time": "2021-01-17T15:25:06.850471", "exception": false, "start_time": "2021-01-17T15:25:06.773526", "status": "completed"} tags=[]
# Next, we will **check the underlying statistics**. For this we will **split the time series into two sections** and check the mean and the variance. You could do more partitions if you wanted.
#
# With this method, `Temperature` and `River_Hydrometry` show **somewhat similar (constant) mean and variance** and could be seen as stationary. However, with this method, we are not able to see the seasonality in the `Temperature` feature.

# %% papermill={"duration": 0.095371, "end_time": "2021-01-17T15:25:07.023184", "exception": false, "start_time": "2021-01-17T15:25:06.927813", "status": "completed"} tags=[]
num_partitions = 2
partition_length = int(len(df) / num_partitions)

partition1_mean = df.head(partition_length).mean()
partition1_var = df.head(partition_length).var()
partition2_mean = df.tail(partition_length).mean()
partition2_var = df.tail(partition_length).var()

# %% _kg_hide-input=true papermill={"duration": 0.119633, "end_time": "2021-01-17T15:25:07.217921", "exception": false, "start_time": "2021-01-17T15:25:07.098288", "status": "completed"} tags=[]
stationarity_test = pd.concat(
    [partition1_mean, partition2_mean, partition1_var, partition2_var], axis=1)
stationarity_test.columns = [
    'Partition 1 Mean', 'Partition 2 Mean', 'Partition 1 Variance',
    'Partition 2 Variance'
]


def highlight_greater(x):
    temp = x.copy()
    temp = temp.round(0).astype(int)
    m1 = (temp['Partition 1 Mean'] == temp['Partition 2 Mean'])
    m2 = (temp['Partition 1 Variance'] == temp['Partition 2 Variance'])
    m3 = (temp['Partition 1 Mean'] < temp['Partition 2 Mean'] + 3) & (
        temp['Partition 1 Mean'] > temp['Partition 2 Mean'] - 3)
    m4 = (temp['Partition 1 Variance'] < temp['Partition 2 Variance'] + 3) & (
        temp['Partition 1 Variance'] > temp['Partition 2 Variance'] - 3)

    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    #rewrite values by boolean masks
    df1['Partition 1 Mean'] = np.where(~m1,
                                       'background-color: {}'.format('salmon'),
                                       df1['Partition 1 Mean'])
    df1['Partition 2 Mean'] = np.where(~m1,
                                       'background-color: {}'.format('salmon'),
                                       df1['Partition 2 Mean'])
    df1['Partition 1 Mean'] = np.where(m3,
                                       'background-color: {}'.format('gold'),
                                       df1['Partition 1 Mean'])
    df1['Partition 2 Mean'] = np.where(m3,
                                       'background-color: {}'.format('gold'),
                                       df1['Partition 2 Mean'])
    df1['Partition 1 Mean'] = np.where(
        m1, 'background-color: {}'.format('mediumseagreen'),
        df1['Partition 1 Mean'])
    df1['Partition 2 Mean'] = np.where(
        m1, 'background-color: {}'.format('mediumseagreen'),
        df1['Partition 2 Mean'])

    df1['Partition 1 Variance'] = np.where(
        ~m2, 'background-color: {}'.format('salmon'),
        df1['Partition 1 Variance'])
    df1['Partition 2 Variance'] = np.where(
        ~m2, 'background-color: {}'.format('salmon'),
        df1['Partition 2 Variance'])
    df1['Partition 1 Variance'] = np.where(
        m4, 'background-color: {}'.format('gold'), df1['Partition 1 Variance'])
    df1['Partition 2 Variance'] = np.where(
        m4, 'background-color: {}'.format('gold'), df1['Partition 2 Variance'])
    df1['Partition 1 Variance'] = np.where(
        m2, 'background-color: {}'.format('mediumseagreen'),
        df1['Partition 1 Variance'])
    df1['Partition 2 Variance'] = np.where(
        m2, 'background-color: {}'.format('mediumseagreen'),
        df1['Partition 2 Variance'])

    return df1


stationarity_test.style.apply(highlight_greater, axis=None).format("{:20,.0f}")

# %% [markdown] papermill={"duration": 0.07589, "end_time": "2021-01-17T15:25:07.369891", "exception": false, "start_time": "2021-01-17T15:25:07.294001", "status": "completed"} tags=[]
# Let's evaluate the histograms. Since we are looking at the mean and variance, we are expecting that the data conforms to a Gaussian distribution (bell shaped distribution) in case of stationarity.

# %% _kg_hide-input=true papermill={"duration": 1.738263, "end_time": "2021-01-17T15:25:09.184244", "exception": false, "start_time": "2021-01-17T15:25:07.445981", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))

sns.distplot(df.Rainfall.fillna(np.inf), ax=ax[0, 0], color='indianred')
ax[0, 0].set_title(
    'Rainfall: Non-stationary \nnon-constant mean & non-constant variance',
    fontsize=14)
ax[0, 0].set_ylabel(ylabel='Rainfall', fontsize=14)

sns.distplot(df.Temperature.fillna(np.inf), ax=ax[1, 0], color='indianred')
ax[1, 0].set_title(
    'Temperature: Non-stationary \nvariance is time-dependent (seasonality)',
    fontsize=14)
ax[1, 0].set_ylabel(ylabel='Temperature', fontsize=14)

sns.distplot(df.River_Hydrometry.fillna(np.inf),
             ax=ax[0, 1],
             color='indianred')
ax[0, 1].set_title(
    'Hydrometry: Non-stationary \nnon-constant mean & non-constant variance',
    fontsize=14)
ax[0, 1].set_ylabel(ylabel='Hydrometry', fontsize=14)

sns.distplot(df.Drainage_Volume.fillna(np.inf), ax=ax[1, 1], color='indianred')
ax[1, 1].set_title(
    'Volume: Non-stationary \nnon-constant mean & non-constant variance',
    fontsize=14)
ax[1, 1].set_ylabel(ylabel='Volume', fontsize=14)

sns.distplot(df.Depth_to_Groundwater.fillna(np.inf),
             ax=ax[2, 0],
             color='indianred')
ax[2, 0].set_title(
    'Depth to Groundwater: Non-stationary \nnon-constant mean & non-constant variance',
    fontsize=14)
ax[2, 0].set_ylabel(ylabel='Depth to Groundwater', fontsize=14)

f.delaxes(ax[2, 1])
plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.079289, "end_time": "2021-01-17T15:25:09.342433", "exception": false, "start_time": "2021-01-17T15:25:09.263144", "status": "completed"} tags=[]
# **Augmented Dickey-Fuller (ADF) test**  is a type of statistical test called a unit root test.  Unit roots are a cause for non-stationarity.
#
# * **Null Hypothesis (H0)**: Time series has a unit root. (Time series is **not stationary**).
#
# * **Alternate Hypothesis (H1)**: Time series has no unit root (Time series is **stationary**).
#
# If the **null hypothesis can be rejected**, we can conclude that the **time series is stationary**.
#
# There are two ways to rejects the null hypothesis:
#
# On the one hand, the null hypothesis can be rejected if the p-value is below a set significance level. The defaults significance level is 5%
#
# * <font color='red'>**p-value > significance level (default: 0.05)**</font>: Fail to reject the null hypothesis (H0), the data has a unit root and is <font color='red'>non-stationary</font>.
# * <font color='green'>**p-value <= significance level (default: 0.05)**</font>: Reject the null hypothesis (H0), the data does not have a unit root and is <font color='green'>stationary</font>.
#
# On the other hand, the null hypothesis can be rejects if the test statistic is less than the critical value.
# * <font color='red'>**ADF statistic > critical value**</font>: Fail to reject the null hypothesis (H0), the data has a unit root and is <font color='red'>non-stationary</font>.
# * <font color='green'>**ADF statistic < critical value**</font>: Reject the null hypothesis (H0), the data does not have a unit root and is <font color='green'>stationary</font>.

# %% papermill={"duration": 0.365045, "end_time": "2021-01-17T15:25:09.787168", "exception": false, "start_time": "2021-01-17T15:25:09.422123", "status": "completed"} tags=[]
from statsmodels.tsa.stattools import adfuller

result = adfuller(df.Depth_to_Groundwater.values)
adf_stat = result[0]
p_val = result[1]
crit_val_1 = result[4]['1%']
crit_val_5 = result[4]['5%']
crit_val_10 = result[4]['10%']

# %% _kg_hide-input=true papermill={"duration": 1.223719, "end_time": "2021-01-17T15:25:11.091679", "exception": false, "start_time": "2021-01-17T15:25:09.867960", "status": "completed"} tags=[]

f, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))


def visualize_adfuller_results(series, title, ax, df):
    result = adfuller(series)
    significance_level = 0.05
    adf_stat = result[0]
    p_val = result[1]
    crit_val_1 = result[4]['1%']
    crit_val_5 = result[4]['5%']
    crit_val_10 = result[4]['10%']

    if (p_val < significance_level) & ((adf_stat < crit_val_1)):
        linecolor = 'forestgreen'
    elif (p_val < significance_level) & (adf_stat < crit_val_5):
        linecolor = 'gold'
    elif (p_val < significance_level) & (adf_stat < crit_val_10):
        linecolor = 'orange'
    else:
        linecolor = 'indianred'
    sns.lineplot(x=df.Date, y=series, ax=ax, color=linecolor)
    ax.set_title(
        f'ADF Statistic {adf_stat:0.3f}, p-value: {p_val:0.3f}\nCritical Values 1%: {crit_val_1:0.3f}, 5%: {crit_val_5:0.3f}, 10%: {crit_val_10:0.3f}',
        fontsize=14)
    ax.set_ylabel(ylabel=title, fontsize=14)


visualize_adfuller_results(df.Rainfall.values, 'Rainfall', ax[0, 0], df)
visualize_adfuller_results(df.Temperature.values, 'Temperature', ax[1, 0], df)
visualize_adfuller_results(df.River_Hydrometry.values, 'River_Hydrometry',
                           ax[0, 1], df)
visualize_adfuller_results(df.Drainage_Volume.values, 'Drainage_Volume',
                           ax[1, 1], df)
visualize_adfuller_results(df.Depth_to_Groundwater.values,
                           'Depth_to_Groundwater', ax[2, 0], df)

f.delaxes(ax[2, 1])
plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.085084, "end_time": "2021-01-17T15:25:11.261673", "exception": false, "start_time": "2021-01-17T15:25:11.176589", "status": "completed"} tags=[]
# **TODO: How to interpret the contradictory results from the different checks...**

# %% [markdown] papermill={"duration": 0.084347, "end_time": "2021-01-17T15:25:11.430525", "exception": false, "start_time": "2021-01-17T15:25:11.346178", "status": "completed"} tags=[]
# If the data is not stationary but we want to use a model that requires with characteristic, the data has to be transformed. However, if the data is not stationary to begin with, we should rethink the choice of model.
#
# The two most common methods to achieve stationarity are:
# * **Transformation**: e.g. log or square root to stabilize non-constant variance
# * **Differencing**: subtracts the current value from the previous
#

# %% papermill={"duration": 0.093741, "end_time": "2021-01-17T15:25:11.608525", "exception": false, "start_time": "2021-01-17T15:25:11.514784", "status": "completed"} tags=[]
# Log Transform of absolute values
# (Log transoform of negative values will return NaN)
df['Depth_to_Groundwater_log'] = np.log(abs(df.Depth_to_Groundwater))

# %% _kg_hide-input=true papermill={"duration": 0.910914, "end_time": "2021-01-17T15:25:12.602248", "exception": false, "start_time": "2021-01-17T15:25:11.691334", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
visualize_adfuller_results(abs(df.Depth_to_Groundwater),
                           'Absolute \n Depth to Groundwater', ax[0, 0], df)

sns.distplot(df.Depth_to_Groundwater_log, ax=ax[0, 1])
visualize_adfuller_results(df.Depth_to_Groundwater_log,
                           'Transformed \n Depth to Groundwater', ax[1, 0], df)

sns.distplot(df.Depth_to_Groundwater_log, ax=ax[1, 1])

plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.085014, "end_time": "2021-01-17T15:25:12.772829", "exception": false, "start_time": "2021-01-17T15:25:12.687815", "status": "completed"} tags=[]
# Differencing can be done in different orders:
# * First order differencing: linear trends with $z_i = y_i - y_{i-1}$
# * Second-order differencing: quadratic trends with $z_i = (y_i - y_{i-1}) - (y_{i-1} - y_{i-2})$
# * and so on...

# %% papermill={"duration": 0.097265, "end_time": "2021-01-17T15:25:12.955144", "exception": false, "start_time": "2021-01-17T15:25:12.857879", "status": "completed"} tags=[]
# First Order Differencing
ts_diff = np.diff(df.Depth_to_Groundwater)
df['Depth_to_Groundwater_diff_1'] = np.append([0], ts_diff)

# Second Order Differencing
ts_diff = np.diff(df.Depth_to_Groundwater_diff_1)
df['Depth_to_Groundwater_diff_2'] = np.append([0], ts_diff)

# %% _kg_hide-input=true papermill={"duration": 0.565564, "end_time": "2021-01-17T15:25:13.607429", "exception": false, "start_time": "2021-01-17T15:25:13.041865", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

visualize_adfuller_results(df.Depth_to_Groundwater_diff_1,
                           'Differenced (1. Order) \n Depth to Groundwater',
                           ax[0], df)
visualize_adfuller_results(df.Depth_to_Groundwater_diff_2,
                           'Differenced (2. Order) \n Depth to Groundwater',
                           ax[1], df)
plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.088768, "end_time": "2021-01-17T15:25:13.786342", "exception": false, "start_time": "2021-01-17T15:25:13.697574", "status": "completed"} tags=[]
# The differencing can be reverted if the the first value before differencing is known. In this case, we can accumulate all values with the function `.cumsum()` and add the first value of the original time series.

# %% papermill={"duration": 0.099347, "end_time": "2021-01-17T15:25:13.974278", "exception": false, "start_time": "2021-01-17T15:25:13.874931", "status": "completed"} tags=[]
df.Depth_to_Groundwater.equals(df.Depth_to_Groundwater_diff_1.cumsum() +
                               df.Depth_to_Groundwater.iloc[0])

# %% [markdown] papermill={"duration": 0.089467, "end_time": "2021-01-17T15:25:14.155862", "exception": false, "start_time": "2021-01-17T15:25:14.066395", "status": "completed"} tags=[]
# # Feature Engineering
#
# ## Time Features

# %% papermill={"duration": 0.123481, "end_time": "2021-01-17T15:25:14.370092", "exception": false, "start_time": "2021-01-17T15:25:14.246611", "status": "completed"} tags=[]
df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['day'] = pd.DatetimeIndex(df['Date']).day
df['day_of_year'] = pd.DatetimeIndex(df['Date']).dayofyear
df['week_of_year'] = pd.DatetimeIndex(df['Date']).weekofyear
df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
df['season'] = df.month % 12 // 3 + 1

df[[
    'Date', 'year', 'month', 'day', 'day_of_year', 'week_of_year', 'quarter',
    'season'
]].head()

# %% [markdown] papermill={"duration": 0.091852, "end_time": "2021-01-17T15:25:14.557349", "exception": false, "start_time": "2021-01-17T15:25:14.465497", "status": "completed"} tags=[]
# ## Encoding Cyclical Features
# The new time features are cyclical. For example,the feature `month` cycles between 1 and 12 for every year.
# While the difference between each month increments by 1 during the year, between two years the `month` feature jumps from 12 (December) to 1 (January). This results in a -11 difference, which can confuse a lot of models.

# %% _kg_hide-input=true papermill={"duration": 0.301777, "end_time": "2021-01-17T15:25:14.949940", "exception": false, "start_time": "2021-01-17T15:25:14.648163", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))

sns.lineplot(x=df.Date, y=df.month, color='dodgerblue')
ax.set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
plt.show()

# %% [markdown] papermill={"duration": 0.092283, "end_time": "2021-01-17T15:25:15.134903", "exception": false, "start_time": "2021-01-17T15:25:15.042620", "status": "completed"} tags=[]
# Ideally, we want the underlying data to represent the same difference between two consecutive months, even between December and January. A common remedy for this issue is to encode cyclical features into two dimensions with sine and cosine transformation.

# %% _kg_hide-input=true _kg_hide-output=true papermill={"duration": 0.104048, "end_time": "2021-01-17T15:25:15.331591", "exception": false, "start_time": "2021-01-17T15:25:15.227543", "status": "completed"} tags=[]
month_in_year = 12
df['month_sin'] = np.sin(2 * np.pi * df.month / month_in_year)
df['month_cos'] = np.cos(2 * np.pi * df.month / month_in_year)

# %% _kg_hide-input=true papermill={"duration": 0.295424, "end_time": "2021-01-17T15:25:15.718934", "exception": false, "start_time": "2021-01-17T15:25:15.423510", "status": "completed"} tags=[]
days_in_month = 30
df['day_sin'] = np.sin(2 * np.pi * df.day / days_in_month)
df['day_cos'] = np.cos(2 * np.pi * df.day / days_in_month)

days_in_year = 365
df['day_of_year_sin'] = np.sin(2 * np.pi * df.day_of_year / days_in_year)
df['day_of_year_cos'] = np.cos(2 * np.pi * df.day_of_year / days_in_year)

weeks_in_year = 52.1429
df['week_of_year_sin'] = np.sin(2 * np.pi * df.week_of_year / weeks_in_year)
df['week_of_year_cos'] = np.cos(2 * np.pi * df.week_of_year / weeks_in_year)

quarters_in_year = 4
df['quarter_sin'] = np.sin(2 * np.pi * df.quarter / quarters_in_year)
df['quarter_cos'] = np.cos(2 * np.pi * df.quarter / quarters_in_year)

seasons_in_year = 4
df['season_sin'] = np.sin(2 * np.pi * df.season / seasons_in_year)
df['season_cos'] = np.cos(2 * np.pi * df.season / seasons_in_year)

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

sns.scatterplot(x=df.month_sin, y=df.month_cos, color='dodgerblue')
plt.show()

# %% [markdown] papermill={"duration": 0.09423, "end_time": "2021-01-17T15:25:15.906123", "exception": false, "start_time": "2021-01-17T15:25:15.811893", "status": "completed"} tags=[]
# ## Decomposition
#
# The **characteristics of a time series** are
# * Trend and Level
# * Seasonality
# * Random / Noise
#
# We can use the function `seasonal_decompose()` from the [statsmodels](https://www.statsmodels.org) library.
#
# * Additive: $y(t) = Level + Trend + Seasonality + Noise$
# * Multiplicative: $y(t) = Level * Trend * Seasonality * Noise$

# %% papermill={"duration": 0.152167, "end_time": "2021-01-17T15:25:16.152057", "exception": false, "start_time": "2021-01-17T15:25:15.999890", "status": "completed"} tags=[]
from statsmodels.tsa.seasonal import seasonal_decompose

decompose_cols = [
    'Rainfall', 'Temperature', 'Drainage_Volume', 'River_Hydrometry',
    'Depth_to_Groundwater'
]

for col in decompose_cols:
    decomp = seasonal_decompose(df[col],
                                freq=52,
                                model='additive',
                                extrapolate_trend='freq')
    df[f"{col}_trend"] = decomp.trend
    df[f"{col}_seasonal"] = decomp.seasonal

# %% _kg_hide-input=true papermill={"duration": 2.38781, "end_time": "2021-01-17T15:25:18.634264", "exception": false, "start_time": "2021-01-17T15:25:16.246454", "status": "completed"} tags=[]
fig, ax = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(16, 8))
res = seasonal_decompose(df.Temperature,
                         freq=52,
                         model='additive',
                         extrapolate_trend='freq')

ax[0, 0].set_title('Decomposition of Temperature', fontsize=16)
res.observed.plot(ax=ax[0, 0], legend=False, color='dodgerblue')
ax[0, 0].set_ylabel('Observed', fontsize=14)
ax[0, 0].set_ylim([-5, 35])

res.trend.plot(ax=ax[1, 0], legend=False, color='dodgerblue')
ax[1, 0].set_ylabel('Trend', fontsize=14)
ax[1, 0].set_ylim([-5, 35])

res.seasonal.plot(ax=ax[2, 0], legend=False, color='dodgerblue')
ax[2, 0].set_ylabel('Seasonal', fontsize=14)
ax[2, 0].set_ylim([-15, 15])

res.resid.plot(ax=ax[3, 0], legend=False, color='dodgerblue')
ax[3, 0].set_ylabel('Residual', fontsize=14)
ax[3, 0].set_ylim([-15, 15])

ax[0, 1].set_title('Decomposition of Depth_to_Groundwater', fontsize=16)
res = seasonal_decompose(df.Depth_to_Groundwater,
                         freq=52,
                         model='additive',
                         extrapolate_trend='freq')

res.observed.plot(ax=ax[0, 1], legend=False, color='dodgerblue')
ax[0, 1].set_ylabel('Observed', fontsize=14)

res.trend.plot(ax=ax[1, 1], legend=False, color='dodgerblue')
ax[1, 1].set_ylabel('Trend', fontsize=14)

res.seasonal.plot(ax=ax[2, 1], legend=False, color='dodgerblue')
ax[2, 1].set_ylabel('Seasonal', fontsize=14)

res.resid.plot(ax=ax[3, 1], legend=False, color='dodgerblue')
ax[3, 1].set_ylabel('Residual', fontsize=14)

plt.show()

# %% _kg_hide-input=true papermill={"duration": 0.131376, "end_time": "2021-01-17T15:25:18.862798", "exception": false, "start_time": "2021-01-17T15:25:18.731422", "status": "completed"} tags=[]
df[['Rainfall', 'Rainfall_trend', 'Rainfall_seasonal',
          'Temperature', 'Temperature_trend', 'Temperature_seasonal',
          'Drainage_Volume', 'Drainage_Volume_trend', 'Drainage_Volume_seasonal',
          'River_Hydrometry', 'River_Hydrometry_trend', 'River_Hydrometry_seasonal',
          'Depth_to_Groundwater', 'Depth_to_Groundwater_trend', 'Depth_to_Groundwater_seasonal']].head()\
.style.set_properties(subset=['Rainfall_trend', 'Rainfall_seasonal',
                              'Temperature_trend', 'Temperature_seasonal',
                              'Drainage_Volume_trend', 'Drainage_Volume_seasonal',
                              'River_Hydrometry_trend', 'River_Hydrometry_seasonal',
                              'Depth_to_Groundwater_trend', 'Depth_to_Groundwater_seasonal'
                             ], **{'background-color': 'dodgerblue'})

# %% [markdown] papermill={"duration": 0.096207, "end_time": "2021-01-17T15:25:19.055201", "exception": false, "start_time": "2021-01-17T15:25:18.958994", "status": "completed"} tags=[]
# ## Lag
# `.shift()`
#
#

# %% papermill={"duration": 0.112268, "end_time": "2021-01-17T15:25:19.263927", "exception": false, "start_time": "2021-01-17T15:25:19.151659", "status": "completed"} tags=[]
weeks_in_month = 4

df['Temperature_seasonal_shift_r_2M'] = df.Temperature_seasonal.shift(
    -2 * weeks_in_month)
df['Temperature_seasonal_shift_r_1M'] = df.Temperature_seasonal.shift(
    -1 * weeks_in_month)
df['Temperature_seasonal_shift_1M'] = df.Temperature_seasonal.shift(
    1 * weeks_in_month)
df['Temperature_seasonal_shift_2M'] = df.Temperature_seasonal.shift(
    2 * weeks_in_month)
df['Temperature_seasonal_shift_3M'] = df.Temperature_seasonal.shift(
    3 * weeks_in_month)

# %% _kg_hide-input=true papermill={"duration": 0.745036, "end_time": "2021-01-17T15:25:20.106199", "exception": false, "start_time": "2021-01-17T15:25:19.361163", "status": "completed"} tags=[]
df['Drainage_Volume_seasonal_shift_r_2M'] = df.Drainage_Volume_seasonal.shift(
    -2 * weeks_in_month)
df['Drainage_Volume_seasonal_shift_r_1M'] = df.Drainage_Volume_seasonal.shift(
    -1 * weeks_in_month)
df['Drainage_Volume_seasonal_shift_1M'] = df.Drainage_Volume_seasonal.shift(
    1 * weeks_in_month)
df['Drainage_Volume_seasonal_shift_2M'] = df.Drainage_Volume_seasonal.shift(
    2 * weeks_in_month)
df['Drainage_Volume_seasonal_shift_3M'] = df.Drainage_Volume_seasonal.shift(
    3 * weeks_in_month)

df['River_Hydrometry_seasonal_shift_r_2M'] = df.River_Hydrometry_seasonal.shift(
    -2 * weeks_in_month)
df['River_Hydrometry_seasonal_shift_r_1M'] = df.River_Hydrometry_seasonal.shift(
    -1 * weeks_in_month)
df['River_Hydrometry_seasonal_shift_1M'] = df.River_Hydrometry_seasonal.shift(
    1 * weeks_in_month)
df['River_Hydrometry_seasonal_shift_2M'] = df.River_Hydrometry_seasonal.shift(
    2 * weeks_in_month)
df['River_Hydrometry_seasonal_shift_3M'] = df.River_Hydrometry_seasonal.shift(
    3 * weeks_in_month)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 4))
sns.lineplot(x=df.Date,
             y=df.Temperature_seasonal_shift_r_2M,
             label='shifted by -2 month',
             ax=ax,
             color='lightblue')
sns.lineplot(x=df.Date,
             y=df.Temperature_seasonal_shift_r_1M,
             label='shifted by -1 month',
             ax=ax,
             color='skyblue')

sns.lineplot(x=df.Date,
             y=df.Temperature_seasonal,
             label='original',
             ax=ax,
             color='darkorange')

sns.lineplot(x=df.Date,
             y=df.Temperature_seasonal_shift_1M,
             label='shifted by 1 month',
             ax=ax,
             color='dodgerblue')
sns.lineplot(x=df.Date,
             y=df.Temperature_seasonal_shift_2M,
             label='shifted by 2 month',
             ax=ax,
             color='blue')
sns.lineplot(x=df.Date,
             y=df.Temperature_seasonal_shift_3M,
             label='shifted by 3 month',
             ax=ax,
             color='navy')

ax.set_title('Shifted Time Series', fontsize=16)

ax.set_xlim([date(2017, 6, 30), date(2020, 6, 30)])
ax.set_ylabel(ylabel='Temperature Bastia Umbra', fontsize=14)

plt.show()

# %% [markdown] papermill={"duration": 0.101968, "end_time": "2021-01-17T15:25:20.309732", "exception": false, "start_time": "2021-01-17T15:25:20.207764", "status": "completed"} tags=[]
# # Exploratory Data Analysis
#
# Let's begin by plotting the seasonal components of each feature and comparing the minima and maxima. By doing this, we can already gain some insights:
# * The depth to groundwater reaches its maximum around May/June and its minimum around November/December
# * The temperature reaches its maxmium around August and its minimum around January
# * The volume reaches its maximum around June and its minimum around August/September. It takes longer to reach its maximum than to reach its minimum.
# * The hydrometry reaches its maximum around March and its minimum around September
#
# * The volume and hydrometry reach their minimum roughly around the same time
# * The volume and hydrometry reach their minimum when the temperature reaches its maximum
# * Temperature lags begind depth to groundwater by around 2 to 3 months

# %% _kg_hide-input=true papermill={"duration": 1.424459, "end_time": "2021-01-17T15:25:21.835676", "exception": false, "start_time": "2021-01-17T15:25:20.411217", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=5, ncols=1, figsize=(15, 12))
f.suptitle('Seasonal Components of Features', fontsize=16)
sns.lineplot(x=df.Date,
             y=df.Depth_to_Groundwater_seasonal,
             ax=ax[0],
             color='dodgerblue',
             label='P25')
ax[0].set_ylabel(ylabel='Depth to Groundwater', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.Temperature_seasonal,
             ax=ax[1],
             color='dodgerblue',
             label='Bastia Umbra')
ax[1].set_ylabel(ylabel='Temperature', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.Drainage_Volume_seasonal,
             ax=ax[2],
             color='dodgerblue')
ax[2].set_ylabel(ylabel='Volume', fontsize=14)

sns.lineplot(x=df.Date,
             y=df.River_Hydrometry_seasonal,
             ax=ax[3],
             color='dodgerblue')
ax[3].set_ylabel(ylabel='Hydrometry', fontsize=14)

sns.lineplot(x=df.Date, y=df.Rainfall_seasonal, ax=ax[4], color='dodgerblue')
ax[4].set_ylabel(ylabel='Rainfall', fontsize=14)

for i in range(5):
    ax[i].set_xlim([date(2017, 9, 30), date(2020, 6, 30)])
plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.104954, "end_time": "2021-01-17T15:25:22.047263", "exception": false, "start_time": "2021-01-17T15:25:21.942309", "status": "completed"} tags=[]
# We can see that the correlation to the target variables increases if we use the time shifted features in comparison to the original features.

# %% _kg_hide-input=true _kg_hide-output=false papermill={"duration": 0.783059, "end_time": "2021-01-17T15:25:22.934789", "exception": false, "start_time": "2021-01-17T15:25:22.151730", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

original_cols = [
    'Depth_to_Groundwater_seasonal', 'Temperature_seasonal',
    'Drainage_Volume_seasonal', 'River_Hydrometry_seasonal'
]

corrmat = df[original_cols].corr()

sns.heatmap(corrmat, annot=True, vmin=-1, vmax=1, cmap='coolwarm_r', ax=ax[0])
ax[0].set_title('Correlation Matrix of Original Features', fontsize=16)

shifted_cols = [
    'Depth_to_Groundwater_seasonal', 'Temperature_seasonal_shift_r_2M',
    'Drainage_Volume_seasonal_shift_1M', 'River_Hydrometry_seasonal_shift_3M'
]
corrmat = df[shifted_cols].corr()

sns.heatmap(corrmat, annot=True, vmin=-1, vmax=1, cmap='coolwarm_r', ax=ax[1])
ax[1].set_title('Correlation Matrix of Shifted Features', fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.105977, "end_time": "2021-01-17T15:25:23.148401", "exception": false, "start_time": "2021-01-17T15:25:23.042424", "status": "completed"} tags=[]
# ## Autocorrelation Analysis
#
# This EDA step is especially important when using [ARIMA](#ARIMA). The autocorrelation analysis helps to identify the AR and MA parameters for the [ARIMA](#ARIMA) model.
#
# Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
#
# * **Autocorrelation  Function (ACF)**: Correlation between time series with a lagged version of itself. The autocorrelation function starts a lag 0, which is the correlation of the time series with itself and therefore results in a correlation of 1. -> <font color='blue'>MA parameter is q significant lags</font>
# * **Partial Autocorrelation Function (PACF)**: Additional correlation explained by each successive lagged term -> <font color='purple'>AR parameter is p significant lags</font>
#
# Autocorrelation pluts help in detecting seasonality.
#
# A good starting point for the AR parameter of the model may be TODO.
#
#
# As we can infer from the graph above, the autocorrelation continues to decrease as the lag increases, confirming that there is no linear association between observations separated by larger lags.
#
# For the AR process, we expect that the ACF plot will gradually decrease and simultaneously the PACF should have a sharp drop after p significant lags. To define a MA process, we expect the opposite from the ACF and PACF plots, meaning that: the ACF should show a sharp drop after a certain q number of lags while PACF should show a geometric or gradual decreasing trend.

# %% papermill={"duration": 0.284427, "end_time": "2021-01-17T15:25:23.540183", "exception": false, "start_time": "2021-01-17T15:25:23.255756", "status": "completed"} tags=[]
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df.Depth_to_Groundwater_diff_1)
plt.show()

# %% [markdown] papermill={"duration": 0.107098, "end_time": "2021-01-17T15:25:23.755173", "exception": false, "start_time": "2021-01-17T15:25:23.648075", "status": "completed"} tags=[]
# We can see some sinusoidal shape in both ACF and PACF functions. This suggests that both AR and MA processes are present.

# %% papermill={"duration": 0.486149, "end_time": "2021-01-17T15:25:24.348758", "exception": false, "start_time": "2021-01-17T15:25:23.862609", "status": "completed"} tags=[]
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

plot_acf(df.Depth_to_Groundwater_diff_1, lags=100, ax=ax[0])
plot_pacf(df.Depth_to_Groundwater_diff_1, lags=100, ax=ax[1])
plt.show()

# %% _kg_hide-input=true _kg_hide-output=true papermill={"duration": 0.118076, "end_time": "2021-01-17T15:25:24.575605", "exception": false, "start_time": "2021-01-17T15:25:24.457529", "status": "completed"} tags=[]
"""
## Spectral Analysis
to analyse cyclic behavior
Frequency domain analysis

## Trend estimation and decomposition
used for seasonal adjustment
"""

# %% [markdown] papermill={"duration": 0.108611, "end_time": "2021-01-17T15:25:24.793172", "exception": false, "start_time": "2021-01-17T15:25:24.684561", "status": "completed"} tags=[]
# # Cross Validation
#
# For cross validation, you can use the [Time Series Split](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) library.
# In [Time Series Forecasting: Building Intuition](https://www.kaggle.com/iamleonie/time-series-forecasting-building-intuition), I go into depth about different types of time series problems and their cross validation methods.

# %% papermill={"duration": 0.132608, "end_time": "2021-01-17T15:25:25.034998", "exception": false, "start_time": "2021-01-17T15:25:24.902390", "status": "completed"} tags=[]
from sklearn.model_selection import TimeSeriesSplit

N_SPLITS = 3

X = df.Date
y = df.Depth_to_Groundwater

folds = TimeSeriesSplit(n_splits=N_SPLITS)

# %% _kg_hide-input=true papermill={"duration": 1.60402, "end_time": "2021-01-17T15:25:26.749090", "exception": false, "start_time": "2021-01-17T15:25:25.145070", "status": "completed"} tags=[]
f, ax = plt.subplots(nrows=N_SPLITS, ncols=2, figsize=(16, 9))

for i, (train_index, valid_index) in enumerate(folds.split(X)):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    sns.lineplot(x=X_train,
                 y=y_train,
                 ax=ax[i, 0],
                 color='dodgerblue',
                 label='train')
    sns.lineplot(
        x=X_train[len(X_train) - len(X_valid):(len(X_train) - len(X_valid) +
                                               len(X_valid))],
        y=y_train[len(X_train) - len(X_valid):(len(X_train) - len(X_valid) +
                                               len(X_valid))],
        ax=ax[i, 1],
        color='dodgerblue',
        label='train')

    for j in range(2):
        sns.lineplot(x=X_valid,
                     y=y_valid,
                     ax=ax[i, j],
                     color='darkorange',
                     label='validation')
    ax[i, 0].set_title(
        f"Rolling Window with Adjusting Training Size (Split {i+1})",
        fontsize=16)
    ax[i, 1].set_title(
        f"Rolling Window with Constant Training Size (Split {i+1})",
        fontsize=16)

for i in range(N_SPLITS):
    ax[i, 0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
    ax[i, 1].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.117539, "end_time": "2021-01-17T15:25:26.982761", "exception": false, "start_time": "2021-01-17T15:25:26.865222", "status": "completed"} tags=[]
# # Models
#
# Time series can be either **univariate or multivariate**:
# * **Univariate** time series only has a single time-dependent variable.
# * **Multivariate** time series have a multiple time-dependent variable.
#
# Our example originally is a multivariate time series because its has multiple features that are all time-dependent. However, by only looking at the target variable `Depth to Groundwater` we can convert it to a univariate time series.
#
# We will focus on a **quarterly forecast**. We will use the **Q2 2020 as test data** and the remaining data will be **split by quarter for cross validation**.
#
# We will evaluate the Mean Absolute Error (MAE) and the Root Mean Square Error (RMSE) of the models. For metrics are better the smaller they are.
#
#

# %% [markdown] papermill={"duration": 0.114342, "end_time": "2021-01-17T15:25:27.211919", "exception": false, "start_time": "2021-01-17T15:25:27.097577", "status": "completed"} tags=[]
# ## Models for Univariate Time Series
#
# * Stochastic Models
#     * [Naive Approach](#Naive-Approach)<br>
#     * [Moving Average](#Moving-Average)<br>
#     * [Exponential Smoothing](#MExponential-Smoothing)<br>
#     * [ARIMA](#ARIMA)<br>
#     * [Prophet](#Prophet)<br>
# * Deep Learning
#     * [LSTM](#LSTM)<br>
#     * [GRU](#GRU)<br>

# %% _kg_hide-input=true _kg_hide-output=true papermill={"duration": 0.136393, "end_time": "2021-01-17T15:25:27.462982", "exception": false, "start_time": "2021-01-17T15:25:27.326589", "status": "completed"} tags=[]
df['quarter_idx'] = (df.quarter != df.quarter.shift(1)).cumsum()

target = 'Depth_to_Groundwater'
features = [feature for feature in df.columns if feature != target]

N_SPLITS = 46

X = df[df.quarter_idx < N_SPLITS][features]
y = df[df.quarter_idx < N_SPLITS][target]

X_test = df[df.quarter_idx == N_SPLITS][features].reset_index(drop=True)
y_test = df[df.quarter_idx == N_SPLITS][target].reset_index(drop=True)

# %% _kg_hide-input=true papermill={"duration": 0.60257, "end_time": "2021-01-17T15:25:28.180699", "exception": false, "start_time": "2021-01-17T15:25:27.578129", "status": "completed"} tags=[]
folds = np.linspace(0, N_SPLITS - 3, num=N_SPLITS - 2)

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

sns.lineplot(x=X.Date, y=y, ax=ax[0], color='dodgerblue', label='train')
sns.lineplot(x=X_test.Date,
             y=y_test,
             ax=ax[0],
             color='darkorange',
             label='test')

sns.lineplot(x=df.Date, y=df.quarter_idx, ax=ax[1], color='dodgerblue')
ax[0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
ax[1].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
ax[1].set_ylim([0, N_SPLITS + 1])
#ax[0].set_ylim([-28, -23])

plt.show()


# %% _kg_hide-input=true _kg_hide-output=true papermill={"duration": 0.134872, "end_time": "2021-01-17T15:25:28.432001", "exception": false, "start_time": "2021-01-17T15:25:28.297129", "status": "completed"} tags=[]
def plot_approach_evaluation(y_pred, score_mae, score_rsme, approach_name,
                             y_valid, X, folds, y, y_valid_pred, y_test,
                             X_test):
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    f.suptitle(approach_name, fontsize=16)
    sns.lineplot(x=X.Date,
                 y=y,
                 ax=ax[0],
                 color='dodgerblue',
                 label='Training',
                 linewidth=2)
    sns.lineplot(x=X_test.Date,
                 y=y_test,
                 ax=ax[0],
                 color='gold',
                 label='Ground Truth',
                 linewidth=2)  #navajowhite
    sns.lineplot(x=X_test.Date,
                 y=y_pred,
                 ax=ax[0],
                 color='darkorange',
                 label='Predicted',
                 linewidth=2)
    ax[0].set_xlim([date(2018, 6, 30), date(2020, 6, 30)])
    ax[0].set_ylim([-27, -23])
    ax[0].set_title(
        f'Prediction \n MAE: {mean_absolute_error(y_test, y_pred):.2f}, RSME: {math.sqrt(mean_squared_error(y_valid, y_valid_pred)):.2f}',
        fontsize=14)
    ax[0].set_xlabel(xlabel='Date', fontsize=14)
    ax[0].set_ylabel(ylabel='Depth to Groundwater P25', fontsize=14)

    sns.lineplot(x=folds, y=score_mae, color='gold', label='MAE',
                 ax=ax[1])  #marker='o',
    sns.lineplot(x=folds,
                 y=score_rsme,
                 color='indianred',
                 label='RSME',
                 ax=ax[1])
    ax[1].set_title('Loss', fontsize=14)
    ax[1].set_xlabel(xlabel='Fold', fontsize=14)
    ax[1].set_ylabel(ylabel='Loss', fontsize=14)
    ax[1].set_ylim([0, 4])
    plt.show()


# %% [markdown] papermill={"duration": 0.124053, "end_time": "2021-01-17T15:25:28.708573", "exception": false, "start_time": "2021-01-17T15:25:28.584520", "status": "completed"} tags=[]
# ### Naive Approach
#
# $\hat y_{t+1} = y_t$

# %% _kg_hide-input=true _kg_hide-output=false papermill={"duration": 0.747855, "end_time": "2021-01-17T15:25:29.570757", "exception": false, "start_time": "2021-01-17T15:25:28.822902", "status": "completed"} tags=[]
score_mae = []
score_rsme = []
for fold, valid_quarter_id in enumerate(range(2, N_SPLITS)):
    # Get indices for this fold
    train_index = df[df.quarter_idx < valid_quarter_id].index
    valid_index = df[df.quarter_idx == valid_quarter_id].index

    # Prepare training and validation data for this fold
    #X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Initialize y_valid_pred
    y_valid_pred = pd.Series(np.ones(len(y_valid)))

    # Prediction: Naive approach
    y_valid_pred = y_valid_pred * y_train.iloc[-1]

    # Calcuate metrics
    score_mae.append(mean_absolute_error(y_valid, y_valid_pred))
    score_rsme.append(math.sqrt(mean_squared_error(y_valid, y_valid_pred)))

y_pred = pd.Series(np.ones(len(X_test))) * y.iloc[-1]

plot_approach_evaluation(y_pred, score_mae, score_rsme, 'Naive Approach',
                         y_valid, X, folds, y, y_valid_pred, y_test, X_test)

# %% [markdown] papermill={"duration": 0.120142, "end_time": "2021-01-17T15:25:29.808436", "exception": false, "start_time": "2021-01-17T15:25:29.688294", "status": "completed"} tags=[]
# ### Moving Average

# %% _kg_hide-input=true papermill={"duration": 1.235442, "end_time": "2021-01-17T15:25:31.161562", "exception": false, "start_time": "2021-01-17T15:25:29.926120", "status": "completed"} tags=[]
score_mae = []
score_rsme = []
for fold, valid_quarter_id in enumerate(range(2, N_SPLITS)):
    # Get indices for this fold
    train_index = df[df.quarter_idx < valid_quarter_id].index
    valid_index = df[df.quarter_idx == valid_quarter_id].index

    # Prepare training and validation data for this fold
    #X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Initialize y_valid_pred
    y_valid_pred = pd.Series(np.ones(len(y_valid)))

    # Prediction: Naive approach
    for i in range(len(y_valid_pred)):
        y_valid_pred.iloc[i] = y_train.append(
            y_valid_pred.iloc[:(i)]).reset_index(
                drop=True).rolling(4).mean().iloc[-1]

    # Calcuate metrics
    score_mae.append(mean_absolute_error(y_valid, y_valid_pred))
    score_rsme.append(math.sqrt(mean_squared_error(y_valid, y_valid_pred)))

y_pred = pd.Series(np.zeros(len(X_test)))

for i in range(len(y_pred)):
    y_pred.iloc[i] = y.append(
        y_pred.iloc[:(i)]).reset_index(drop=True).rolling(4).mean().iloc[-1]

plot_approach_evaluation(y_pred, score_mae, score_rsme,
                         'Moving Average (Window = 4 Weeks)', y_valid, X,
                         folds, y, y_valid_pred, y_test, X_test)

# %% [markdown] papermill={"duration": 0.118398, "end_time": "2021-01-17T15:25:31.400077", "exception": false, "start_time": "2021-01-17T15:25:31.281679", "status": "completed"} tags=[]
# Neither the Naive Approach nor the Moving Average Approach are yielding good results for our example. Usually, these approaches serve as a benchmark rather than the method of choice.

# %% [markdown] papermill={"duration": 0.121728, "end_time": "2021-01-17T15:25:31.643253", "exception": false, "start_time": "2021-01-17T15:25:31.521525", "status": "completed"} tags=[]
# ### ARIMA
# The Auto-Regressive Integrated Moving Average (ARIMA) model describes the **autocorrelations** in the data. The model assumes that the time-series is **stationary**. It consists of three main parts:
# * <font color='purple'>Auto-Regressive (AR) filter (long term)</font>:
#
#     $\color{purple}{y_t = c + \alpha_1 y_{t-1} + \dots \alpha_{\color{purple}p}y_{t-\color{purple}p} + \epsilon_t = c + \sum_{i=1}^p{\alpha_i}y_{t-i} + \epsilon_t}$  -> p
# * <font color='orange'> Integration filter (stochastic trend)</font>
#
#     -> d
# * <font color='blue'>Moving Average (MA) filter (short term)</font>:
#
#     $\color{blue}{y_t = c + \epsilon_t + \beta_1 \epsilon_{t-1} + \dots + \beta_{q} \epsilon_{t-q} = c + \epsilon_t + \sum_{i=1}^q{\beta_i}\epsilon_{t-i}} $  -> q
#
#
# **ARIMA**: $y_t = c + \color{purple}{\alpha_1 y_{t-1} + \dots + \alpha_{\color{purple}p}y_{t-\color{purple}p}}
# + \color{blue}{\epsilon_t + \beta_1 \epsilon_{t-1} + \dots + \beta_{q} \epsilon_{t-q}}$
#
#
# ARIMA(
# <font color='purple'>p</font>,
# <font color='orange'>d</font>,
# <font color='blue'>q</font>)
#
# * <font color='purple'>p</font>: Lag order (to determine see  PACF in [Autocorrelation Analysis](#Autocorrelation-Analysis))
# * <font color='orange'>d</font>: Degree of differencing. (to determine see  Differencing in [Stationarity](#Stationarity))
# * <font color='blue'>q</font>: Order of moving average (to determine see  ACF in [Autocorrelation Analysis](#Autocorrelation-Analysis))
#
# In our example, we can use <font color='orange'>d=0</font> if we use the feature `Depth_to_Groundwater_diff_1`, which is `Depth_to_Groundwater` differenced by the first degree. Otherwise, if we were to use the non-stationary feature `Depth_to_Groundwater` as it is, we should set <font color='orange'>d=1</font>.
#
# (work in progress...)

# %% _kg_hide-input=true _kg_hide-output=false papermill={"duration": 4.61287, "end_time": "2021-01-17T15:25:36.376551", "exception": false, "start_time": "2021-01-17T15:25:31.763681", "status": "completed"} tags=[]
from statsmodels.tsa.arima.model import ARIMA

score_mae = []
score_rsme = []

for fold, valid_quarter_id in enumerate(range(2, N_SPLITS)):
    # Get indices for this fold
    train_index = df[df.quarter_idx < valid_quarter_id].index
    valid_index = df[df.quarter_idx == valid_quarter_id].index

    # Prepare training and validation data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Fit model with Vector Auto Regression (VAR)
    model = ARIMA(y_train, order=(1, 1, 1))
    model_fit = model.fit()

    # Prediction with Vector Auto Regression (VAR)
    y_valid_pred = model_fit.predict(valid_index[0], valid_index[-1])

    # Calcuate metrics
    score_mae.append(mean_absolute_error(y_valid, y_valid_pred))
    score_rsme.append(math.sqrt(mean_squared_error(y_valid, y_valid_pred)))

# Fit model with Vector Auto Regression (VAR)
model = ARIMA(y, order=(1, 1, 1))
model_fit = model.fit()

# Prediction with Vector Auto Regression (VAR)
y_pred = model_fit.predict(y.index[-1] + 1,
                           y.index[-1] + len(y_test)).reset_index(drop=True)
plot_approach_evaluation(y_pred, score_mae, score_rsme, 'ARIMA', y_valid, X,
                         folds, y, y_valid_pred, y_test, X_test)

# %% [markdown] papermill={"duration": 0.122224, "end_time": "2021-01-17T15:25:36.622529", "exception": false, "start_time": "2021-01-17T15:25:36.500305", "status": "completed"} tags=[]
# ## Models for Multivariate Time Series
#
# ### Vector Auto Regression (VAR)

# %% _kg_hide-input=true papermill={"duration": 1.136181, "end_time": "2021-01-17T15:25:37.879666", "exception": false, "start_time": "2021-01-17T15:25:36.743485", "status": "completed"} tags=[]
from statsmodels.tsa.api import VAR

score_mae = []
score_rsme = []

features = ['Temperature', 'Drainage_Volume', 'River_Hydrometry', 'Rainfall']
for fold, valid_quarter_id in enumerate(range(2, N_SPLITS)):
    # Get indices for this fold
    train_index = df[df.quarter_idx < valid_quarter_id].index
    valid_index = df[df.quarter_idx == valid_quarter_id].index

    # Prepare training and validation data for this fold
    X_train, X_valid = X.iloc[train_index][features], X.iloc[valid_index][
        features]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Fit model with Vector Auto Regression (VAR)
    model = VAR(pd.concat([y_train, X_train], axis=1))
    model_fit = model.fit()

    # Prediction with Vector Auto Regression (VAR)
    y_valid_pred = model_fit.forecast(model_fit.y, steps=len(X_valid))
    y_valid_pred = pd.Series(y_valid_pred[:, 0])

    # Calcuate metrics
    score_mae.append(mean_absolute_error(y_valid, y_valid_pred))
    score_rsme.append(math.sqrt(mean_squared_error(y_valid, y_valid_pred)))

# Fit model with Vector Auto Regression (VAR)
model = VAR(pd.concat([y, X[features]], axis=1))
model_fit = model.fit()

# Prediction with Vector Auto Regression (VAR)
y_pred = model_fit.forecast(model_fit.y, steps=len(X_valid))
y_pred = pd.Series(y_pred[:, 0])

plot_approach_evaluation(y_pred, score_mae, score_rsme,
                         'Vector Auto Regression (VAR)', y_valid, X, folds, y,
                         y_valid_pred, y_test, X_test)

# %% [markdown] papermill={"duration": 0.12475, "end_time": "2021-01-17T15:25:38.132380", "exception": false, "start_time": "2021-01-17T15:25:38.007630", "status": "completed"} tags=[]
# # Additional Ressources
#
# My other notebook on time series:
# * [Time Series Forecasting: Building Intuition](https://www.kaggle.com/iamleonie/time-series-forecasting-building-intuition)
#
# Here are some additional ressources that helped me learn about time series
# * [Getting started with Time Series using Pandas ](https://www.kaggle.com/parulpandey/getting-started-with-time-series-using-pandas)
# * [Time Series Analysis || An Introductory Start](https://www.kaggle.com/janiobachmann/time-series-analysis-an-introductory-start)
# * [Everything you can do with a time series](https://www.kaggle.com/thebrownviking20/everything-you-can-do-with-a-time-series)
# * [Deep Learning for Time Series | Dimitry Larko | Kaggle Days](https://www.youtube.com/watch?v=svNwWSgz2NM)
# * [Encoding Cyclical Features for Deep Learning](https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning)
# * [Tamara Louie: Applying Statistical Modeling & Machine Learning to Perform Time-Series Forecasting](https://www.youtube.com/watch?v=JntA9XaTebs)
# * [Forecasting: Principles and Practice](https://otexts.com/fpp2/)
# * [Easy Guide on Time Series Forecasting](https://beingdatum.com/time-series-forecasting/)
#
# # Useful Libraries
# * [statsmodels](https://www.statsmodels.org)
# * [Pandas Time series / date functionality](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
# * [Time Series Split](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

# %% _kg_hide-input=true _kg_hide-output=true papermill={"duration": 0.133916, "end_time": "2021-01-17T15:25:38.391010", "exception": false, "start_time": "2021-01-17T15:25:38.257094", "status": "completed"} tags=[]
"""
### Exponential Smoothing
based on a description of the **trend and seasonality** in the data

### Prophet 

### LSTM

### GRU

(work in progress)

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)

Recurrent Neural Network (RNN)
"""
