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

# %% _kg_hide-input=true _kg_hide-output=true papermill={"duration": 0.059258, "end_time": "2021-10-08T04:14:39.074653", "exception": false, "start_time": "2021-10-08T04:14:39.015395", "status": "completed"} tags=[]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown] papermill={"duration": 0.027866, "end_time": "2021-10-08T04:14:39.135536", "exception": false, "start_time": "2021-10-08T04:14:39.107670", "status": "completed"} tags=[]
# ## Customer Segmentation
#
# <img src="https://github.com/KarnikaKapoor/Files/blob/main/Colorful%20Handwritten%20About%20Me%20Blank%20Education%20Presentation.gif?raw=true">
#
# In this project, I will be performing an unsupervised clustering of data on the customer's records from a groceries firm's database. Customer segmentation is the practice of separating customers into groups that reflect similarities among customers in each cluster. I will divide customers into segments to optimize the significance of each customer to the business. To modify products according to distinct needs and behaviours of the customers. It also helps the business to cater to the concerns of different types of customers.
#
#
#    <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
## TABLE OF CONTENTS
#
# * [1. IMPORTING LIBRARIES](#1)
#
# * [2. LOADING DATA](#2)
#
# * [3. DATA CLEANING](#3)
#
# * [4. DATA PREPROCESSING](#4)
#
# * [5. DIMENSIONALITY REDUCTION](#5)
#
# * [6. CLUSTERING](#6)
#
# * [7. EVALUATING MODELS](#7)
#
# * [8. PROFILING](#8)
#
# * [9. CONCLUSION](#9)
#
# * [10. END](#10)
#

# %% [markdown] papermill={"duration": 0.028812, "end_time": "2021-10-08T04:14:39.192253", "exception": false, "start_time": "2021-10-08T04:14:39.163441", "status": "completed"} tags=[]
# <a id="1"></a>
# ## IMPORTING LIBRARIES

# %% papermill={"duration": 1.53841, "end_time": "2021-10-08T04:14:40.759736", "exception": false, "start_time": "2021-10-08T04:14:39.221326", "status": "completed"} tags=[]
#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

# %% [markdown] papermill={"duration": 0.029156, "end_time": "2021-10-08T04:14:40.817772", "exception": false, "start_time": "2021-10-08T04:14:40.788616", "status": "completed"} tags=[]
# <a id="2"></a>
# ## LOADING DATA

# %% papermill={"duration": 0.087752, "end_time": "2021-10-08T04:14:40.933340", "exception": false, "start_time": "2021-10-08T04:14:40.845588", "status": "completed"} tags=[]
#Loading the dataset
data = pd.read_csv("input/marketing_campaign.csv", sep="\t")
print("Number of datapoints:", len(data))
data.head()

# %% [markdown] papermill={"duration": 0.028, "end_time": "2021-10-08T04:14:40.989700", "exception": false, "start_time": "2021-10-08T04:14:40.961700", "status": "completed"} tags=[]
# <img src="https://github.com/KarnikaKapoor/Files/blob/main/Colorful%20Handwritten%20About%20Me%20Blank%20Education%20Presentation.png?raw=true">
#
# For more information on the attributes visit [here](https://www.kaggle.com/imakash3011/customer-personality-analysis).
#
# <a id="3"></a>
# ## DATA CLEANING
#
#
# **In this section**
# * Data Cleaning
# * Feature Engineering
#
# In order to, get a full grasp of what steps should I be taking to clean the dataset.
# Let us have a look at the information in data.
#

# %% papermill={"duration": 0.058629, "end_time": "2021-10-08T04:14:41.076744", "exception": false, "start_time": "2021-10-08T04:14:41.018115", "status": "completed"} tags=[]
#Information on features
data.info()

# %% [markdown] papermill={"duration": 0.028603, "end_time": "2021-10-08T04:14:41.136192", "exception": false, "start_time": "2021-10-08T04:14:41.107589", "status": "completed"} tags=[]
# **From the above output, we can conclude and note that:**
#
# * There are missing values in income
# * Dt_Customer that indicates the date a customer joined the database is not parsed as DateTime
# * There are some categorical features in our data frame; as there are some features in dtype: object). So we will need to encode them into numeric forms later.
#
# First of all, for the missing values, I am simply going to drop the rows that have missing income values.

# %% papermill={"duration": 0.042312, "end_time": "2021-10-08T04:14:41.207425", "exception": false, "start_time": "2021-10-08T04:14:41.165113", "status": "completed"} tags=[]
#To remove the NA values
data = data.dropna()
print(
    "The total number of data-points after removing the rows with missing values are:",
    len(data))

# %% [markdown] papermill={"duration": 0.028861, "end_time": "2021-10-08T04:14:41.266284", "exception": false, "start_time": "2021-10-08T04:14:41.237423", "status": "completed"} tags=[]
# In the next step, I am going to create a feature out of **"Dt_Customer"** that indicates the number of days a customer is registered in the firm's database. However, in order to keep it simple, I am taking this value relative to the most recent customer in the record.
#
# Thus to get the values I must check the newest and oldest recorded dates.

# %% papermill={"duration": 0.046416, "end_time": "2021-10-08T04:14:41.341737", "exception": false, "start_time": "2021-10-08T04:14:41.295321", "status": "completed"} tags=[]
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)
#Dates of the newest and oldest recorded customer
print("The newest customer's enrolment date in therecords:", max(dates))
print("The oldest customer's enrolment date in the records:", min(dates))

# %% [markdown] papermill={"duration": 0.029401, "end_time": "2021-10-08T04:14:41.400477", "exception": false, "start_time": "2021-10-08T04:14:41.371076", "status": "completed"} tags=[]
# Creating a feature **("Customer_For")** of the number of days the customers started to shop in the store relative to the last recorded date

# %% papermill={"duration": 0.051941, "end_time": "2021-10-08T04:14:41.482563", "exception": false, "start_time": "2021-10-08T04:14:41.430622", "status": "completed"} tags=[]
#Created a feature "Customer_For"
days = []
d1 = max(dates)  #taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

# %% [markdown] papermill={"duration": 0.040084, "end_time": "2021-10-08T04:14:41.565744", "exception": false, "start_time": "2021-10-08T04:14:41.525660", "status": "completed"} tags=[]
# Now we will be exploring the unique values in the categorical features to get a clear idea of the data.

# %% papermill={"duration": 0.043279, "end_time": "2021-10-08T04:14:41.640142", "exception": false, "start_time": "2021-10-08T04:14:41.596863", "status": "completed"} tags=[]
print("Total categories in the feature Marital_Status:\n",
      data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n",
      data["Education"].value_counts())

# %% [markdown] papermill={"duration": 0.029496, "end_time": "2021-10-08T04:14:41.700019", "exception": false, "start_time": "2021-10-08T04:14:41.670523", "status": "completed"} tags=[]
# **In the next bit, I will be performing the following steps to engineer some new features:**
#
# * Extract the **"Age"** of a customer by the **"Year_Birth"** indicating the birth year of the respective person.
# * Create another feature **"Spent"** indicating the total amount spent by the customer in various categories over the span of two years.
# * Create another feature **"Living_With"** out of **"Marital_Status"** to extract the living situation of couples.
# * Create a feature **"Children"** to indicate total children in a household that is, kids and teenagers.
# * To get further clarity of household, Creating feature indicating **"Family_Size"**
# * Create a feature **"Is_Parent"** to indicate parenthood status
# * Lastly, I will create three categories in the **"Education"** by simplifying its value counts.
# * Dropping some of the redundant features

# %% papermill={"duration": 0.055238, "end_time": "2021-10-08T04:14:41.784923", "exception": false, "start_time": "2021-10-08T04:14:41.729685", "status": "completed"} tags=[]
#Feature Engineering
#Age of customer today
data["Age"] = 2021 - data["Year_Birth"]

#Total spendings on various items
data["Spent"] = data["MntWines"] + data["MntFruits"] + data[
    "MntMeatProducts"] + data["MntFishProducts"] + data[
        "MntSweetProducts"] + data["MntGoldProds"]

#Deriving living situation by marital status"Alone"
data["Living_With"] = data["Marital_Status"].replace({
    "Married": "Partner",
    "Together": "Partner",
    "Absurd": "Alone",
    "Widow": "Alone",
    "YOLO": "Alone",
    "Divorced": "Alone",
    "Single": "Alone",
})

#Feature indicating total children living in the household
data["Children"] = data["Kidhome"] + data["Teenhome"]

#Feature for total members in the householde
data["Family_Size"] = data["Living_With"].replace({
    "Alone": 1,
    "Partner": 2
}) + data["Children"]

#Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

#Segmenting education levels in three groups
data["Education"] = data["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate"
})

#For clarity
data = data.rename(
    columns={
        "MntWines": "Wines",
        "MntFruits": "Fruits",
        "MntMeatProducts": "Meat",
        "MntFishProducts": "Fish",
        "MntSweetProducts": "Sweets",
        "MntGoldProds": "Gold"
    })

#Dropping some of the redundant features
to_drop = [
    "Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue",
    "Year_Birth", "ID"
]
data = data.drop(to_drop, axis=1)

# %% [markdown] papermill={"duration": 0.029572, "end_time": "2021-10-08T04:14:41.843846", "exception": false, "start_time": "2021-10-08T04:14:41.814274", "status": "completed"} tags=[]
# Now that we have some new features let's have a look at the data's stats.

# %% papermill={"duration": 0.109706, "end_time": "2021-10-08T04:14:41.983356", "exception": false, "start_time": "2021-10-08T04:14:41.873650", "status": "completed"} tags=[]
data.describe()

# %% [markdown] papermill={"duration": 0.030087, "end_time": "2021-10-08T04:14:42.043835", "exception": false, "start_time": "2021-10-08T04:14:42.013748", "status": "completed"} tags=[]
# The above stats show some discrepancies in mean Income and Age and max Income and age.
#
# Do note that  max-age is 128 years, As I calculated the age that would be today (i.e. 2021) and the data is old.
#
# I must take a look at the broader view of the data.
# I will plot some of the selected features.

# %% papermill={"duration": 8.979335, "end_time": "2021-10-08T04:14:51.053279", "exception": false, "start_time": "2021-10-08T04:14:42.073944", "status": "completed"} tags=[]
#To plot some selected features
#Setting up colors prefrences
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(
    ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
#Plotting following features
To_Plot = ["Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(data[To_Plot], hue="Is_Parent", palette=(["#682F2F", "#F3AB60"]))
#Taking hue
plt.show()

# %% [markdown] papermill={"duration": 0.046896, "end_time": "2021-10-08T04:14:51.147774", "exception": false, "start_time": "2021-10-08T04:14:51.100878", "status": "completed"} tags=[]
# Clearly, there are a few outliers in the Income and Age features.
# I will be deleting the outliers in the data.

# %% papermill={"duration": 0.05833, "end_time": "2021-10-08T04:14:51.252647", "exception": false, "start_time": "2021-10-08T04:14:51.194317", "status": "completed"} tags=[]
#Dropping the outliers by setting a cap on Age and income.
data = data[(data["Age"] < 90)]
data = data[(data["Income"] < 600000)]
print("The total number of data-points after removing the outliers are:",
      len(data))

# %% [markdown] papermill={"duration": 0.046761, "end_time": "2021-10-08T04:14:51.348041", "exception": false, "start_time": "2021-10-08T04:14:51.301280", "status": "completed"} tags=[]
# Next, let us look at the correlation amongst the features.
# (Excluding the categorical attributes at this point)

# %% papermill={"duration": 4.47676, "end_time": "2021-10-08T04:14:55.873081", "exception": false, "start_time": "2021-10-08T04:14:51.396321", "status": "completed"} tags=[]
#correlation matrix
corrmat = data.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corrmat, annot=True, cmap=cmap, center=0)

# %% [markdown] papermill={"duration": 0.063054, "end_time": "2021-10-08T04:14:56.005374", "exception": false, "start_time": "2021-10-08T04:14:55.942320", "status": "completed"} tags=[]
# The data is quite clean and the new features have been included. I will proceed to the next step. That is, preprocessing the data.
#
# <a id="4"></a>
# ## DATA PREPROCESSING
#
# In this section, I will be preprocessing the data to perform clustering operations.
#
# **The following steps are applied to preprocess the data:**
#
# * Label encoding the categorical features
# * Scaling the features using the standard scaler
# * Creating a subset dataframe for dimensionality reduction

# %% papermill={"duration": 0.085186, "end_time": "2021-10-08T04:14:56.148485", "exception": false, "start_time": "2021-10-08T04:14:56.063299", "status": "completed"} tags=[]
#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

# %% papermill={"duration": 0.078164, "end_time": "2021-10-08T04:14:56.296208", "exception": false, "start_time": "2021-10-08T04:14:56.218044", "status": "completed"} tags=[]
#Label Encoding the object dtypes.
LE = LabelEncoder()
for i in object_cols:
    data[i] = data[[i]].apply(LE.fit_transform)

print("All features are now numerical")

# %% papermill={"duration": 0.075769, "end_time": "2021-10-08T04:14:56.430085", "exception": false, "start_time": "2021-10-08T04:14:56.354316", "status": "completed"} tags=[]
#Creating a copy of data
ds = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = [
    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
    'AcceptedCmp2', 'Complain', 'Response'
]
ds = ds.drop(cols_del, axis=1)
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)
print("All features are now scaled")

# %% papermill={"duration": 0.087305, "end_time": "2021-10-08T04:14:56.575069", "exception": false, "start_time": "2021-10-08T04:14:56.487764", "status": "completed"} tags=[]
#Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
scaled_ds.head()

# %% [markdown] papermill={"duration": 0.062052, "end_time": "2021-10-08T04:14:56.694594", "exception": false, "start_time": "2021-10-08T04:14:56.632542", "status": "completed"} tags=[]
# <a id="5"></a>
# ## DIMENSIONALITY REDUCTION
# In this problem, there are many factors on the basis of which the final classification will be done. These factors are basically attributes or features. The higher the number of features, the harder it is to work with it. Many of these features are correlated, and hence redundant. This is why I will be performing dimensionality reduction on the selected features before putting them through a classifier.
# *Dimensionality reduction is the process of reducing the number of random variables under consideration, by obtaining a set of principal variables.*
#
# **Principal component analysis (PCA)** is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimizing information loss.
#
# **Steps in this section:**
# * Dimensionality reduction with PCA
# * Plotting the reduced dataframe
#
# **Dimensionality reduction with PCA**
#
# For this project, I will be reducing the dimensions to 3.

# %% papermill={"duration": 0.118668, "end_time": "2021-10-08T04:14:56.870748", "exception": false, "start_time": "2021-10-08T04:14:56.752080", "status": "completed"} tags=[]
#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds),
                      columns=(["col1", "col2", "col3"]))
PCA_ds.describe().T

# %% papermill={"duration": 0.367462, "end_time": "2021-10-08T04:14:57.337354", "exception": false, "start_time": "2021-10-08T04:14:56.969892", "status": "completed"} tags=[]
#A 3D Projection Of Data In The Reduced Dimension
x = PCA_ds["col1"]
y = PCA_ds["col2"]
z = PCA_ds["col3"]
#To plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="maroon", marker="o")
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()

# %% [markdown] papermill={"duration": 0.061136, "end_time": "2021-10-08T04:14:57.460284", "exception": false, "start_time": "2021-10-08T04:14:57.399148", "status": "completed"} tags=[]
# <a id="6"></a>
# ## CLUSTERING
#
# Now that I have reduced the attributes to three dimensions, I will be performing clustering via Agglomerative clustering. Agglomerative clustering is a hierarchical clustering method.  It involves merging examples until the desired number of clusters is achieved.
#
# **Steps involved in the Clustering**
# * Elbow Method to determine the number of clusters to be formed
# * Clustering via Agglomerative Clustering
# * Examining the clusters formed via scatter plot

# %% papermill={"duration": 1.74229, "end_time": "2021-10-08T04:14:59.264051", "exception": false, "start_time": "2021-10-08T04:14:57.521761", "status": "completed"} tags=[]
# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()

# %% [markdown] papermill={"duration": 0.063192, "end_time": "2021-10-08T04:14:59.390530", "exception": false, "start_time": "2021-10-08T04:14:59.327338", "status": "completed"} tags=[]
# The above cell indicates that four will be an optimal number of clusters for this data.
# Next, we will be fitting the Agglomerative Clustering Model to get the final clusters.

# %% papermill={"duration": 0.213864, "end_time": "2021-10-08T04:14:59.670550", "exception": false, "start_time": "2021-10-08T04:14:59.456686", "status": "completed"} tags=[]
#Initiating the Agglomerative Clustering model
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
data["Clusters"] = yhat_AC

# %% [markdown] papermill={"duration": 0.063434, "end_time": "2021-10-08T04:14:59.798310", "exception": false, "start_time": "2021-10-08T04:14:59.734876", "status": "completed"} tags=[]
# To examine the clusters formed let's have a look at the 3-D distribution of the clusters.

# %% papermill={"duration": 0.381844, "end_time": "2021-10-08T04:15:00.244379", "exception": false, "start_time": "2021-10-08T04:14:59.862535", "status": "completed"} tags=[]
#Plotting the clusters
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
ax.set_title("The Plot Of The Clusters")
plt.show()

# %% [markdown] papermill={"duration": 0.065606, "end_time": "2021-10-08T04:15:00.376433", "exception": false, "start_time": "2021-10-08T04:15:00.310827", "status": "completed"} tags=[]
# <a id="7"></a>
# ## EVALUATING MODELS
#
# Since this is an unsupervised clustering. We do not have a tagged feature to evaluate or score our model. The purpose of this section is to study the patterns in the clusters formed and determine the nature of the clusters' patterns.
#
# For that, we will be having a look at the data in light of clusters via exploratory data analysis and drawing conclusions.
#
# **Firstly, let us have a look at the group distribution of clustring**

# %% papermill={"duration": 0.470936, "end_time": "2021-10-08T04:15:00.913213", "exception": false, "start_time": "2021-10-08T04:15:00.442277", "status": "completed"} tags=[]
#Plotting countplot of clusters
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
pl = sns.countplot(x=data["Clusters"], palette=pal)
pl.set_title("Distribution Of The Clusters")
plt.show()

# %% [markdown] papermill={"duration": 0.067211, "end_time": "2021-10-08T04:15:01.051651", "exception": false, "start_time": "2021-10-08T04:15:00.984440", "status": "completed"} tags=[]
#
#

# %% [markdown] papermill={"duration": 0.066727, "end_time": "2021-10-08T04:15:01.186736", "exception": false, "start_time": "2021-10-08T04:15:01.120009", "status": "completed"} tags=[]
# The clusters seem to be fairly distributed.

# %% papermill={"duration": 0.510132, "end_time": "2021-10-08T04:15:01.794103", "exception": false, "start_time": "2021-10-08T04:15:01.283971", "status": "completed"} tags=[]
pl = sns.scatterplot(data=data,
                     x=data["Spent"],
                     y=data["Income"],
                     hue=data["Clusters"],
                     palette=pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

# %% [markdown] papermill={"duration": 0.07143, "end_time": "2021-10-08T04:15:01.938650", "exception": false, "start_time": "2021-10-08T04:15:01.867220", "status": "completed"} tags=[]
# **Income vs  spending plot shows the clusters pattern**
# * group 0: high spending & average income
# * group 1: high spending & high income
# * group 2: low spending & low income
# * group 3: high spending & low income
#
# Next, I will be looking at the detailed distribution of clusters as per the various products in the data. Namely: Wines, Fruits, Meat, Fish, Sweets and Gold

# %% papermill={"duration": 4.684948, "end_time": "2021-10-08T04:15:06.695522", "exception": false, "start_time": "2021-10-08T04:15:02.010574", "status": "completed"} tags=[]
plt.figure()
pl = sns.swarmplot(x=data["Clusters"],
                   y=data["Spent"],
                   color="#CBEDDD",
                   alpha=0.5)
pl = sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal)
plt.show()

# %% [markdown] papermill={"duration": 0.07193, "end_time": "2021-10-08T04:15:06.840213", "exception": false, "start_time": "2021-10-08T04:15:06.768283", "status": "completed"} tags=[]
#
# From the above plot, it can be clearly seen that cluster 1 is our biggest set of customers closely followed by cluster 0.
# We can explore what each cluster is spending on for the targeted marketing strategies.
#

# %% [markdown] papermill={"duration": 0.074689, "end_time": "2021-10-08T04:15:06.988145", "exception": false, "start_time": "2021-10-08T04:15:06.913456", "status": "completed"} tags=[]
# Let us next explore how did our campaigns do in the past.

# %% papermill={"duration": 0.373835, "end_time": "2021-10-08T04:15:07.435114", "exception": false, "start_time": "2021-10-08T04:15:07.061279", "status": "completed"} tags=[]
#Creating a feature to get a sum of accepted promotions
data["Total_Promos"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data[
    "AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=data["Total_Promos"], hue=data["Clusters"], palette=pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

# %% [markdown] papermill={"duration": 0.075057, "end_time": "2021-10-08T04:15:07.585345", "exception": false, "start_time": "2021-10-08T04:15:07.510288", "status": "completed"} tags=[]
# There has not been an overwhelming response to the campaigns so far. Very few participants overall. Moreover, no one part take in all 5 of them. Perhaps better-targeted and well-planned campaigns are required to boost sales.
#

# %% papermill={"duration": 0.325298, "end_time": "2021-10-08T04:15:07.983731", "exception": false, "start_time": "2021-10-08T04:15:07.658433", "status": "completed"} tags=[]
#Plotting the number of deals purchased
plt.figure()
pl = sns.boxenplot(y=data["NumDealsPurchases"],
                   x=data["Clusters"],
                   palette=pal)
pl.set_title("Number of Deals Purchased")
plt.show()

# %% [markdown] papermill={"duration": 0.075506, "end_time": "2021-10-08T04:15:08.136009", "exception": false, "start_time": "2021-10-08T04:15:08.060503", "status": "completed"} tags=[]
# Unlike campaigns, the deals offered did well. It has best outcome with cluster 0 and cluster 3.
# However, our star customers cluster 1 are not much into the deals.
# Nothing seems to attract cluster 2 overwhelmingly
#

# %% _kg_hide-input=true _kg_hide-output=true papermill={"duration": 3.296251, "end_time": "2021-10-08T04:15:11.506721", "exception": false, "start_time": "2021-10-08T04:15:08.210470", "status": "completed"} tags=[]
#for more details on the purchasing style
Places = [
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth"
]

for i in Places:
    plt.figure()
    sns.jointplot(x=data[i],
                  y=data["Spent"],
                  hue=data["Clusters"],
                  palette=pal)
    plt.show()

# %% [markdown] papermill={"duration": 0.083796, "end_time": "2021-10-08T04:15:11.674202", "exception": false, "start_time": "2021-10-08T04:15:11.590406", "status": "completed"} tags=[]
# <a id="8"></a>
# ## PROFILING
#
# Now that we have formed the clusters and looked at their purchasing habits.
# Let us see who all are there in these clusters. For that, we will be profiling the clusters formed and come to a conclusion about who is our star customer and who needs more attention from the retail store's marketing team.
#
# To decide that I will be plotting some of the features that are indicative of the customer's personal traits in light of the cluster they are in.
# On the basis of the outcomes, I will be arriving at the conclusions.

# %% papermill={"duration": 25.901281, "end_time": "2021-10-08T04:15:37.659466", "exception": false, "start_time": "2021-10-08T04:15:11.758185", "status": "completed"} tags=[]
Personal = [
    "Kidhome", "Teenhome", "Customer_For", "Age", "Children", "Family_Size",
    "Is_Parent", "Education", "Living_With"
]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data[i],
                  y=data["Spent"],
                  hue=data["Clusters"],
                  kind="kde",
                  palette=pal)
    plt.show()

# %% [markdown] papermill={"duration": 0.10691, "end_time": "2021-10-08T04:15:37.873431", "exception": false, "start_time": "2021-10-08T04:15:37.766521", "status": "completed"} tags=[]
# **Points to be noted:**
#
# The following information can be deduced about the customers in different clusters.
#
# <img src="https://github.com/KarnikaKapoor/Files/blob/main/Colorful%20Handwritten%20About%20Me%20Blank%20Education%20Presentation%20(3).png?raw=true">
#

# %% [markdown] papermill={"duration": 0.106499, "end_time": "2021-10-08T04:15:38.087648", "exception": false, "start_time": "2021-10-08T04:15:37.981149", "status": "completed"} tags=[]
# <a id="9"></a>
# ## CONCLUSION
#
# In this project, I performed unsupervised clustering.
# I did use dimensionality reduction followed by agglomerative clustering.
# I came up with 4 clusters and further used them in profiling customers in clusters according to their family structures and income/spending.
# This can be used in planning better marketing strategies.
#
# **<span style="color:#682F2F;"> If you liked this Notebook, please do upvote.</span>**
#
# **<span style="color:#682F2F;">If you have any questions, feel free to comment!</span>**
#
# **<span style="color:#682F2F;"> Best Wishes!</span>**
#
# <a id="10"></a>
# ## END
