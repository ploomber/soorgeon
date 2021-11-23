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

# %% [markdown]
# # FEATURE SELECTION & DATA VISUALIZATION

# %% [markdown] _cell_guid="b72f3424-c0c2-47e8-b18b-bc89253eab3f" _execution_state="idle" _uuid="bbc2abb85c63b8c357f5c2e0140105ed195f7fb4"
# ## What will you learn from this project?
# ![deasd.png](attachment:2834b410-2f8d-4dc3-b461-cfa22817f85b.png)

# %% [markdown]
# ## Introduction
# In this data analysis report, I usually focus on feature visualization and selection as a different from other kernels. Feature selection with correlation, univariate feature selection, recursive feature elimination, recursive feature elimination with cross validation and tree based feature selection methods are used with random forest classification. Apart from these, principle component analysis are used to observe number of components.
#
# **Enjoy your data analysis!!!**
#
# ![Cancer_Awareness-2019_blog_v1.0_noText.jpg](attachment:6511de54-b562-4594-bcf9-a12222fbef97.jpg)

# %% [markdown] _cell_guid="8d7fdae8-1d31-4873-a021-d553e2c4087c" _execution_state="idle" _uuid="c28fa7775a99901a882aee31e890ea99fe796d91"
# ## Data Analysis Content
# 1. [Python Libraries](#1)
# 1. [Data Content](#2)
# 1. [Read and Analyse Data](#3)
# 1. [Visualization](#4)
# 1. [Feature Selection and Random Forest Classification](#5)
#     1. [Feature selection with correlation and random forest classification](#6)
#     1. [Univariate feature selection and random forest classification](#7)
#     1. [Recursive feature elimination (RFE) with random forest](#8)
#     1. [Recursive feature elimination with cross validation and random forest classification](#9)
#     1. [Tree based feature selection and random forest classification](#10)
# 1. [Feature Extraction with PCA](#11)
# 1. [Conclusion](#12)

# %% [markdown]
# <a id='1'></a>
# ## Python Libraries
# * In this section, we import used libraries during this kernel.

# %% _cell_guid="52942f7b-e58d-4275-86f0-ced1bcea06f9" _execution_state="idle" _uuid="d7dc365d2933b6675c57c98d438356e4cc1e6125"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualization library
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
from subprocess import check_output

print(check_output(["ls", "input"]).decode("utf8"))
#import warnings library
import warnings
# ignore all warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.

# %% [markdown]
# <a id='2'></a>
# ## Data Content
# 1. **ID number**
# 1. **Diagnosis (M = malignant, B = benign)**
# 1. **radius (mean of distances from center to points on the perimeter)**
# 1. **texture (standard deviation of gray-scale values)**
# 1. **perimeter**
# 1. **area**
# 1. **smoothness (local variation in radius lengths)**
# 1. **compactness (perimeter^2 / area - 1.0)**
# 1. **concavity (severity of concave portions of the contour)**
# 1. **concave points (number of concave portions of the contour)**
# 1. **symmetry**
# 1. **fractal dimension ("coastline approximation" - 1)**
# <br>
# * The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# * All feature values are recoded with four significant digits.
# * Missing attribute values: none
# * Class distribution: 357 benign, 212 malignant

# %% [markdown]
# <a id='3'></a>
# ## Read and Analyse Data

# %% _cell_guid="c9bd4680-5a5d-4ce5-8b85-1820d2e478d0" _execution_state="idle" _uuid="4a65810840012b075b5a359994931bec8acf9ab0"
data = pd.read_csv('input/data.csv')

# %% [markdown] _cell_guid="48131d2a-cb21-4c1f-8213-f0e78647287c" _execution_state="idle" _uuid="81d7851cbda2bc774e989259005e98999b84c0b2"
# Before making anything like feature selection,feature extraction and classification, firstly we start with basic data analysis.
# Lets look at features of data.

# %% _cell_guid="d30f1486-bb97-40d7-9125-67e6f15286dc" _execution_state="idle" _uuid="3e01972c830afa1ce55025c0b7a202d4b204dd1d"
data.head()  # head method show only first 5 rows

# %% [markdown] _cell_guid="d3b8329c-20eb-4f00-b900-ffc9278cd82a" _execution_state="idle" _uuid="16fbbd5b380761a5e22478133c1f8ead52ca7abb"
# **There are 4 things that take my attention**
# 1) There is an **id** that cannot be used for classificaiton
# 2) **Diagnosis** is our class label
# 3) **Unnamed: 32** feature includes NaN so we do not need it.
# 4) I do not have any idea about other feature names actually I do not need because machine learning is awesome **:)**
#
# Therefore, drop these unnecessary features. However do not forget this is not a feature selection. This is like a browse a pub, we do not choose our drink yet !!!

# %% _cell_guid="60308baf-344a-41fb-8580-cef707ce5aa8" _execution_state="idle" _uuid="54600377cdbec016505dcb970bb1988afbc260a2"
# feature names as a list
col = data.columns  # .columns gives columns names in data
print(col)

# %% _cell_guid="8764b4cf-9963-4c1a-b449-059de8153e4c" _execution_state="idle" _uuid="94ea75618315ac7af54cf80a501c42b40e77ecbc"
# y includes our labels and x includes our features
y = data.diagnosis  # M or B
list = ['Unnamed: 32', 'id', 'diagnosis']
x = data.drop(list, axis=1)
x.head()

# %% _cell_guid="31ec8d06-ea25-4b34-84ca-c322b3d8a10f" _execution_state="idle" _uuid="71fecf26e957a2d670182d607aca5a7b92b4a3b6"
ax = sns.countplot(y, label="Count")  # M = 212, B = 357
B, M = y.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant : ', M)

# %% [markdown] _cell_guid="8c7d5b2b-445c-45ec-81da-5b3853617711" _execution_state="idle" _uuid="af412eff1a8fa83f1f9fb6daaeac78cf498d589a"
# Okey, now we have features but **what does they mean** or actually **how much do we need to know about these features**
# The answer is that we do not need to know meaning of these features however in order to imagine in our mind we should know something like variance, standart deviation, number of sample (count) or max min values.
# These type of information helps to understand about what is going on data. For example , the question is appeared in my mind the **area_mean** feature's max value is 2500 and **smoothness_mean** features' max 0.16340. Therefore **do we need standardization or normalization before visualization, feature selection, feature extraction or classificaiton?** The answer is yes and no not surprising ha :) Anyway lets go step by step and start with visualization.

# %% _cell_guid="c92292c2-d999-42f3-8618-73b231c163e6" _execution_state="idle" _uuid="7d909ed445dd83306413a72986cebc17d1814cc7"
x.describe()

# %% [markdown] _cell_guid="6179a010-0819-481e-8095-72ba1021fdcd" _execution_state="idle" _uuid="3edac4b24f82f00d32efe9d812aed40fb06fdbed"
# <a id='4'></a>
# ## Visualization
# In order to visualizate data we are going to use seaborn plots that is not used in other kernels to inform you and for diversity of plots. What I use in real life is mostly violin plot and swarm plot. Do not forget we are not selecting feature, we are trying to know data like looking at the drink list at the pub door.

# %% [markdown] _cell_guid="fa50c6cf-ccb6-49fa-b4bb-e7f14a3c4a09" _execution_state="idle" _uuid="d3d8066df83b9a30087610eed09782c1dec7c4cf"
# Before violin and swarm plot we need to normalization or standirdization. Because differences between values of features are very high to observe on plot. I plot features in 3 group and each group includes 10 features to observe better.

# %% _cell_guid="d58052d6-9e8c-46f4-a2d5-b9e82b247f27" _execution_state="idle" _uuid="d640909614b5ff561e35b33e555458df70b22486"
# first ten features
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())  # standardization
data = pd.concat([y, data_n_2.iloc[:, 0:10]], axis=1)
data = pd.melt(data,
               id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.violinplot(x="features",
               y="value",
               hue="diagnosis",
               data=data,
               split=True,
               inner="quart")
plt.xticks(rotation=90)

# %% [markdown] _cell_guid="8eb078f8-63a8-405d-8cdb-10c9e8a06863" _execution_state="idle" _uuid="6c3ce53008a590504b1afef35e5edff183ed0ce4"
# Lets interpret the plot above together. For example, in **texture_mean** feature, median of the *Malignant* and *Benign* looks like separated so it can be good for classification. However, in **fractal_dimension_mean** feature,  median of the *Malignant* and *Benign* does not looks like separated so it does not gives good information for classification.

# %% _cell_guid="46ee71d3-93c1-4c2d-bb00-6995f7a1c816" _execution_state="idle" _uuid="0a18301387ce26b58a68e5a2d340b39e86c1f5e0"
# Second ten features
data = pd.concat([y, data_n_2.iloc[:, 10:20]], axis=1)
data = pd.melt(data,
               id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.violinplot(x="features",
               y="value",
               hue="diagnosis",
               data=data,
               split=True,
               inner="quart")
plt.xticks(rotation=90)

# %% _cell_guid="58f17ef1-6530-4db8-bcd9-32691363e8a9" _execution_state="idle" _uuid="d1c4e84c3d6bda4b9ff9e284d8c790dd46980c31"
# Second ten features
data = pd.concat([y, data_n_2.iloc[:, 20:31]], axis=1)
data = pd.melt(data,
               id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.violinplot(x="features",
               y="value",
               hue="diagnosis",
               data=data,
               split=True,
               inner="quart")
plt.xticks(rotation=90)

# %% _uuid="54c8d1570b21a65ea707c21440d90f6701b15e2b"
# As an alternative of violin plot, box plot can be used
# box plots are also useful in terms of seeing outliers
# I do not visualize all features with box plot
# In order to show you lets have an example of box plot
# If you want, you can visualize other features as well.
plt.figure(figsize=(10, 10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)

# %% [markdown] _cell_guid="7e3412af-5856-4cb1-922d-6a3dd6c5f238" _execution_state="idle" _uuid="80d14a320a514d9e727135db6d64155bea3ca35b"
# Lets interpret one more thing about plot above, variable of **concavity_worst** and **concave point_worst** looks like similar but how can we decide whether they are correlated with each other or not.
# (Not always true but, basically if the features are correlated with each other we can drop one of them)

# %% [markdown] _cell_guid="d6282b97-002d-4aeb-a09a-ce4ef138a1ca" _execution_state="idle" _uuid="f76a463fada43aa587f7bd035148698e71db6309"
# In order to compare two features deeper, lets use joint plot. Look at this in joint plot below, it is really correlated.
#  Pearsonr value is correlation value and 1 is the highest. Therefore, 0.86 is looks enough to say that they are correlated.
# Do not forget, we are not choosing features yet, we are just looking to have an idea about them.

# %% _cell_guid="47880bbb-5dbe-4836-938c-0816a03e8fb4" _execution_state="idle" _uuid="859ec665af4be178c3e36b1c2799f44c5ccef901"
sns.jointplot(x.loc[:, 'concavity_worst'],
              x.loc[:, 'concave points_worst'],
              kind="reg",
              color="#ce1414")

# %% [markdown] _cell_guid="008330ba-393e-4d31-b086-b90332a613c5" _execution_state="idle" _uuid="43de55ee2437101c0450d989216a426d6f1f30c7"
# What about three or more feauture comparision ? For this purpose we can use pair grid plot. Also it seems very cool :)
# And we discover one more thing **radius_worst**, **perimeter_worst** and **area_worst** are correlated as it can be seen pair grid plot. We definetely use these discoveries for feature selection.

# %% _cell_guid="3bda33fe-daf9-4f74-acbc-d9d3c8fc83d9" _execution_state="idle" _uuid="381ecb55ced22383c96320ced2299f5da37ce4b6"
sns.set(style="white")
df = x.loc[:, ['radius_worst', 'perimeter_worst', 'area_worst']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)

# %% [markdown] _cell_guid="c9ef0921-19bc-4d40-aa72-0bc2333bd4b4" _execution_state="idle" _uuid="f8355f83ac16e414e3b88e67b77d33ef31c3574d"
# Up to this point, we make some comments and discoveries on data already. If you like what we did, I am sure swarm plot will open the pub's door :)

# %% [markdown] _cell_guid="03abd05a-d67a-4bfc-b951-6394de8c6fc9" _execution_state="idle" _uuid="c3807ef7f6e17b33ae383349bdee7ebfced2a847"
# In swarm plot, I will do three part like violin plot not to make plot very complex appearance

# %% _cell_guid="ef378d49-8aed-4b9e-96e8-e7d2458fdd89" _execution_state="idle" _uuid="85a2413b70c1b3d69f26a2c122c22d55f930e774"
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())  # standardization
data = pd.concat([y, data_n_2.iloc[:, 0:10]], axis=1)
data = pd.melt(data,
               id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)

# %% _cell_guid="428c75b6-b5d0-47e3-a568-17b5d1896c0c" _execution_state="idle" _uuid="75dfd5e9e50adceb1d42dd000ce779e79b069cce"
data = pd.concat([y, data_n_2.iloc[:, 10:20]], axis=1)
data = pd.melt(data,
               id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)

# %% _cell_guid="ee64bbc8-0431-482a-b08f-cdca43e41390" _execution_state="idle" _uuid="209e9e9120d6e889696d2d1190e663b5c1885a82"
data = pd.concat([y, data_n_2.iloc[:, 20:31]], axis=1)
data = pd.melt(data,
               id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
toc = time.time()
plt.xticks(rotation=90)
print("swarm plot time: ", toc - tic, " s")

# %% [markdown] _cell_guid="5c9efa5e-4938-477e-ab9a-4b084cc0b870" _execution_state="idle" _uuid="d23e2f9b040a92feb0d7ceb8e01e74c758f5dbc3"
# They looks cool right. And you can see variance more clear. Let me ask you a question, **in these three plots which feature looks like more clear in terms of classification.** In my opinion **area_worst** in last swarm plot looks like malignant and benign are seprated not totaly but mostly. Hovewer, **smoothness_se** in swarm plot 2 looks like malignant and benign are mixed so it is hard to classfy while using this feature.

# %% [markdown] _cell_guid="c4c68f34-e876-4e5a-a4a7-09c07381425a" _execution_state="idle" _uuid="b46f98eb7ca8d36dc7bf1516895599524bab694d"
# **What if we want to observe all correlation between features?** Yes, you are right. The answer is heatmap that is old but powerful plot method.

# %% _cell_guid="9e1e7d8a-bbf2-4aab-90e7-78d4c4ccf416" _execution_state="idle" _uuid="0eeb70ddffc8ac332ee076f2f6b2833a6ffddd2d"
#correlation map
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

# %% [markdown] _cell_guid="8ee2e02b-9fc6-42f7-8a83-f6dd77df6c11" _execution_state="idle" _uuid="b4e3afecb204a8262330d22e4539554b2af7975a"
# Well, finaly we are in the pub and lets choose our drinks at feature selection part while using heatmap(correlation matrix).

# %% [markdown] _cell_guid="6786734a-40a9-46b6-a13a-97ee9c569636" _execution_state="idle" _uuid="84b145dd0c13a3a0d4dd4f9b9fd1bd782e11fcf8"
# <a id='5'></a>
# ## Feature Selection and Random Forest Classification
# Today our purpuse is to try new cocktails. For example, we are finaly in the pub and we want to drink different tastes. Therefore, we need to compare ingredients of drinks. If one of them includes lemon, after drinking it we need to eliminate other drinks which includes lemon so as to experience very different tastes.

# %% [markdown] _cell_guid="c7b2df4e-270e-4c94-8789-177f5e90ac46" _execution_state="idle" _uuid="a042df90ef7138d6f101463e93936119176bdc0d"
# In this part we will select feature with different methods that are feature selection with correlation, univariate feature selection, recursive feature elimination (RFE), recursive feature elimination with cross validation (RFECV) and tree based feature selection. We will use random forest classification in order to train our model and predict.

# %% [markdown] _cell_guid="94d217e3-b2b3-4016-b72e-f8d521d17af7" _execution_state="idle" _uuid="39003c7b75f265bf0826f407433e65923c4dd017"
# <a id='6'></a>
# ### 1) Feature selection with correlation and random forest classification

# %% [markdown] _cell_guid="785bd27a-30d9-4e08-a864-cde7e5630aad" _execution_state="idle" _uuid="1e6ef08c98cb4bf0dedf275e4c08fae743bb3801"
# As it can be seen in map heat figure **radius_mean, perimeter_mean and area_mean** are correlated with each other so we will use only **area_mean**. If you ask how i choose **area_mean** as a feature to use, well actually there is no correct answer, I just look at swarm plots and **area_mean** looks like clear for me but we cannot make exact separation among other correlated features without trying. So lets find other correlated features and look accuracy with random forest classifier.

# %% [markdown] _cell_guid="eea2971b-b703-4e1b-b048-128501506f33" _execution_state="idle" _uuid="acde9c0b406d72122473f8292d641a9fcb8a8682"
# **Compactness_mean, concavity_mean and concave points_mean** are correlated with each other.Therefore I only choose **concavity_mean**. Apart from these, **radius_se, perimeter_se and area_se** are correlated and I only use **area_se**.  **radius_worst, perimeter_worst and area_worst** are correlated so I use **area_worst**.  **Compactness_worst, concavity_worst and concave points_worst** so I use **concavity_worst**.  **Compactness_se, concavity_se and concave points_se** so I use **concavity_se**. **texture_mean and texture_worst are correlated** and I use **texture_mean**. **area_worst and area_mean** are correlated, I use **area_mean**.
#
#
#

# %% _cell_guid="ef8d06df-bfcc-4e9a-a3ba-5016ec0c5bd5" _execution_state="idle" _uuid="117f3e858e806f3f26a68dadf3fc89d471010156"
drop_list1 = [
    'perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean',
    'radius_se', 'perimeter_se', 'radius_worst', 'perimeter_worst',
    'compactness_worst', 'concave points_worst', 'compactness_se',
    'concave points_se', 'texture_worst', 'area_worst'
]
x_1 = x.drop(drop_list1, axis=1)  # do not modify x, we will use it later
x_1.head()

# %% [markdown] _cell_guid="6de99062-7a5a-4b70-879c-54d6c8a4a7e2" _execution_state="idle" _uuid="1ab3852ed7fbeba8718e6722e8a40521033bdf29"
# After drop correlated features, as it can be seen in below correlation matrix, there are no more correlated features. Actually, I know and you see there is correlation value 0.9 but lets see together what happen if we do not drop it.

# %% _cell_guid="733f0784-4a3f-410c-a220-f98591825f2e" _execution_state="idle" _uuid="eec5424039036e1af43ba0795b76393805308f97"
#correlation map
f, ax = plt.subplots(figsize=(14, 14))
sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

# %% [markdown] _cell_guid="f6551fe9-f3a8-4738-ace3-b0427be4aeb4" _execution_state="idle" _uuid="0eaf4ae8e33c2d6352a862953c1fe5eecf46ed27"
# Well, we choose our features but **did we choose correctly ?** Lets use random forest and find accuracy according to chosen features.

# %% _cell_guid="111af932-96f8-4105-8deb-ba1172edd203" _execution_state="idle" _uuid="c7a6af60a44959f81593d788934a49c9259d8b43"
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x_1,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)
clr_rf = clf_rf.fit(x_train, y_train)

ac = accuracy_score(y_test, clf_rf.predict(x_test))
print('Accuracy is: ', ac)
cm = confusion_matrix(y_test, clf_rf.predict(x_test))
sns.heatmap(cm, annot=True, fmt="d")

# %% [markdown] _cell_guid="1503384d-ca2b-4b52-82b5-f1131b014269" _execution_state="idle" _uuid="21cda299619940f7b22acc9a804ee56bff71d3e7"
# Accuracy is almost 95% and as it can be seen in confusion matrix, we make few wrong prediction.
# Now lets see other feature selection methods to find better results.

# %% [markdown] _cell_guid="3eed9ac3-e601-4e16-85bc-26a1a6fff850" _execution_state="idle" _uuid="decd86422aee506b061c905e8573abb3612734e4"
# <a id='7'></a>
# ### 2) Univariate feature selection and random forest classification
# In univariate feature selection, we will use SelectKBest that removes all but the k highest scoring features.
# <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest>

# %% [markdown] _cell_guid="f053659d-9dfe-4858-a220-ef327df3bc36" _execution_state="idle" _uuid="f681583a7b20e4fb86e557b910021a263573cf18"
# In this method we need to choose how many features we will use. For example, will k (number of features) be 5 or 10 or 15? The answer is only trying or intuitively. I do not try all combinations but I only choose k = 5 and find best 5 features.

# %% _cell_guid="4f43c8bd-48f7-4ed9-aa2d-6aa8a29c0c58" _execution_state="idle" _uuid="8159f9efb106f1219dc4e8c2a340399b88f224d8" jupyter={"outputs_hidden": true}
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# find best scored 5 features
select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)

# %% _cell_guid="c9684618-06fe-4b0a-835f-ceea46da397c" _execution_state="idle" _uuid="d9dcd1495cbf33c190a0d1211df4bac5e79bc4e5"
print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)

# %% [markdown] _cell_guid="b0c426ef-3072-4f6e-bf24-b5d927a98316" _execution_state="idle" _uuid="0f46ea1a9b1282a377549406d1c5e093380954b6"
# Best 5 feature to classify is that **area_mean, area_se, texture_mean, concavity_worst and concavity_mean**. So lets se what happens if we use only these best scored 5 feature.

# %% _cell_guid="efc70e04-bc9c-4f93-bcd3-b1d7160d0d5c" _execution_state="idle" _uuid="9a2bd21537f7c600f3c9baaf833c001084d6ba00"
x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()
clr_rf_2 = clf_rf_2.fit(x_train_2, y_train)
ac_2 = accuracy_score(y_test, clf_rf_2.predict(x_test_2))
print('Accuracy is: ', ac_2)
cm_2 = confusion_matrix(y_test, clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2, annot=True, fmt="d")

# %% [markdown] _cell_guid="d8888dc1-b50b-46b4-b202-4e33c2630406" _execution_state="idle" _uuid="575005da62c41d12bbb3999b3e26148e12930ce3"
# Accuracy is almost 96% and as it can be seen in confusion matrix, we make few wrong prediction. What we did up to now is that we choose features according to correlation matrix and according to selectkBest method. Although we use 5 features in selectkBest method accuracies look similar.
# Now lets see other feature selection methods to find better results.

# %% [markdown] _cell_guid="702ad2b3-5b12-4d15-93b1-e7d62dfd1040" _execution_state="idle" _uuid="7a3c3050dd9d694e52962c7c712b1ea16aab6fdf"
# <a id='8'></a>
# ### 3) Recursive feature elimination (RFE) with random forest
# <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>
# Basically, it uses one of the classification methods (random forest in our example), assign weights to each of features. Whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features

# %% [markdown] _cell_guid="8a34a801-c568-4598-8a07-85fc20ad0386" _execution_state="idle" _uuid="3ea45c46bb231c767160fe13ad3b21a70f0d0375"
# Like previous method, we will use 5 features. However, which 5 features will we use ? We will choose them with RFE method.

# %% _cell_guid="8df88bb5-8003-4696-9efe-63ebf8d609a5" _execution_state="idle" _uuid="c384a5240d1c1e9e2a6750e5d218dadaf24d2035"
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(x_train, y_train)

# %% _cell_guid="51d63d0b-4e00-4dc1-816c-809287b60806" _execution_state="idle" _uuid="29ba35a98954d0ae686ce46295179d1f1a27b74c"
print('Chosen best 5 feature by rfe:', x_train.columns[rfe.support_])

# %% [markdown] _cell_guid="92aa6013-3e16-4005-ab1b-b7ce53e78bd3" _execution_state="idle" _uuid="ce670f778a661e8ddc3b7b21a43ccb48a551581a"
# Chosen 5 best features by rfe is **texture_mean, area_mean, concavity_mean, area_se, concavity_worst**. They are exactly similar with previous (selectkBest) method. Therefore we do not need to calculate accuracy again. Shortly, we can say that we make good feature selection with rfe and selectkBest methods. However as you can see there is a problem, okey I except we find best 5 feature with two different method and these features are same but why it is **5**. Maybe if we use best 2 or best 15 feature we will have better accuracy. Therefore lets see how many feature we need to use with rfecv method.

# %% [markdown] _cell_guid="22a4f840-2a37-4047-9804-129e7f68f74a" _execution_state="idle" _uuid="42a8c3f2ef0e5978b620eea737e6e234dc79cfe8"
# <a id='9'></a>
# ### 4) Recursive feature elimination with cross validation and random forest classification
# <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html>
# Now we will not only **find best features** but we also find **how many features do we need** for best accuracy.

# %% _cell_guid="7a5d4d69-7734-4465-89cc-f46b4af4c548" _execution_state="idle" _uuid="0d7803966979745a8bdbdbc44a1927558485640a"
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,
              scoring='accuracy')  #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])

# %% [markdown] _cell_guid="be0f7ce5-55c0-48c9-a125-616750f13943" _execution_state="idle" _uuid="cef5972f5e9fa830e92bae00ca5dd7d2b0ac8c58"
# Finally, we find best 11 features that are **texture_mean, area_mean, concavity_mean, texture_se, area_se, concavity_se, symmetry_se, smoothness_worst, concavity_worst, symmetry_worst and fractal_dimension_worst** for best classification. Lets look at best accuracy with plot.
#

# %% _cell_guid="5b69144b-72e4-4ac3-b8a8-c9ebbf8ffa3b" _execution_state="idle" _uuid="f362bfa341032f2bb1bacc1c50675a1916f5c536"
# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# %% [markdown] _cell_guid="580f19b6-1182-43d5-932c-21e2fb5bb2d9" _execution_state="idle" _uuid="f071ec67bd63d5c458c2cb6303fe6f54458db57b"
# Lets look at what we did up to this point. Lets accept that guys this data is very easy to classification. However, our first purpose is actually not finding good accuracy. Our purpose is learning how to make **feature selection and understanding data.** Then last make our last feature selection method.

# %% [markdown] _cell_guid="2637e8bc-d986-41c0-acef-ce76afc4c350" _execution_state="idle" _uuid="8bc3105398fc618e19deec4de957950cfb45c054"
# <a id='10'></a>
# ### 5) Tree based feature selection and random forest classification
# <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
# In random forest classification method there is a **feature_importances_** attributes that is the feature importances (the higher, the more important the feature). **!!! To use feature_importance method, in training data there should not be correlated features. Random forest choose randomly at each iteration, therefore sequence of feature importance list can change.**
#

# %% _cell_guid="df8abc8d-3279-4c31-a6b6-e4f272ca0b47" _execution_state="idle" _uuid="31d4b248f723930ff7120ffaff2c260f07e3f0fc"
clf_rf_5 = RandomForestClassifier()
clr_rf_5 = clf_rf_5.fit(x_train, y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]),
        importances[indices],
        color="g",
        yerr=std[indices],
        align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()

# %% [markdown] _cell_guid="00008678-9012-4e50-aaab-010b3353ac98" _execution_state="idle" _uuid="760b045b33388f6fb7b53acdf931e8204eea80cd"
# As you can seen in plot above, after 5 best features importance of features decrease. Therefore we can focus these 5 features. As I sad before, I give importance to understand features and find best of them.

# %% [markdown] _cell_guid="21ef3e97-1eba-4f30-9714-cc45a3c1a594" _execution_state="idle" _uuid="c3fd1de4be5be26252a4105501a217a026d116b1"
# <a id='11'></a>
# ## Feature Extraction with PCA
# <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>
# We will use principle component analysis (PCA) for feature extraction. Before PCA, we need to normalize data for better performance of PCA.
#

# %% _cell_guid="aa440ca8-8282-4cc3-9683-d5ebdd992140" _execution_state="idle" _uuid="cf72d82fea5d8330db8ec324fb18abc6e969bac6"
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)
#normalization
x_train_N = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
x_test_N = (x_test - x_test.mean()) / (x_test.max() - x_test.min())

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(x_train_N)

plt.figure(1, figsize=(14, 13))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')

# %% [markdown] _cell_guid="264e1dba-1f40-4a28-bd4b-5261a7df185b" _execution_state="idle" _uuid="a50cb8de4ec6e2e8727a2b3b021ee42c4ce21b29"
# * According to variance ration, 3 component can be chosen.
# * If you have any doubt about PCA, you can check my intuitive way of PCA tutorial.

# %% [markdown] _cell_guid="224c9c75-256d-4650-af3e-c4c39b565661" _execution_state="idle" _uuid="5e0f4ba2b385fe9312ced2455eed4fb87d39a0b8"
# <a id='12'></a>
# # Conclusion
# Shortly, I tried to show importance of feature selection and data visualization.
# Default data includes 33 feature but after feature selection we drop this number from 33 to 5 with accuracy 95%. In this kernel we just tried basic things, I am sure with these data visualization and feature selection methods, you can easily ecxeed the % 95 accuracy. Maybe you can use other classification methods.
# ### I hope you enjoy in this kernel
# ## If you have any question or advise, I will be apreciate to listen them ...
