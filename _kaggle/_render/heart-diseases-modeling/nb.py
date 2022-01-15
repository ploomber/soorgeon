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

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
models = [
    LogisticRegression, KNeighborsClassifier, SVC, MLPClassifier,
    DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier,
    XGBClassifier, LGBMClassifier
]  #,CatBoostClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# %% [markdown]
# ## Adding Functions


# %%
def degisken_tiplerine_ayirma(data, cat_th, car_th):
    """
   Veri:data parametresi ili fonksiyona girilen verinin değişkenlerin sınıflandırılması.
   Parameters
   ----------
   data: pandas.DataFrame
   İşlem yapılacak veri seti

   cat_th:int
   categoric değişken threshold değeri

   car_th:int
   Cardinal değişkenler için threshold değeri

   Returns
   -------
    cat_deg:list
    categorik değişken listesi
    num_deg:list
    numeric değişken listesi
    car_deg:list
    categoric ama cardinal değişken listesi

   Examples
   -------
    df = dataset_yukle("breast_cancer")
    cat,num,car=degisken_tiplerine_ayirma(df,10,20)
   Notes
   -------
    cat_deg + num_deg + car_deg = toplam değişken sayısı

   """

    num_but_cat = [
        i for i in data.columns
        if data[i].dtypes != "O" and data[i].nunique() < cat_th
    ]

    car_deg = [
        i for i in data.columns
        if data[i].dtypes == "O" and data[i].nunique() > car_th
    ]

    num_deg = [
        i for i in data.columns
        if data[i].dtypes != "O" and i not in num_but_cat
    ]

    cat_deg = [
        i for i in data.columns if data[i].dtypes == "O" and i not in car_deg
    ]

    cat_deg = cat_deg + num_but_cat

    print(f"Dataset kolon/değişken sayısı: {data.shape[1]}")
    print(f"Dataset satır/veri sayısı: {data.shape[0]}")
    print("********************************************")
    print(f"Datasetin numeric değişken sayısı: {len(num_deg)}")
    print(f"Datasetin numeric değişkenler: {num_deg}")
    print("********************************************")
    print(f"Datasetin categoric değişken sayısı: {len(cat_deg)}")
    print(f"Datasetin categoric değişkenler: {cat_deg}")
    print("********************************************")
    print(f"Datasetin cardinal değişken sayısı: {len(car_deg)}")
    print(f"Datasetin cardinal değişkenler: {car_deg}")
    print("********************************************")

    return cat_deg, num_deg, car_deg


def categoric_ozet(data, degisken, plot=False, null_control=False):
    """
    Task
    ----------
    Datasetinde bulunan categoric değişkenlerin değişken tiplerinin sayısını ve totale karşı oranını bulur.
    Ayrıca isteğe bağlı olarak değişken dağılımının grafiğini ve değişken içinde bulunan null sayısını çıkartır.

    Parameters
    ----------
    data:pandas.DataFrame
    categoric değişkenin bulunduğu dataset.
    degisken:String
    Categoric değişken ismi.
    plot:bool
    Fonksiyonda categoric değişken dağılımının grafiğini çizdirmek için opsiyonel özellik.
    null_control:bool
    Fonksiyonda değişken içinde null değer kontolü için opsiyonel özellik

    Returns
    -------
    tablo:pandas.DataFrame
    Unique değişkenlerin ratio olarak oran tablosu
    Examples
    -------
    df=dataset_yukle("titanic")
    cat_deg,num_deg,car_deg=degisken_tiplerine_ayirma(df,10,20)
    for i in cat_deg:
        tablo=categoric_ozet(df,i,True,True)
    """

    print(
        pd.DataFrame({
            degisken: data[degisken].value_counts(),
            "Ratio": 100 * data[degisken].value_counts() / len(data)
        }))
    tablo = pd.DataFrame({
        degisken:
        data[degisken].value_counts(),
        "Ratio":
        100 * data[degisken].value_counts() / len(data)
    })
    print("##########################################")
    if plot:
        sns.countplot(x=data[degisken], data=data)
        plt.show(block=True)
    if null_control:
        print(f"Null veri sayısı: {data[degisken].isnull().sum()}")

    return tablo


def dataset_ozet(data, head=5):
    print("##################### Shape #####################")
    print(f"Satır sayısı: {data.shape[0]}")
    print(f"Kolon sayısı: {data.shape[1]}")

    print("##################### Types #####################")
    print(data.dtypes)

    print("##################### Head #####################")
    print(data.head(head))

    print("##################### Tail #####################")
    print(data.tail(head))

    print("##################### NA Kontrolü #####################")
    print(data.isnull().sum())

    print("##################### Quantiles #####################")
    print(data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("##################### Describe Tablosu #####################")
    print(data.describe().T)


def outlier_threshold(data, degisken):
    Q1 = data[degisken].quantile(0.01)
    Q3 = data[degisken].quantile(0.99)
    Q_Inter_Range = Q3 - Q1
    alt_limit = Q1 - 1.5 * Q_Inter_Range
    ust_limit = Q3 + 1.5 * Q_Inter_Range
    return alt_limit, ust_limit


def threshold_degisimi(data, degisken):
    alt_limit, ust_limit = outlier_threshold(data, degisken)
    data.loc[(data[degisken] < alt_limit), degisken] = alt_limit
    data.loc[(data[degisken] > ust_limit), degisken] = ust_limit
    #data[data[degisken]<alt_limit][degisken]=alt_limit
    #data[data[degisken]>ust_limit][degisken]=ust_limit
    return data


def numeric_ozet(data, degisken, plot=False, null_control=False):
    """
    Task
    ----------
    Datasetinde bulunan numeric değişkenlerin değişken tiplerinin sayısını ve totale karşı oranını bulur.
    Ayrıca isteğe bağlı olarak değişken dağılımının grafiğini ve değişken içinde bulunan null sayısını çıkartır.

    Parameters
    ----------
    data:pandas.DataFrame
    categoric değişkenin bulunduğu dataset.
    degisken:String
    Categoric değişken ismi.
    plot:bool
    Fonksiyonda categoric değişken dağılımının grafiğini çizdirmek için opsiyonel özellik.
    null_control:bool
    Fonksiyonda değişken içinde null değer kontolü için opsiyonel özellik

    Returns
    -------
    tablo:pandas.DataFrame
    Unique değişkenlerin ratio olarak oran tablosu
    Examples
    -------
    df=dataset_yukle("titanic")
    cat_deg,num_deg,car_deg=degisken_tiplerine_ayirma(df,10,20)
    for i in cat_deg:
        tablo=categoric_ozet(df,i,True,True)
    """
    quantiles = [
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99
    ]
    print(data[degisken].describe(quantiles).T)

    if plot:
        data[degisken].hist(bins=20)
        plt.xlabel(degisken)
        plt.title(degisken)
        plt.show(block=True)
    print("##########################################")

    if null_control:
        print(f"Null veri sayısı: {data[degisken].isnull().sum()}")


def missing_values_table(dataframe, na_name=False):
    na_columns = [
        col for col in dataframe.columns if dataframe[col].isnull().sum() > 0
    ]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] *
             100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)],
                           axis=1,
                           keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe,
                               columns=categorical_cols,
                               drop_first=drop_first)
    return dataframe


def model_karsilastirma(df, model, target):
    X = df.drop(columns=target)

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.15,
                                                        random_state=42)
    model_fit = model().fit(X_train, y_train)
    y_pred = model_fit.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(model, "için sonuç doğruluk değeri:", acc)
    return acc


def target_analyser(dataframe, target, num_deg, cat_deg):
    for degisken in dataframe.columns:
        if degisken in cat_deg:
            print(degisken, ":", len(dataframe[degisken].value_counts()))
            print(pd.DataFrame({
                "COUNT":
                dataframe[degisken].value_counts(),
                "RATIO":
                dataframe[degisken].value_counts() / len(dataframe),
                "TARGET_MEAN":
                dataframe.groupby(degisken)[target].mean()
            }),
                  end="\n\n\n")
        if degisken in num_deg:
            print(pd.DataFrame(
                {"TARGET_MEAN": dataframe.groupby(target)[degisken].mean()}),
                  end="\n\n\n")


# %% [markdown]
# ## Some image
# ![This is an image](https://www.sbbs-soc.com/wp-content/uploads/2020/09/Heart-Disease.jpg)

# %%
#loading dataset
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()

# %% [markdown]
# ## Some info
# * age: The person's age in years
# * sex: The person's sex (1 = male, 0 = female)
# * cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# * trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# * chol: The person's cholesterol measurement in mg/dl
# * fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# * restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# * thalach: The person's maximum heart rate achieved
# * exang: Exercise induced angina (1 = yes; 0 = no)
# * oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# * slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# * ca: The number of major vessels (0-3)
# * thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# * target: Heart disease (0 = no, 1 = yes)

# %%
#Analysis of Dataset
dataset_ozet(df)
cat_deg, num_deg, car_deg = degisken_tiplerine_ayirma(df, 10, 20)

# %%
#EDA of Dataset
for i in cat_deg:
    categoric_ozet(df, i, True, True)

for i in num_deg:
    numeric_ozet(df, i, True, True)

# %%
#All columns analaysis based on target column
target_analyser(df, "target", num_deg, cat_deg)

# %%
#Filling missing values
null_cols = missing_values_table(df, True)
for i in null_cols:
    df[i].fillna(df[i].transform("mean"), inplace=True)
#There is no missing values

# %%
#Outlier processing
for i in num_deg:
    df = threshold_degisimi(df, i)

# %%
#Data Extraction

df.age.describe()
df.loc[(df["age"] < 40), 'NEW_AGE_CAT'] = 'Young'
df.loc[(df["age"] >= 40) & (df["age"] < 50), 'NEW_AGE_CAT'] = 'Middle Age'
df.loc[(df["age"] >= 50) & (df["age"] < 60), 'NEW_AGE_CAT'] = 'Pre-Old'
df.loc[(df["age"] >= 60), 'NEW_AGE_CAT'] = 'Old'
df.groupby('NEW_AGE_CAT')["target"].mean()

# %%
df.trestbps.describe()
df.loc[(df["trestbps"] < 90), 'NEW_RBP_CAT'] = 'Low'
df.loc[(df["trestbps"] >= 90) & (df["trestbps"] < 120),
       'NEW_RBP_CAT'] = 'Ideal'
df.loc[(df["trestbps"] >= 120) & (df["trestbps"] < 140),
       'NEW_RBP_CAT'] = 'Pre-HIGH'
df.loc[(df["trestbps"] >= 140), 'NEW_RBP_CAT'] = 'Hypertension'
df.groupby('NEW_RBP_CAT')["target"].mean()

# %%
df.chol.describe()
df.loc[(df["chol"] < 200), 'NEW_CHOL_CAT'] = 'Ideal'
df.loc[(df["chol"] >= 200) & (df["chol"] < 240), 'NEW_CHOL_CAT'] = 'HIGH'
df.loc[(df["chol"] >= 240), 'NEW_CHOL_CAT'] = 'Very Risky'
df.groupby('NEW_CHOL_CAT')["target"].mean()

# %%
#Encoding of categoric columns
cat_deg, num_deg, car_deg = degisken_tiplerine_ayirma(df, 10, 20)
cat_deg = [i for i in cat_deg if i != "target"]
df = one_hot_encoder(df, cat_deg)
df.head()

# %%
#Scaling of numeric columns
scaler = StandardScaler()
df[num_deg] = scaler.fit_transform(df[num_deg])

# %%
#Comparing of all models
for mod in models:
    model_karsilastirma(df, mod, "target")

# %% [markdown]
# ## SVM Tuning

# %%
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15,
                                                    random_state=42)

svm = SVC()
svm_tuned = SVC(C=1, kernel="linear").fit(X_train, y_train)

y_pred = svm_tuned.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("SVM accuracy: ", acc)

# %% [markdown]
# ## Logistic Regression Tuning

# %%
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15,
                                                    random_state=42)

loj_model = LogisticRegression(solver="liblinear").fit(X_train, y_train)

y_pred = loj_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Lojistic_model accuracy: ", acc)

# %% [markdown]
# ## Light GBM Model Tuning

# %%
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15,
                                                    random_state=42)

lgbm_tuned = LGBMClassifier(learning_rate=0.01, max_depth=5,
                            n_estimators=250).fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("LGBM accuracy: ", acc)
