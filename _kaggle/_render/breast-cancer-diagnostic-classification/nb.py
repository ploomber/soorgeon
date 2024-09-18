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

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import os
import warnings

warnings.filterwarnings("ignore")
import time as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# %% [markdown]
# ## Data Preprocessing
# 1. Data Loading
# 2. Data Cleaning
# 3. X y split
# 4. Data Scaling


# %%
def data_load():  # check for the availability of the dataset and change cwd if not found
    df = pd.read_csv("../input/breast-cancer-prediction/data.csv")
    return df


def data_clean(df):
    return df


def X_y_split(df):
    X = df.drop(["diagnosis"], axis=1)
    y = df["diagnosis"]
    return X, y


def data_split_scale(X, y, sampling):
    # Splitting dataset into Train and Test Set
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.3)
    # Feature Scaling using Standardization
    ss = StandardScaler()
    X_tr = ss.fit_transform(X_tr)
    X_test = ss.fit_transform(X_test)
    print(
        "'For 'Sampling strategies', I have 3 options. \n \t'1' stands for 'Upsampling'\n \t'2' stands for 'downsampling'. \n \t'3' stands for 'SMOTE''"
    )
    samp_sel = int(input("Now enter your selection for sampling strategy: \t"))
    samp = [sampling.upsample, sampling.downsample, sampling.smote]
    temp = samp[samp_sel - 1]
    X_train, y_train = temp(X_train=pd.DataFrame(X_tr), y_train=pd.DataFrame(y_tr))
    return pd.DataFrame(X_train), pd.DataFrame(X_test), y_train, y_test


# %% [markdown]
# ## Class Balancing
# 1. Upsampling
# 2. Downsampling
# 3. SMOTE


# %%
class sampling:

    def upsample(X_train, y_train):
        # combine them back for resampling
        train_data = pd.concat([X_train, y_train], axis=1)
        # separate minority and majority classes
        negative = train_data[train_data.diagnosis == 0]
        positive = train_data[train_data.diagnosis == 1]
        # upsample minority
        pos_upsampled = resample(
            positive, replace=True, n_samples=len(negative), random_state=30
        )
        # combine majority and upsampled minority
        upsampled = pd.concat([negative, pos_upsampled])
        # check new class counts
        # print(upsampled.diagnosis.value_counts())
        print(upsampled.diagnosis.value_counts())
        upsampled = upsampled.sample(frac=1)
        X_train = upsampled.iloc[:, 0:-2]
        y_train = upsampled.iloc[:, -1]
        # graph barplot counts
        return X_train, y_train

    def downsample(X_train, y_train):
        # combine them back for resampling
        train_data = pd.concat([X_train, y_train], axis=1)
        # separate minority and majority classes
        negative = train_data[train_data.diagnosis == 0]
        positive = train_data[train_data.diagnosis == 1]
        # downsample majority
        neg_downsampled = resample(
            negative,
            replace=True,  # sample with replacement
            n_samples=len(positive),  # match number in minority class
            random_state=30,
        )  # reproducible results
        # combine minority and downsampled majority
        downsampled = pd.concat([positive, neg_downsampled])
        downsampled = downsampled.sample(frac=1)
        X_train = downsampled.iloc[:, 0:-2]
        y_train = downsampled.iloc[:, -1]
        # check new class counts
        print(downsampled.diagnosis.value_counts())
        # graph
        return X_train, y_train

    def smote(X_train, y_train):
        sm = SMOTE(random_state=30)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.DataFrame(y_train, columns=["diagnosis"])
        print(y_train.diagnosis.value_counts())
        # graph
        return X_train, y_train


# %% [markdown]
# ## Feature Selection
# 1. All Features
# 2. Mean Features
# 3. Squared Error Features
# 4. Worst Features
# 5. Selected Features


# %%
class feat:

    def feat1():
        # All Features
        df = data_load()  # Loading Dataset into Dataframe
        X, y = X_y_split(data_clean(df))
        return data_split_scale(X, y, sampling)

    def feat2():
        # Mean Features
        df = data_load()  # Loading Dataset into Dataframe
        df = data_clean(df)
        df_mean = df[df.columns[:11]]
        X, y = X_y_split(df_mean)
        return data_split_scale(X, y, sampling)

    def feat3():
        # Squared error Features
        df = data_load()  # Loading Dataset into Dataframe
        df = data_clean(df)
        df_se = df.drop(df.columns[1:11], axis=1)
        df_se = df_se.drop(df_se.columns[11:], axis=1)
        X, y = X_y_split(df_se)
        return data_split_scale(X, y, sampling)

    def feat4():
        # Worst Features
        df = data_load()  # Loading Dataset into Dataframe
        df = data_clean(df)
        df_worst = df.drop(df.columns[1:21], axis=1)
        X, y = X_y_split(df_worst)
        return data_split_scale(X, y, sampling)

    def feat5():
        # Selected Features
        df = data_load()  # Loading Dataset into Dataframe
        df = data_clean(df)
        drop_cols = [
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
            "perimeter_mean",
            "perimeter_se",
            "area_mean",
            "area_se",
            "concavity_mean",
            "concavity_se",
            "concave points_mean",
            "concave points_se",
        ]
        df_sf = df.drop(drop_cols, axis=1)
        X, y = X_y_split(df_sf)
        return data_split_scale(X, y, sampling)

    def feature():
        print(
            "'\t The number '1' stands for 'ALL- FEATURES'. \n \t The number '2' stands for 'MEAN- FEATURES' . \n \t The number '3' stands for 'SQUARED- ERROR FEATURES'. \n \t The number '4' stands for 'WORST- FEATURES'. \n \t The number '5' stands for 'SELECTED- FEATURES'.'"
        )
        selection = input("\t Enter your choice of feature selection: \t")
        feat_options = [feat.feat1, feat.feat2, feat.feat3, feat.feat4, feat.feat5]
        return feat_options[int(selection) - 1]()


# %% [markdown]
# ## Classification Algorithms
# 1. Logistic Regression
# 2. Decision Tree Classifier
# 3. Random Forest Classifier
# 4. K-Nearest Neighbors
# 5. Linear SVM
# 6. Kernal SVM
# 7. Gaussian Naive Bayes


# %%
class models:

    def lr(dat):
        # Logistic Regression
        start = t.time()
        lr = LogisticRegression()
        model_lr = lr.fit(dat[0], dat[2])
        pred = model_lr.predict(dat[1])
        pred_prob = model_lr.predict_proba(dat[1])
        stop = t.time()
        return model_lr, (stop - start), pred, pred_prob

    def dtc(dat):
        # Decision Tree Classifier
        start = t.time()
        dtc = DecisionTreeClassifier()
        model_dtc = dtc.fit(dat[0], dat[2])
        pred = model_dtc.predict(dat[1])
        pred_prob = model_dtc.predict_proba(dat[1])
        stop = t.time()
        return model_dtc, (stop - start), pred, pred_prob

    def rfc(dat):
        # Random Forest Classifier
        start = t.time()
        rfc = RandomForestClassifier()
        model_rfc = rfc.fit(dat[0], dat[2])
        pred = model_rfc.predict(dat[1])
        pred_prob = model_rfc.predict_proba(dat[1])
        stop = t.time()
        return model_rfc, (stop - start), pred, pred_prob

    def knn(dat):
        # K-Nearest Neighbors
        start = t.time()
        knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
        model_knn = knn.fit(dat[0], dat[2])
        pred = model_knn.predict(dat[1])
        pred_prob = model_knn.predict_proba(dat[1])
        stop = t.time()
        return model_knn, (stop - start), pred, pred_prob

    def svc_l(dat):
        # Linear SVM
        start = t.time()
        svc_l = SVC(kernel="linear", random_state=0, probability=True)
        model_svc_l = svc_l.fit(dat[0], dat[2])
        pred = model_svc_l.predict(dat[1])
        pred_prob = model_svc_l.predict_proba(dat[1])
        stop = t.time()
        return model_svc_l, (stop - start), pred, pred_prob

    def svc_r(dat):
        # Kernel SVM
        start = t.time()
        svc_r = SVC(kernel="rbf", random_state=0, probability=True)
        model_svc_r = svc_r.fit(dat[0], dat[2])
        pred = model_svc_r.predict(dat[1])
        pred_prob = model_svc_r.predict_proba(dat[1])
        stop = t.time()
        return model_svc_r, (stop - start), pred, pred_prob

    def gnb(dat):
        # GaussianNB
        start = t.time()
        gnb = GaussianNB()
        model_gnb = gnb.fit(dat[0], dat[2])
        pred = model_gnb.predict(dat[1])
        pred_prob = model_gnb.predict_proba(dat[1])
        stop = t.time()
        return model_gnb, (stop - start), pred, pred_prob


# %% [markdown]
# ## Training and Testing


# %%
def train_n_test():
    ft = feat.feature()
    modelsss = [
        models.lr,
        models.dtc,
        models.rfc,
        models.knn,
        models.svc_l,
        models.svc_r,
        models.gnb,
    ]
    print(
        "'\t The number '1' stands for 'LOGISTIC REGRESSION'. \n \t The number '2' stands for 'Decision Tree' . \n \t The number '3' stands for 'Random Forest Classifier'. \n \t The number '4' stands for 'KNN'. \n \t The number '5' stands for 'Liner SVM'. \n \t The number '6' stands for 'Kernal SVM'. \n \t The number '7' stands for 'Guassian NB'.'"
    )
    mdl_selection = int(input("Please enter your selection for models: \t"))
    model = modelsss[mdl_selection - 1]
    return model(ft), ft[3], mdl_selection


# %% [markdown]
# ## Model Performance Evaluation


# %%
def performance():
    out, y_test, mdl_selection = train_n_test()
    models = [
        "Logistic Regression",
        "Desicion Tree Classifier",
        "Random Forest Classifier",
        "KNN",
        "Liner SVM",
        "Kernal SVM",
        "Guassian NB",
    ]
    cm_lr = confusion_matrix(y_test, out[2])
    sns.heatmap(cm_lr, annot=True, cmap="Reds")
    plt.title("Confusion Matrix for {}".format(models[mdl_selection - 1]))
    acs = accuracy_score(y_test, out[2])
    rs = recall_score(y_test, out[2])
    fs = f1_score(y_test, out[2])
    ps = precision_score(y_test, out[2])
    # Report Bar Plot
    report = pd.DataFrame(classification_report(y_test, out[2], output_dict=True))
    rg = report.drop(report.index[3]).drop(report.columns[2:], axis=1)
    plt.style.use("seaborn")
    rg.plot(kind="bar", color=["red", "salmon"])
    plt.title("Classification Report of {}".format(models[mdl_selection - 1]))
    plt.legend(report.columns, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    plt.yticks(np.arange(0, 1.05, step=0.05))
    print(
        "\n\t The accuracy score of {} with given parameters is: {}%.".format(
            models[mdl_selection - 1], acs * 100
        )
    )
    print(
        "\n\t The recall score of {} with given parameters is: {}%.".format(
            models[mdl_selection - 1], rs * 100
        )
    )
    print(
        "\n\t The precision score of {} with given parameters is: {}%.".format(
            models[mdl_selection - 1], ps * 100
        )
    )
    print(
        "\n\t The F1 score of {} with given parameters is: {}%.".format(
            models[mdl_selection - 1], fs * 100
        )
    )
    print(
        "\n\t The training and testing time taken by {} with given parameters is: {} seconds.".format(
            models[mdl_selection - 1], out[1]
        )
    )
    prob = out[3]
    prob = prob[:, 1]
    # ROC
    false_pos, true_pos, thresh = roc_curve(y_test, prob, pos_label=1)
    auc_score = roc_auc_score(y_test, prob)
    rand_pr = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, rand_pr, pos_label=1)
    plt.figure()
    plt.style.use("seaborn")
    plt.plot(
        false_pos,
        true_pos,
        linestyle="--",
        color="orange",
        label=models[mdl_selection - 1],
    )
    plt.plot(p_fpr, p_tpr, linestyle="--", color="green")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")

    return out[0], out[2], auc_score


# %% [markdown]
# ## Final Step [User Inputs Required]

# %%
trained_model, pred, auc = performance()
