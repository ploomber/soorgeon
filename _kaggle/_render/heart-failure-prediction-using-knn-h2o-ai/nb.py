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

# %% [markdown]
# ## **Heart Failure Prediction!**
#
# In this Notebook we will see how to apply KNN and how to use H2o.ai automl library for classification task. If you find this notebook usefull please Upvote!

# %% id="-eFeHGM7wjXi"
# importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% id="BeFbH5mmwjXj" outputId="1d28b870-a332-42d9-db22-6bf5979bd77b"
# importing dataset
df = pd.read_csv("../input/heart-failure-prediction/heart.csv")
df.head()

# %% id="58V7HLIKwjXk" outputId="16830d86-2484-458f-bf53-30e988f99219"
df.shape

# %% id="EKqElv-D1s5Z" outputId="65eba788-4183-4354-f74a-ad7c4a1e6317"
df.info()

# %% id="A82b8jbLwjXk" outputId="a916ec4d-6c0d-4bb0-841e-5d675676cc4e"
df.describe()

# %% id="vCpUngBKVac2" outputId="68162c6f-6bd5-4794-c332-e15701dabfe5"
df.isnull().sum()
# There is no null values

# %% [markdown] id="o2qmlr1DdLX9"
# ## Data Exploration

# %% [markdown] id="dBRqUEwuy1KY"
#
#  Now we can plot the distribution of data wrt dependent variable i.e HeartDisease

# %% id="JsDv6UI4wjXn" outputId="6b218281-de0c-472c-b871-6595a7d069d2"
sns.pairplot(df, hue="HeartDisease")

# %% [markdown] id="ahSOpNk1zEta"
#   5. Which are most useful variable in classification? Prove using correlation.

# %% id="uXcHuo7pzCLZ" outputId="6c4cdfd1-7b22-4967-89e8-ffb3e9e2a152"
corr = df.corr()
corr.style.background_gradient(cmap="coolwarm")

# %% id="OG7hK1UJ0Rja" outputId="85b4bd55-da22-45b9-cce1-4cfea00ab94f"
sns.set_theme(style="whitegrid")
sns.boxplot(x="Age", data=df, palette="Set3")
plt.title("Age Distribution")

# %% id="wMVgYoGw0oIl" outputId="2c985389-575f-4635-a0bf-cf2f56bb137e"
fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
df.hist(ax=ax)

# %% id="kzgu_ezi03y8" outputId="76bbae80-a830-45c2-b0e6-e6c787ecd193"
df.HeartDisease.value_counts().plot(kind="bar")
plt.xlabel("Heart Diseases or Not")
plt.ylabel("Count")
plt.title("Heart Diseases")
# Here we can see that dataset is not much imbalanced so there is no need to balance.

# %% [markdown] id="-7E3IpLKdRV1"
# ## Data Preprocessing

# %% id="zaHcUNcbWjkZ"
cat = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# %% id="tVKS4fLEWBZc"
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df[cat] = df[cat].apply(lb.fit_transform)

# %% id="4OPXop0HwjXo" outputId="997a4068-3bfd-4298-8a60-584126552665"
X = df.drop("HeartDisease", axis=1)
X.head()

# %% id="KLJJYpttwjXo" outputId="00502435-bf1a-42e9-d631-c767be3f82bc"
y = df["HeartDisease"]
y.head()

# %% id="0T5IVw1awjXp"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% id="Cw17uwfRwjXp" outputId="e218a7c7-4a0e-42b3-c151-3b0b3d4e5205"
X_train.shape

# %% id="p3L49Xf8wjXq" outputId="09340038-570a-453a-9cf8-ed9afee82303"
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown] id="o2xpOahkwjXt"
# ## Using KNN
#
# K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. However, it is mainly used for classification predictive problems in industry. The following two properties would define KNN well −
#
# * Lazy learning algorithm − KNN is a lazy learning algorithm because it does not have a specialized training phase and uses all the data for training while classification.
#
# * Non-parametric learning algorithm − KNN is also a non-parametric learning algorithm because it doesn’t assume anything about the underlying data.

# %% id="WB5aOGmiwjXv"
from sklearn.neighbors import KNeighborsClassifier

# %% id="ruOf41A6wjXw" outputId="d39b639a-1acf-484a-dbea-0684c9945b3b"
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", p=2)
knn.fit(X_train, y_train)

# %% id="To358l1ewjXw" outputId="67755f95-06f1-427d-aa9b-49660f3c2c51"
y_pred = knn.predict(X_test)
y_pred

# %% id="T0iEKtMKwjXx" outputId="175787b0-d6b1-4839-cfa5-d5420308b378"
knn.score(X_test, y_test)

# %% id="KGBIL8jwwjXx"
from sklearn.metrics import accuracy_score
from sklearn import metrics

# %% id="7piDOhe1wjXx" outputId="a66fe438-6dfb-4b08-e5fc-80c758664db2"
metrics.accuracy_score(y_test, y_pred)

# %% id="oE_JuTbAwjXy" outputId="3c59caf9-3488-429c-f2be-bc2102738ee3"
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, y_pred)
mat

# %% id="vefokjVQ2loi" outputId="73ce3707-11c8-40c1-c6b4-b5f4b9eefb18"
from sklearn.metrics import classification_report

target_names = ["Heart Diseases", "Normal"]
print(classification_report(y_test, y_pred, target_names=target_names))

# %% [markdown] id="mIL4fznQ3A4P"
# To select optimize k value we will use elbow method

# %% id="OrZlJRGMwjXy"
# For selecting K value
error_rate = []

# Will take some time
for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# %% id="UdPt0WK3wjXy" outputId="852d55a4-9a39-4298-b145-94bc7f23eb7a"
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, 40),
    error_rate,
    color="red",
    linestyle="dashed",
    marker="o",
    markerfacecolor="green",
    markersize=10,
)
plt.title("Error Rate vs. K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")

# %% id="AgeM11Jv3F9X" outputId="d58693b2-1908-4924-e1a6-2b16792815d9"
# From graph we can see that optimize k value is 16,17,18
# Now we will train our KNN classifier with this k values

knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean", p=2)
knn.fit(X_train, y_train)

# %% id="dCLyRoU13X9n" outputId="f7ae117c-2e18-496c-925d-a78728eefabe"
y_pred = knn.predict(X_test)
y_pred

# %% id="MTDLrkM33upo" outputId="49eb41dd-bf1b-4398-ad41-f606dbd5f0fe"
knn.score(X_test, y_test)

# %% id="yXXP59Ff3r5Q" outputId="dfb33483-6385-48e4-8988-f565adbaafe1"
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(mat, annot=True)

# %% id="Z7NJY1xg3a2v" outputId="203352f9-cb86-433d-c406-68e139682617"
from sklearn.metrics import classification_report

target_names = ["Diabetes", "Normal"]
print(classification_report(y_test, y_pred, target_names=target_names))

# %% [markdown] id="IkXajhdm4Pur"
# 6. Quantify goodness of your model and discuss steps taken for improvement.
#
#     For this dataset KNN had archive 87% accuracy. We can further improve accuracy by using bagging and boosting techniques.
#
# 7. Can we use KNN for regression also? Why / Why not?
#
#     KNN algorithm can be used for both classification and regression problems. The KNN algorithm uses ‘feature similarity’ to predict the values of any new data points. This means that the new point is assigned a value based on how closely it resembles the points in the training set.
#
# 8. Discuss drawbacks of algorithms such as KNN
#
#     -> It does not work well with large dataset and high dimensional dataset.
#
#     -> Knn is noise sensitive dataset, we need to do feature engineering like outlier removal, handling missing value,etc.
#
#     -> Require high memory – need to store all of the training data
#
#     -> Given that it stores all of the training, it can be computationally expensive
#

# %% [markdown] id="OQM4NIy8daNG"
# ## Using H2o.ai AutoML

# %% id="CuEIHC-zYRfi" outputId="e04d6864-4028-483a-df10-249e6275fc47"

# %% id="bXVffHNX4JkT" outputId="360dcee6-ea22-4555-9cfe-0e117b36fe41"
import h2o

# We will be using default parameter Here with H2O init method
h2o.init()

# %% id="q9PVDyTmYQcL" outputId="b1f4d3ef-b531-4cba-a7bc-0d9c71b92c04"
# Convert to h2o dataframe
hf = h2o.H2OFrame(df)

# %% id="DlXM7bxrY8Fn" outputId="28a1baa7-be21-4f52-8b36-05d90ba90c7e"
# Data Transform - Split train : test datasets
train, valid = hf.split_frame(ratios=[0.80], seed=1234)
print("Training Dataset", train.shape)
print("Validation Dataset", valid.shape)

# %% id="hzGc-1jGZAO0" outputId="1b638585-7444-4c59-8a33-25885d3e94fe"
train.head(5)

# %% id="SchOpTAhaN71" outputId="2690d099-f8f8-4031-93df-ceaf8a44206a"
valid.head()

# %% id="hWcWh0jXZCSV"
# Identify predictors and response
featureColumns = train.columns
targetColumn = "HeartDisease"
featureColumns.remove(targetColumn)

# %% id="ITk8vdwOZLVe" outputId="4b37ab65-dc09-4b26-e081-aeaa1cc8d99e"
import time
from h2o.automl import H2OAutoML

# Run AutoML for YY base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=12, seed=1234, balance_classes=True)
aml.train(
    x=featureColumns, y=targetColumn, training_frame=train, validation_frame=valid
)

# %% id="ql70026xZfdI" outputId="95e3404d-ad3b-46fe-a432-ab34593de3bc"
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

# Explain an AutoML object i.e. explain all models
exa = aml.explain(valid)

# %% id="pLsL0vEdZp1r"
# Evaluate the best model with testing data.
model = aml.leader

# %% id="osb0CpR5Z5IF" outputId="f4e8a259-dcfc-4018-b784-80b123e4d4fd"

# %% id="mihl7WKqbPow" outputId="86853480-ae0c-42c1-b831-9f6df1bc442b"
# For Classification
import scikitplot as skplt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Predict with the best model.
predicted_y = model.predict(valid[featureColumns])

predicted_data = predicted_y.as_data_frame()
valid_dataset = valid.as_data_frame()

# Evaluate the skill of the Trained model
acc = accuracy_score(
    valid_dataset[targetColumn], np.round(abs(predicted_data["predict"]))
)
classReport = classification_report(
    valid_dataset[targetColumn], np.round(abs(predicted_data["predict"]))
)
confMatrix = confusion_matrix(
    valid_dataset[targetColumn], np.round(abs(predicted_data["predict"]))
)

print()
print("Testing Results of the trained model: ")
print()
print("Accuracy : ", acc)
print()
print("Confusion Matrix :\n", confMatrix)
print()
print("Classification Report :\n", classReport)

# Confusion matrix
skplt.metrics.plot_confusion_matrix(
    valid_dataset[targetColumn],
    np.round(abs(predicted_data["predict"])),
    figsize=(7, 7),
)
plt.show()
