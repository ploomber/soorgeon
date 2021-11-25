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

# %% [markdown] _uuid="66406036d8dd7a0071295d1aee64f13bffc44e3a" _cell_guid="551ce207-0976-48c3-9242-fac3e6bdf527"
# # Introduction: Home Credit Default Risk Competition
#
# This notebook is intended for those who are new to machine learning competitions or want a gentle introduction to the problem. I purposely avoid jumping into complicated models or joining together lots of data in order to show the basics of how to get started in machine learning! Any comments or suggestions are much appreciated.
#
# In this notebook, we will take an initial look at the Home Credit default risk machine learning competition currently hosted on Kaggle. The objective of this competition is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. This is a standard supervised classification task:
#
# * __Supervised__: The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features
# * __Classification__: The label is a binary variable, 0 (will repay loan on time), 1 (will have difficulty repaying loan)
#
#
# # Data
#
# The data is provided by [Home Credit](http://www.homecredit.net/about-us.aspx), a service dedicated to provided lines of credit (loans) to the unbanked population. Predicting whether or not a client will repay a loan or have difficulty is a critical business need, and Home Credit is hosting this competition on Kaggle to see what sort of models the machine learning community can develop to help them in this task.
#
# There are 7 different sources of data:
#
# * application_train/application_test: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature `SK_ID_CURR`. The training application data comes with the `TARGET` indicating 0: the loan was repaid or 1: the loan was not repaid.
# * bureau: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.
# * bureau_balance: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length.
# * previous_application: previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature `SK_ID_PREV`.
# * POS_CASH_BALANCE: monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
# * credit_card_balance: monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.
# * installments_payment: payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.
#
# This diagram shows how all of the data is related:
#
# ![image](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)
#
# Moreover, we are provided with the definitions of all the columns (in `HomeCredit_columns_description.csv`) and an example of the expected submission file.
#
# In this notebook, we will stick to using only the main application training and testing data. Although if we want to have any hope of seriously competing, we need to use all the data, for now we will stick to one file which should be more manageable. This will let us establish a baseline that we can then improve upon. With these projects, it's best to build up an understanding of the problem a little at a time rather than diving all the way in and getting completely lost!
#
# ## Metric: ROC AUC
#
# Once we have a grasp of the data (reading through the [column descriptions](https://www.kaggle.com/c/home-credit-default-risk/data) helps immensely), we need to understand the metric by which our submission is judged. In this case, it is a common classification metric known as the [Receiver Operating Characteristic Area Under the Curve (ROC AUC, also sometimes called AUROC)](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it).
#
# The ROC AUC may sound intimidating, but it is relatively straightforward once you can get your head around the two individual concepts. The [Reciever Operating Characteristic (ROC) curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) graphs the true positive rate versus the false positive rate:
#
# ![image](http://www.statisticshowto.com/wp-content/uploads/2016/08/ROC-curve.png)
#
# A single line on the graph indicates the curve for a single model, and movement along a line indicates changing the threshold used for classifying a positive instance. The threshold starts at 0 in the upper right to and goes to 1 in the lower left. A curve that is to the left and above another curve indicates a better model. For example, the blue model is better than the red model, which is better than the black diagonal line which indicates a naive random guessing model.
#
# The [Area Under the Curve (AUC)](http://gim.unmc.edu/dxtests/roc3.htm) explains itself by its name! It is simply the area under the ROC curve. (This is the integral of the curve.) This metric is between 0 and 1 with a better model scoring higher. A model that simply guesses at random will have an ROC AUC of 0.5.
#
# When we measure a classifier according to the ROC AUC, we do not generation 0 or 1 predictions, but rather a probability between 0 and 1. This may be confusing because we usually like to think in terms of accuracy, but when we get into problems with inbalanced classes (we will see this is the case), accuracy is not the best metric. For example, if I wanted to build a model that could detect terrorists with 99.9999% accuracy, I would simply make a model that predicted every single person was not a terrorist. Clearly, this would not be effective (the recall would be zero) and we use more advanced metrics such as ROC AUC or the [F1 score](https://en.wikipedia.org/wiki/F1_score) to more accurately reflect the performance of a classifier. A model with a high ROC AUC will also have a high accuracy, but the [ROC AUC is a better representation of model performance.](https://datascience.stackexchange.com/questions/806/advantages-of-auc-vs-standard-accuracy)
#
# Not that we know the background of the data we are using and the metric to maximize, let's get into exploring the data. In this notebook, as mentioned previously, we will stick to the main data sources and simple models which we can build upon in future work.
#
# __Follow-up Notebooks__
#
# For those looking to keep working on this problem, I have a series of follow-up notebooks:
#
# * [Manual Feature Engineering Part One](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering)
# * [Manual Feature Engineering Part Two](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2)
# * [Introduction to Automated Feature Engineering](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics)
# * [Advanced Automated Feature Engineering](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory)
# * [Feature Selection](https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection)
# * [Intro to Model Tuning: Grid and Random Search](https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search)
# * [Automated Model Tuning](https://www.kaggle.com/willkoehrsen/automated-model-tuning)
# * [Model Tuning Results](https://www.kaggle.com/willkoehrsen/model-tuning-results-random-vs-bayesian-opt/notebook)
#
#
# I'll add more notebooks as I finish them! Thanks for all the comments!

# %% [markdown] _uuid="eb13bf76d4e1e60d0703856ec391cdc2c5bdf1fb" _cell_guid="d632b08c-d252-4238-b496-e2c6edebec4b"
# ## Imports
#
# We are using a typical data science stack: `numpy`, `pandas`, `sklearn`, `matplotlib`.

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown] _uuid="ded520f73b9e94ed47ac2e994a5fb1bcb9093d0f" _cell_guid="a5e67831-4751-4f11-8e07-527e3e092671"
# ## Read in Data
#
# First, we can list all the available data files. There are a total of 9 files: 1 main file for training (with target) 1 main file for testing (without the target), 1 example submission file, and 6 other files containing additional information about each loan.

# %% _uuid="c54e1559611512ebd447ac24f2226c2fffd61dcd" _cell_guid="2cdca894-e637-43a9-8f80-5791c2bb9041"
# List files available
print(os.listdir("input/"))

# %% _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0"
# Training data
app_train = pd.read_csv('input/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()

# %% [markdown] _uuid="4695541966d3d29e8a7a8975b072d01caff1631d"
# The training data has 307511 observations (each one a separate loan) and 122 features (variables) including the `TARGET` (the label we want to predict).

# %% _uuid="cbd1c4111df6f07bc0d479b51f50895e728b717a" _cell_guid="d077aee0-5271-440e-bc07-6087eab40b74"
# Testing data features
app_test = pd.read_csv('input/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()

# %% [markdown] _uuid="e351f02c8a5886756507a2d4f1ddba4791220f12"
# The test set is considerably smaller and lacks a `TARGET` column.

# %% [markdown] _uuid="0b1a02afd367d1c4ee3a3a936382ca42fb921b9d"
# # Exploratory Data Analysis
#
# Exploratory Data Analysis (EDA) is an open-ended process where we calculate statistics and make figures to find trends, anomalies, patterns, or relationships within the data. The goal of EDA is to learn what our data can tell us. It generally starts out with a high level overview, then narrows in to specific areas as we find intriguing areas of the data. The findings may be interesting in their own right, or they can be used to inform our modeling choices, such as by helping us decide which features to use.

# %% [markdown] _uuid="7c006a09627df1333c557dc11a09f372bde34dda" _cell_guid="23b20e53-3484-4c4b-bec9-2d8ac2ac918d"
# ## Examine the Distribution of the Target Column
#
# The target is what we are asked to predict: either a 0 for the loan was repaid on time, or a 1 indicating the client had payment difficulties. We can first examine the number of loans falling into each category.

# %% _uuid="2163ca09678b53dbe88388ccbc7d0e0f7d6c6230" _cell_guid="5fb6ab16-1b38-4ecf-8123-e48c7c061773"
app_train['TARGET'].value_counts()

# %% _uuid="1b2611fb3cf392023c3f40fd2f7b96f56f5dee7d" _cell_guid="0e93c1e2-f6b8-4a0b-82b6-7dad8df56048"
app_train['TARGET'].astype(int).plot.hist()

# %% [markdown] _uuid="119106000875202a0030109f14b73245fc4285e1" _cell_guid="48f008ff-d81e-46b2-80a3-e58f2a6627ca"
# From this information, we see this is an [_imbalanced class problem_](http://www.chioka.in/class-imbalance-problem/). There are far more loans that were repaid on time than loans that were not repaid. Once we get into more sophisticated machine learning models, we can [weight the classes](http://xgboost.readthedocs.io/en/latest/parameter.html) by their representation in the data to reflect this imbalance.

# %% [markdown] _uuid="58851dfef481f32b3026e89b086534ea3683440d" _cell_guid="507ec6b1-99d0-4324-a3ed-bdea2f916227"
# ## Examine Missing Values
#
# Next we can look at the number and percentage of missing values in each column.


# %% _uuid="7a2f5c72c45fa04d9fa95e8051ae595be806e9a2" _cell_guid="fc4c675f-e4a1-4e4f-9ece-3c59e5c8f7fd"
# Function to calculate missing values by column# Funct
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns={
        0: 'Missing Values',
        1: '% of Total Values'
    })

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# %% _uuid="98b0a82a3009b8f6d0bc718a2e1eaba779b4ace9" _cell_guid="786881f0-235e-441c-8319-f715a3b7d920"
# Missing values statistics
missing_values = missing_values_table(app_train)
missing_values.head(20)

# %% [markdown] _uuid="0b1be19103910ce83ebf54eeed99e42829643578" _cell_guid="df3c6a1d-b3ff-4565-bb32-5cba43c52729"
# When it comes time to build our machine learning models, we will have to fill in these missing values (known as imputation). In later work, we will use models such as XGBoost that can [handle missing values with no need for imputation](https://stats.stackexchange.com/questions/235489/xgboost-can-handle-missing-data-in-the-forecasting-phase). Another option would be to drop columns with a high percentage of missing values, although it is impossible to know ahead of time if these columns will be helpful to our model. Therefore, we will keep all of the columns for now.

# %% [markdown] _uuid="0672e40c3ab75a7901c0de35d248b322a227dc7f"
# ## Column Types
#
# Let's look at the number of columns of each data type. `int64` and `float64` are numeric variables ([which can be either discrete or continuous](https://stats.stackexchange.com/questions/206/what-is-the-difference-between-discrete-data-and-continuous-data)). `object` columns contain strings and are  [categorical features.](http://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/supporting-topics/basics/what-are-categorical-discrete-and-continuous-variables/) .

# %% _uuid="a03caadd76fa32f4b193e52467d4f39f2145d7b6"
# Number of each type of column
app_train.dtypes.value_counts()

# %% [markdown] _uuid="5859303c9acc63f7ff7acce063a9cd022a6d38cd"
# Let's now look at the number of unique entries in each of the `object` (categorical) columns.

# %% _uuid="2d021eda10939a19b141292d34491b357acd201a"
# Number of unique classes in each object column
app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

# %% [markdown] _uuid="10ceaf3ba31e26c822b242b1278d93ebfbefcc0a"
# Most of the categorical variables have a relatively small number of unique entries. We will need to find a way to deal with these categorical variables!

# %% [markdown] _uuid="1b49e667293daabffd8a4b2b6d02cf44bf6a3ba8" _cell_guid="86d1b309-5524-4298-b873-2c1c09eddec6"
# ## Encoding Categorical Variables
#
# Before we go any further, we need to deal with pesky categorical variables.  A machine learning model unfortunately cannot deal with categorical variables (except for some models such as [LightGBM](http://lightgbm.readthedocs.io/en/latest/Features.html)). Therefore, we have to find a way to encode (represent) these variables as numbers before handing them off to the model. There are two main ways to carry out this process:
#
# * Label encoding: assign each unique category in a categorical variable with an integer. No new columns are created. An example is shown below
#
# ![image](https://raw.githubusercontent.com/WillKoehrsen/Machine-Learning-Projects/master/label_encoding.png)
#
# * One-hot encoding: create a new column for each unique category in a categorical variable. Each observation recieves a 1 in the column for its corresponding category and a 0 in all other new columns.
#
# ![image](https://raw.githubusercontent.com/WillKoehrsen/Machine-Learning-Projects/master/one_hot_encoding.png)
#
# The problem with label encoding is that it gives the categories an arbitrary ordering. The value assigned to each of the categories is random and does not reflect any inherent aspect of the category. In the example above, programmer recieves a 4 and data scientist a 1, but if we did the same process again, the labels could be reversed or completely different. The actual assignment of the integers is arbitrary. Therefore, when we perform label encoding, the model might use the relative value of the feature (for example programmer = 4 and data scientist = 1) to assign weights which is not what we want. If we only have two unique values for a categorical variable (such as Male/Female), then label encoding is fine, but for more than 2 unique categories, one-hot encoding is the safe option.
#
# There is some debate about the relative merits of these approaches, and some models can deal with label encoded categorical variables with no issues. [Here is a good Stack Overflow discussion](https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor). I think (and this is just a personal opinion) for categorical variables with many classes, one-hot encoding is the safest approach because it does not impose arbitrary values to categories. The only downside to one-hot encoding is that the number of features (dimensions of the data) can explode with categorical variables with many categories. To deal with this, we can perform one-hot encoding followed by [PCA](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf) or other [dimensionality reduction methods](https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/) to reduce the number of dimensions (while still trying to preserve information).
#
# In this notebook, we will use Label Encoding for any categorical variables with only 2 categories and One-Hot Encoding for any categorical variables with more than 2 categories. This process may need to change as we get further into the project, but for now, we will see where this gets us. (We will also not use any dimensionality reduction in this notebook but will explore in future iterations).

# %% [markdown] _uuid="46f5bf9a6de52e270aa911ffd895e704da5426ec" _cell_guid="95627792-157e-457a-88a8-3b3875c7e1d5"
# ### Label Encoding and One-Hot Encoding
#
# Let's implement the policy described above: for any categorical variable (`dtype == object`) with 2 unique categories, we will use label encoding, and for any categorical variable with more than 2 unique categories, we will use one-hot encoding.
#
# For label encoding, we use the Scikit-Learn `LabelEncoder` and for one-hot encoding, the pandas `get_dummies(df)` function.

# %% _uuid="ddfaae5c3dcc7ec6bb47a2dffc10d364e8d25355" _cell_guid="70641d4d-1075-4837-8972-e58d70d8f242"
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])

            # Keep track of how many columns were label encoded
            le_count += 1

print('%d columns were label encoded.' % le_count)

# %% _uuid="6796c6dc793a08e162b6e20c6f185ef37bdf51f3" _cell_guid="0851773b-39fd-4cf0-9a66-e30adeef3e57"
# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# %% [markdown] _uuid="1b2c4198638ec8e5155097d112249de8754eb5c0" _cell_guid="61d910b5-84f5-4655-bd8a-d29672c13741"
# ### Aligning Training and Testing Data
#
# There need to be the same features (columns) in both the training and testing data. One-hot encoding has created more columns in the training data because there were some categorical variables with categories not represented in the testing data. To remove the columns in the training data that are not in the testing data, we need to `align` the dataframes. First we extract the target column from the training data (because this is not in the testing data but we need to keep this information). When we do the align, we must make sure to set `axis = 1` to align the dataframes based on the columns and not on the rows!

# %% _uuid="e0d12a13cb95521c19b10d8829e8abe2b1118396" _cell_guid="d99ca215-e893-490c-a6a4-83f3e8a067b3"
train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join='inner', axis=1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# %% [markdown] _uuid="802ffdbae02e43a9ca4e256ffcd6bd40ae15f3e9"
# The training and testing datasets now have the same features which is required for machine learning. The number of features has grown significantly due to one-hot encoding. At some point we probably will want to try [dimensionality reduction (removing features that are not relevant)](https://en.wikipedia.org/wiki/Dimensionality_reduction) to reduce the size of the datasets.

# %% [markdown] _uuid="4d7c8dd1d5bb5a0ef84cb78e6bff927249e62145" _cell_guid="13918211-0e6b-4d72-955b-f997db19eea2"
# ## Back to Exploratory Data Analysis
#
# ### Anomalies
#
# One problem we always want to be on the lookout for when doing EDA is anomalies within the data. These may be due to mis-typed numbers, errors in measuring equipment, or they could be valid but extreme measurements. One way to support anomalies quantitatively is by looking at the statistics of a column using the `describe` method. The numbers in the `DAYS_BIRTH` column are negative because they are recorded relative to the current loan application. To see these stats in years, we can mutliple by -1 and divide by the number of days in a year:
#
#

# %% _uuid="a60be93c2d7d63855e6d65c1109f408ad85da134"
(app_train['DAYS_BIRTH'] / -365).describe()

# %% [markdown] _uuid="acb37a3e3f2e0b2fd581259788b9255398314157"
# Those ages look reasonable. There are no outliers for the age on either the high or low end. How about the days of employment?

# %% _uuid="600c59dd5d970d3ccfea3a6af0036d85958adc91"
app_train['DAYS_EMPLOYED'].describe()

# %% [markdown] _uuid="1cdd9dafce28e497e08062cd3b189ac353c04cd9"
# That doesn't look right! The maximum value (besides being positive) is about 1000 years!

# %% _uuid="2878bfb3a2be4554f33e03e1a04d4c1978b52a08"
app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
plt.xlabel('Days Employment')

# %% [markdown] _uuid="d28ca1e799c0a6113cc5e920297e1dc93d380af4"
# Just out of curiousity, let's subset the anomalous clients and see if they tend to have higher or low rates of default than the rest of the clients.

# %% _uuid="67ea87d9ef6974b1780a7db1eefd13f90f81b5be"
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' %
      (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' %
      (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

# %% [markdown] _uuid="1edfcf786aadb004f083e9896989a29e43bf80da"
# Well that is extremely interesting! It turns out that the anomalies have a lower rate of default.
#
# Handling the anomalies depends on the exact situation, with no set rules. One of the safest approaches is just to set the anomalies to a missing value and then have them filled in (using Imputation) before machine learning. In this case, since all the anomalies have the exact same value, we want to fill them in with the same value in case all of these loans share something in common. The anomalous values seem to have some importance, so we want to tell the machine learning model if we did in fact fill in these values. As a solution, we will fill in the anomalous values with not a number (`np.nan`) and then create a new boolean column indicating whether or not the value was anomalous.
#
#

# %% _uuid="e23ec3cb89428f3dd994b572f718cc729740cfab"
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
plt.xlabel('Days Employment')

# %% [markdown] _uuid="839595437c0721e2480f6b4ee58f3060b222f166"
# The distribution looks to be much more in line with what we would expect, and we also have created a new column to tell the model that these values were originally anomalous (becuase we will have to fill in the nans with some value, probably the median of the column). The other columns with `DAYS` in the dataframe look to be about what we expect with no obvious outliers.
#
# As an extremely important note, anything we do to the training data we also have to do to the testing data. Let's make sure to create the new column and fill in the existing column with `np.nan` in the testing data.

# %% _uuid="a0d7c77b2adecaa878f39cf86ffddcfbbe51a190"
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

print('There are %d anomalies in the test data out of %d entries' %
      (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

# %% [markdown] _uuid="fd656b392faad3b34ecfa448b55ad03e75449e0a"
# ### Correlations
#
# Now that we have dealt with the categorical variables and the outliers, let's continue with the EDA. One way to try and understand the data is by looking for correlations between the features and the target. We can calculate the Pearson correlation coefficient between every variable and the target using the `.corr` dataframe method.
#
# The correlation coefficient is not the greatest method to represent "relevance" of a feature, but it does give us an idea of possible relationships within the data. Some [general interpretations of the absolute value of the correlation coefficent](http://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf) are:
#
#
# * .00-.19 “very weak”
# *  .20-.39 “weak”
# *  .40-.59 “moderate”
# *  .60-.79 “strong”
# * .80-1.0 “very strong”
#

# %% _uuid="d39d15d64db1f2c9015c6f542911ef9a9cac119e" _cell_guid="02acdb8d-d95f-41b9-8ad1-e2b6cb26f398"
# Find correlations with the target and sort
correlations = app_train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

# %% [markdown] _uuid="67e1f0f22ec8e26c38827c24ca1e9409d73c9c64" _cell_guid="8cfa409c-ec74-4fa4-8093-e7d00596c9c5"
# Let's take a look at some of more significant correlations: the `DAYS_BIRTH` is the most positive correlation. (except for `TARGET` because the correlation of a variable with itself is always 1!) Looking at the documentation, `DAYS_BIRTH` is the age in days of the client at the time of the loan in negative days (for whatever reason!). The correlation is positive, but the value of this feature is actually negative, meaning that as the client gets older, they are less likely to default on their loan (ie the target == 0). That's a little confusing, so we will take the absolute value of the feature and then the correlation will be negative.

# %% [markdown] _uuid="c1b831b6d1c3221efb123fbc1a4882aa1f598ec0" _cell_guid="0f7b1cfb-9e5c-4720-9618-ad326940f3f3"
# ### Effect of Age on Repayment

# %% _uuid="f705c7aa49486ec3bf119c4edc4e4af58861b88d" _cell_guid="b0ab583c-dfbb-4ff7-80e5-d747fc408499"
# Find the correlation of the positive days since birth and target
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])

# %% [markdown] _uuid="2b95e2c33bdd50682e7105d0f27b9cc3ad5b482d" _cell_guid="3fde277c-ebf1-4eaf-a353-c18fc4b518a6"
# As the client gets older, there is a negative linear relationship with the target meaning that as clients get older, they tend to repay their loans on time more often.
#
# Let's start looking at this variable. First, we can make a histogram of the age. We will put the x axis in years to make the plot a little more understandable.

# %% _uuid="739226c4594130d6aabeb25ffb8742c37657d7a4" _cell_guid="35e36393-e388-488e-ba7a-7473169d3e6f"
# Set the style of plots
plt.style.use('fivethirtyeight')

# Plot the distribution of ages in years
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
plt.title('Age of Client')
plt.xlabel('Age (years)')
plt.ylabel('Count')

# %% [markdown] _uuid="340680b4a4ecf310a6369808157b17cac7c13461" _cell_guid="02f5d3c5-e527-430b-a38d-531aeb8f3dd1"
# By itself, the distribution of age does not tell us much other than that there are no outliers as all the ages are reasonable. To visualize the effect of the age on the target, we will next make a [kernel density estimation plot](https://en.wikipedia.org/wiki/Kernel_density_estimation) (KDE) colored by the value of the target. A [kernel density estimate plot shows the distribution of a single variable](https://chemicalstatistician.wordpress.com/2013/06/09/exploratory-data-analysis-kernel-density-estimation-in-r-on-ozone-pollution-data-in-new-york-and-ozonopolis/) and can be thought of as a smoothed histogram (it is created by computing a kernel, usually a Gaussian, at each data point and then averaging all the individual kernels to develop a single smooth curve). We will use the seaborn `kdeplot` for this graph.

# %% _uuid="2e045e65f048789b577477356df4337c9e5e2087" _cell_guid="3982a18f-2731-4bb2-80c9-831b2377421f"
plt.figure(figsize=(10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365,
            label='target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365,
            label='target == 1')

# Labeling of plot
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Ages')

# %% [markdown] _uuid="57757e02285b8067b61e3f586174ad64bec78ac1" _cell_guid="9749e164-efea-47d2-ab60-5a8b89ff0570"
# The target == 1 curve skews towards the younger end of the range. Although this is not a significant correlation (-0.07 correlation coefficient), this variable is likely going to be useful in a machine learning model because it does affect the target. Let's look at this relationship in another way: average failure to repay loans by age bracket.
#
# To make this graph, first we `cut` the age category into bins of 5 years each. Then, for each bin, we calculate the average value of the target, which tells us the ratio of loans that were not repaid in each age category.

# %% _uuid="6c50572f095bff250bfed1993e2c53118277b5dd" _cell_guid="4296e926-7245-40df-bb0a-f6e59d8e566a"
# Age information into a separate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'],
                                  bins=np.linspace(20, 70, num=11))
age_data.head(10)

# %% _uuid="7082483e5fd9114856926de28968e5ae0b478b36" _cell_guid="18873d6b-3877-4c77-830e-0f3e10e5e7fb"
# Group by the bin and calculate averages
age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups

# %% _uuid="823b5032f472b05ce079ae5a7680389f31ddd8b7" _cell_guid="004d1021-d73f-4356-9ef8-0464c95d1708"
plt.figure(figsize=(8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation=75)
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')

# %% [markdown] _uuid="eb2bd6392ed6d6f7e002bc8dbea6aab0f30487d9" _cell_guid="2dad060f-bcab-4fe3-aa19-29fbf3e6fdab"
# There is a clear trend: younger applicants are more likely to not repay the loan! The rate of failure to repay is above 10% for the youngest three age groups and beolow 5% for the oldest age group.
#
# This is information that could be directly used by the bank: because younger clients are less likely to repay the loan, maybe they should be provided with more guidance or financial planning tips. This does not mean the bank should discriminate against younger clients, but it would be smart to take precautionary measures to help younger clients pay on time.

# %% [markdown] _uuid="43a3bb87bdaa65509e9dc887492239ae06cd1c77" _cell_guid="4749204f-ec63-4eeb-8d25-9c80967348f1"
# ### Exterior Sources
#
# The 3 variables with the strongest negative correlations with the target are `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3`.
# According to the documentation, these features represent a "normalized score from external data source". I'm not sure what this exactly means, but it may be a cumulative sort of credit rating made using numerous sources of data.
#
# Let's take a look at these variables.
#
# First, we can show the correlations of the `EXT_SOURCE` features with the target and with each other.

# %% _uuid="6197819149feaff75176e64e54c65ea6be3864fe" _cell_guid="e2ab3b7f-3a53-4495-a1de-31ad287f032a"
# Extract the EXT_SOURCE variables and show correlations
ext_data = app_train[[
    'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
]]
ext_data_corrs = ext_data.corr()
ext_data_corrs

# %% _uuid="20b21a6b4e15a726c29596abeb01346dc416729c" _cell_guid="0479863d-cfa9-47ab-83e6-7d7877e3e939"
plt.figure(figsize=(8, 6))

# Heatmap of correlations
sns.heatmap(ext_data_corrs,
            cmap=plt.cm.RdYlBu_r,
            vmin=-0.25,
            annot=True,
            vmax=0.6)
plt.title('Correlation Heatmap')

# %% [markdown] _uuid="6a592aa7c01858b268489ccb8fd00690cd26cd58" _cell_guid="78bd5acc-003d-4795-a57a-a6c4fc9c8c5f"
# All three `EXT_SOURCE` featureshave negative correlations with the target, indicating that as the value of the `EXT_SOURCE` increases, the client is more likely to repay the loan. We can also see that `DAYS_BIRTH` is positively correlated with `EXT_SOURCE_1` indicating that maybe one of the factors in this score is the client age.
#
# Next we can look at the distribution of each of these features colored by the value of the target. This will let us visualize the effect of this variable on the target.

# %% _uuid="49afab6b3790abcc2dea04c483f462f39e536503" _cell_guid="5e2b6507-96d1-4f96-964f-d8241e321f09"
plt.figure(figsize=(10, 12))

# iterate through the sources
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):

    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source],
                label='target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source],
                label='target == 1')

    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source)
    plt.ylabel('Density')

plt.tight_layout(h_pad=2.5)

# %% [markdown] _uuid="71ce5855665256dacfd7c52bceb11c68f5c58759" _cell_guid="0ee531e8-f131-4ae3-b542-d4bf550d9bd5"
# `EXT_SOURCE_3` displays the greatest difference between the values of the target. We can clearly see that this feature has some relationship to the likelihood of an applicant to repay a loan. The relationship is not very strong (in fact they are all [considered very weak](http://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf), but these variables will still be useful for a machine learning model to predict whether or not an applicant will repay a loan on time.

# %% [markdown] _uuid="53f486249f8afec0496d3de25120e57d956c2eb7"
# ## Pairs Plot
#
# As a final exploratory plot, we can make a pairs plot of the `EXT_SOURCE` variables and the `DAYS_BIRTH` variable. The [Pairs Plot](https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166) is a great exploration tool because it lets us see relationships between multiple pairs of variables as well as distributions of single variables. Here we are using the seaborn visualization library and the PairGrid function to create a Pairs Plot with scatterplots on the upper triangle, histograms on the diagonal, and 2D kernel density plots and correlation coefficients on the lower triangle.
#
# If you don't understand this code, that's all right! Plotting in Python can be overly complex, and for anything beyond the simplest graphs, I usually find an existing implementation and adapt the code (don't repeat yourself)!

# %% _uuid="9400f9d2810f4331005c9b91e040818279d1eaf8" _cell_guid="7b185a4e-ac04-4ff2-b5cb-46eacf6a70b6"
# Copy the data for plotting
plot_data = ext_data.drop(columns=['DAYS_BIRTH']).copy()

# Add in the age of the client in years
plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

# Drop na values and limit to first 100000 rows
plot_data = plot_data.dropna().loc[:100000, :]


# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8),
                xycoords=ax.transAxes,
                size=20)


# Create the pairgrid object
grid = sns.PairGrid(data=plot_data,
                    size=3,
                    diag_sharey=False,
                    hue='TARGET',
                    vars=[x for x in list(plot_data.columns) if x != 'TARGET'])

# Upper is a scatter plot
grid.map_upper(plt.scatter, alpha=0.2)

# Diagonal is a histogram
grid.map_diag(sns.kdeplot)

# Bottom is density plot
grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r)

plt.suptitle('Ext Source and Age Features Pairs Plot', size=32, y=1.05)

# %% [markdown] _uuid="88f9f486c74856bd87ff7699998088d9ee7fd926" _cell_guid="839f51f5-02f4-472d-9de4-aa2f760c171c"
# In this plot, the red indicates loans that were not repaid and the blue are loans that are paid. We can see the different relationships within the data. There does appear to be a moderate positive linear relationship between the `EXT_SOURCE_1` and the `DAYS_BIRTH` (or equivalently `YEARS_BIRTH`), indicating that this feature may take into account the age of the client.

# %% [markdown] _uuid="d5506d0483af10dbf71e8ed11c99b2d5253680fb" _cell_guid="bd49d18b-e35f-4122-a005-dd06d8f2f7ca"
# # Feature Engineering
#
# Kaggle competitions are won by feature engineering: those win are those who can create the most useful features out of the data. (This is true for the most part as the winning models, at least for structured data, all tend to be variants on [gradient boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)). This represents one of the patterns in machine learning: feature engineering has a greater return on investment than model building and hyperparameter tuning. [This is a great article on the subject)](https://www.featurelabs.com/blog/secret-to-data-science-success/). As Andrew Ng is fond of saying: "applied machine learning is basically feature engineering."
#
# While choosing the right model and optimal settings are important, the model can only learn from the data it is given. Making sure this data is as relevant to the task as possible is the job of the data scientist (and maybe some [automated tools](https://docs.featuretools.com/getting_started/install.html) to help us out).
#
# Feature engineering refers to a geneal process and can involve both feature construction: adding new features from the existing data, and feature selection: choosing only the most important features or other methods of dimensionality reduction. There are many techniques we can use to both create features and select features.
#
# We will do a lot of feature engineering when we start using the other data sources, but in this notebook we will try only two simple feature construction methods:
#
# * Polynomial features
# * Domain knowledge features
#

# %% [markdown] _uuid="70322dd11709dcaaf879a56103fde8fc787b7d4c" _cell_guid="464705a1-7ecf-47ba-a1fe-9f870102eb85"
# ## Polynomial Features
#
# One simple feature construction method is called [polynomial features](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html). In this method, we make features that are powers of existing features as well as interaction terms between existing features. For example, we can create variables `EXT_SOURCE_1^2` and `EXT_SOURCE_2^2` and also variables such as `EXT_SOURCE_1` x `EXT_SOURCE_2`, `EXT_SOURCE_1` x `EXT_SOURCE_2^2`, `EXT_SOURCE_1^2` x   `EXT_SOURCE_2^2`, and so on. These features that are a combination of multiple individual variables are called [interaction terms](https://en.wikipedia.org/wiki/Interaction_(statistics) because they  capture the interactions between variables. In other words, while two variables by themselves  may not have a strong influence on the target, combining them together into a single interaction variable might show a relationship with the target. [Interaction terms are commonly used in statistical models](https://www.theanalysisfactor.com/interpreting-interactions-in-regression/) to capture the effects of multiple variables, but I do not see them used as often in machine learning. Nonetheless, we can try out a few to see if they might help our model to predict whether or not a client will repay a loan.
#
# Jake VanderPlas writes about [polynomial features in his excellent book Python for Data Science](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html) for those who want more information.
#
# In the following code, we create polynomial features using the `EXT_SOURCE` variables and the `DAYS_BIRTH` variable. [Scikit-Learn has a useful class called `PolynomialFeatures`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) that creates the polynomials and the interaction terms up to a specified degree. We can use a degree of 3 to see the results (when we are creating polynomial features, we want to avoid using too high of a degree, both because the number of features scales exponentially with the degree, and because we can run into [problems with overfitting](http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py)).

# %% _uuid="a63d53dcac14c4ac2e31ea9c5e16b5d161c2415b" _cell_guid="e5b0efd9-67ac-4aa0-91e9-2141a87a6a8a"
# Make a new dataframe for polynomial features
poly_features = app_train[[
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET'
]]
poly_features_test = app_test[[
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
]]

# imputer for handling missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns=['TARGET'])

# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree=3)

# %% _uuid="72c5ecaae9c6ff038d16cbd9208f1abb69912631" _cell_guid="2be7c1ab-d1e5-40f2-b8e7-e2b2ce1e2f9a"
# Train the polynomial features
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)

# %% [markdown] _uuid="4d837e47bada5411ffce06266605f043c6ffe19e" _cell_guid="a7833b1e-714c-4988-8cbf-757d01290d8f"
# This creates a considerable number of new features. To get the names we have to use the polynomial features `get_feature_names` method.

# %% _uuid="121f98d2ec9c81c5dabb911dc68562d0b2b6d737" _cell_guid="7465d1e6-d360-4029-afa7-67cb34f60249"
poly_transformer.get_feature_names(input_features=[
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
])[:15]

# %% [markdown] _uuid="4e68b80b2738ef46863b53b7b781f299d602d316" _cell_guid="7eaeb645-bb25-4ac5-884f-d1231ab6d88f"
# There are 35 features with individual features raised to powers up to degree 3 and interaction terms. Now, we can see whether any of these new features are correlated with the target.

# %% _uuid="e712923de757457bb87a35ecaccd27007b351e6c" _cell_guid="95725a63-f8f2-4680-8f7a-4252f04e7f7f"
# Create a dataframe of the features
poly_features = pd.DataFrame(poly_features,
                             columns=poly_transformer.get_feature_names([
                                 'EXT_SOURCE_1', 'EXT_SOURCE_2',
                                 'EXT_SOURCE_3', 'DAYS_BIRTH'
                             ]))

# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))

# %% [markdown] _uuid="082ac97a068afed6758ed191acf5ab485e39230c" _cell_guid="971de432-c65e-4c9a-a3c0-c923ed27ddcb"
# Several of the new variables have a greater (in terms of absolute magnitude) correlation with the target than the original features. When we build machine learning models, we can try with and without these features to determine if they actually help the model learn.
#
# We will add these features to a copy of the training and testing data and then evaluate models with and without the features. Many times in machine learning, the only way to know if an approach will work is to try it out!

# %% _uuid="ed758ed436a86f92a8ee574999aa91089242ca7a"
# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test,
                                  columns=poly_transformer.get_feature_names([
                                      'EXT_SOURCE_1', 'EXT_SOURCE_2',
                                      'EXT_SOURCE_3', 'DAYS_BIRTH'
                                  ]))

# Merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

# Align the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly,
                                                     join='inner',
                                                     axis=1)

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)

# %% [markdown] _uuid="9b27fad1522263c32b57a8127c84ad0e08ff9d8f"
# ## Domain Knowledge Features
#
# Maybe it's not entirely correct to call this "domain knowledge" because I'm not a credit expert, but perhaps we could call this "attempts at applying limited financial knowledge". In this frame of mind, we can make a couple features that attempt to capture what we think may be important for telling whether a client will default on a loan. Here I'm going to use five features that were inspired by [this script](https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features) by Aguiar:
#
# * `CREDIT_INCOME_PERCENT`: the percentage of the credit amount relative to a client's income
# * `ANNUITY_INCOME_PERCENT`: the percentage of the loan annuity relative to a client's income
# * `CREDIT_TERM`:  the length of the payment in months (since the annuity is the monthly amount due
# * `DAYS_EMPLOYED_PERCENT`: the percentage of the days employed relative to the client's age
#
# Again, thanks to Aguiar and [his great script](https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features) for exploring these features.
#
#

# %% _uuid="c8d4b165b45da6c3120911de18e9348d8726c70c"
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain[
    'AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain[
    'AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain[
    'AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain[
    'DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

# %% _uuid="d017103871bd4935a8c29599d6be33e0e74b2f83"
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain[
    'AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain[
    'AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain[
    'AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain[
    'DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']

# %% [markdown] _uuid="7e917d654c05bd0ca3251d4f51c8176d82fe613f"
# #### Visualize New Variables
#
# We should explore these __domain knowledge__ variables visually in a graph. For all of these, we will make the same KDE plot colored by the value of the `TARGET`.

# %% _uuid="e9c10d7f55b4c636335f815762b93598fe4acb0a"
plt.figure(figsize=(12, 20))
# iterate through the new features
for i, feature in enumerate([
        'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM',
        'DAYS_EMPLOYED_PERCENT'
]):

    # create a new subplot for each source
    plt.subplot(4, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature],
                label='target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature],
                label='target == 1')

    # Label the plots
    plt.title('Distribution of %s by Target Value' % feature)
    plt.xlabel('%s' % feature)
    plt.ylabel('Density')

plt.tight_layout(h_pad=2.5)

# %% [markdown] _uuid="e27d400f3ec5447cfe5e908952351f271d521784"
# It's hard to say ahead of time if these new features will be useful. The only way to tell for sure is to try them out!

# %% [markdown] _uuid="8bf057e523b2d99833f6dc9d95fe6141fb4e325a" _cell_guid="ebb64e63-6222-4509-a43c-302c6435ce09"
# # Baseline
#
# For a naive baseline, we could guess the same value for all examples on the testing set.  We are asked to predict the probability of not repaying the loan, so if we are entirely unsure, we would guess 0.5 for all observations on the test set. This  will get us a Reciever Operating Characteristic Area Under the Curve (AUC ROC) of 0.5 in the competition ([random guessing on a classification task will score a 0.5](https://stats.stackexchange.com/questions/266387/can-auc-roc-be-between-0-0-5)).
#
# Since we already know what score we are going to get, we don't really need to make a naive baseline guess. Let's use a slightly more sophisticated model for our actual baseline: Logistic Regression.
#
# ## Logistic Regression Implementation
#
# Here I will focus on implementing the model rather than explaining the details, but for those who want to learn more about the theory of machine learning algorithms, I recommend both [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) and [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do). Both of these books present the theory and also the code needed to make the models (in R and Python respectively). They both teach with the mindset that the best way to learn is by doing, and they are very effective!
#
# To get a baseline, we will use all of the features after encoding the categorical variables. We will preprocess the data by filling in the missing values (imputation) and normalizing the range of the features (feature scaling). The following code performs both of these preprocessing steps.

# %% _uuid="784ae2f91cf7792702595a9973ba773b2acdec00" _cell_guid="60ef8744-ca3a-4810-8439-2835fbfc1833"
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns=['TARGET'])
else:
    train = app_train.copy()

# Feature names
features = list(train.columns)

# Copy of the testing data
test = app_test.copy()

# Median imputation of missing values
imputer = SimpleImputer(strategy='median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(app_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

# %% [markdown] _uuid="364f0835a46f7a7bb7be487b54d92f5ff50ed341" _cell_guid="1bcfab25-cc1c-4553-9473-96fcfeb2a61a"
# We will use [`LogisticRegression`from Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for our first model. The only change we will make from the default model settings is to lower the [regularization parameter](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), C, which controls the amount of overfitting (a lower value should decrease overfitting). This will get us slightly better results than the default `LogisticRegression`, but it still will set a low bar for any future models.
#
# Here we use the familiar Scikit-Learn modeling syntax: we first create the model, then we train the model using `.fit` and then we make predictions on the testing data using `.predict_proba` (remember that we want probabilities and not a 0 or 1).

# %% _uuid="9e8aba9401e8367f9902d710ba49e820294870e1" _cell_guid="6462ff85-e3b6-4a5f-b95c-9416841413b1"
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C=0.0001)

# Train on the training data
log_reg.fit(train, train_labels)

# %% [markdown] _uuid="0ad71fb750fac4af2845f30b0af73f5817e46101" _cell_guid="fe98191a-da57-4d50-8d56-7d8077fc6c26"
# Now that the model has been trained, we can use it to make predictions. We want to predict the probabilities of not paying a loan, so we use the model `predict.proba` method. This returns an m x 2 array where m is the number of observations. The first column is the probability of the target being 0 and the second column is the probability of the target being 1 (so for a single row, the two columns must sum to 1). We want the probability the loan is not repaid, so we will select the second column.
#
# The following code makes the predictions and selects the correct column.

# %% _uuid="2138782ddbfc9a803dc99a938460fc27d15972a9" _cell_guid="80c77c89-3fa9-4311-b441-412a4fbb1480"
# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]

# %% [markdown] _uuid="3a2612e95b13a94a13f679c1754b6c4fb28c332d"
# The predictions must be in the format shown in the `sample_submission.csv` file, where there are only two columns: `SK_ID_CURR` and `TARGET`. We will create a dataframe in this format from the test set and the predictions called `submit`.

# %% _uuid="09a3d281e4c7ee6820f402e32f31775851113089"
# Submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()

# %% [markdown] _uuid="2a1bf4f54df8b37a71a7732e61a7bfebafd8be11"
# The predictions represent a probability between 0 and 1 that the loan will not be repaid. If we were using these predictions to classify applicants, we could set a probability threshold for determining that a loan is risky.

# %% _uuid="fcaf338e52d8f42f119b31d437b516e336e787ec" _cell_guid="77204f15-c3b9-4c67-8d93-173fa3afceaa"
# Save the submission to a csv file
submit.to_csv('log_reg_baseline.csv', index=False)

# %% [markdown] _uuid="11e18bd5c4e75b931f90a22bc6ff84441a13570c"
# The submission has now been saved to the virtual environment in which our notebook is running. To access the submission, at the end of the notebook, we will hit the blue Commit & Run button at the upper right of the kernel. This runs the entire notebook and then lets us download any files that are created during the run.
#
# Once we run the notebook, the files created are available in the Versions tab under the Output sub-tab. From here, the submission files can be submitted to the competition or downloaded. Since there are several models in this notebook, there will be multiple output files.
#
# __The logistic regression baseline should score around 0.671 when submitted.__

# %% [markdown] _uuid="92687ac866441f6ee2919aa5e5c935490c172afc" _cell_guid="462ea34f-3f66-490a-a61f-24a991271f69"
# ## Improved Model: Random Forest
#
# To try and beat the poor performance of our baseline, we can update the algorithm. Let's try using a Random Forest on the same training data to see how that affects performance. The Random Forest is a much more powerful model especially when we use hundreds of trees. We will use 100 trees in the random forest.

# %% _uuid="cf05e2318904b8f3575ae233c185cd995fd07643" _cell_guid="6643479e-7980-431c-a6a2-9087acdb0f42"
from sklearn.ensemble import RandomForestClassifier

# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators=100,
                                       random_state=50,
                                       verbose=1,
                                       n_jobs=-1)

# %% _uuid="52258a9b89b3069bc1d82829107e8e7c1ef05fd6" _cell_guid="020f0856-8f24-4b22-bca5-aac7f137f032"
# Train on the training data
random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({
    'feature': features,
    'importance': feature_importance_values
})

# Make predictions on the test data
predictions = random_forest.predict_proba(test)[:, 1]

# %% _uuid="1da4b02502388d2b8a2bc5376027c5bef50272f3" _cell_guid="25145966-669e-426d-89a3-98e30b861057"
# Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline.csv', index=False)

# %% [markdown] _uuid="cf6f600ed10c511dd26d4bd5efa7997ab8d6916a"
# These predictions will also be available when we run the entire notebook.
#
# __This model should score around 0.678 when submitted.__

# %% [markdown] _uuid="43d979aed7cdfd6d7bd6a995b5756a384bd2b7dc"
# ### Make Predictions using Engineered Features
#
# The only way to see if the Polynomial Features and Domain knowledge improved the model is to train a test a model on these features! We can then compare the submission performance to that for the model without these features to gauge the effect of our feature engineering.

# %% _uuid="d9d49008fb73b8d15c797850c64d5e6f81375163"
poly_features_names = list(app_train_poly.columns)

# Impute the polynomial features
imputer = SimpleImputer(strategy='median')

poly_features = imputer.fit_transform(app_train_poly)
poly_features_test = imputer.transform(app_test_poly)

# Scale the polynomial features
scaler = MinMaxScaler(feature_range=(0, 1))

poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

random_forest_poly = RandomForestClassifier(n_estimators=100,
                                            random_state=50,
                                            verbose=1,
                                            n_jobs=-1)

# %% _uuid="a7d3f3b6cdf8231832c56224c8a694056e456593"
# Train on the training data
random_forest_poly.fit(poly_features, train_labels)

# Make predictions on the test data
predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]

# %% _uuid="cd923eed057b6d61354db27473d9a36f1411dd5c"
# Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline_engineered.csv', index=False)

# %% [markdown] _uuid="ec50627c874a9d78d6789e01a47e829c820f9615"
# This model scored 0.678 when submitted to the competition, exactly the same as that without the engineered features. Given these results, it does not appear that our feature construction helped in this case.
#
# #### Testing Domain Features
#
# Now we can test the domain features we made by hand.

# %% _uuid="04b93e7d3629c1a5ba27a6eed037900862dc039d"
app_train_domain = app_train_domain.drop(columns='TARGET')

domain_features_names = list(app_train_domain.columns)

# Impute the domainnomial features
imputer = SimpleImputer(strategy='median')

domain_features = imputer.fit_transform(app_train_domain)
domain_features_test = imputer.transform(app_test_domain)

# Scale the domainnomial features
scaler = MinMaxScaler(feature_range=(0, 1))

domain_features = scaler.fit_transform(domain_features)
domain_features_test = scaler.transform(domain_features_test)

random_forest_domain = RandomForestClassifier(n_estimators=100,
                                              random_state=50,
                                              verbose=1,
                                              n_jobs=-1)

# Train on the training data
random_forest_domain.fit(domain_features, train_labels)

# Extract feature importances
feature_importance_values_domain = random_forest_domain.feature_importances_
feature_importances_domain = pd.DataFrame({
    'feature':
    domain_features_names,
    'importance':
    feature_importance_values_domain
})

# Make predictions on the test data
predictions = random_forest_domain.predict_proba(domain_features_test)[:, 1]

# %% _uuid="27598fb499df4c3282be63356422e4a6f6d6dd17"
# Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline_domain.csv', index=False)

# %% [markdown] _uuid="240fb8ba2b5fe73da4d021543fd64baa104fb418"
# This scores 0.679 when submitted which probably shows that the engineered features do not help in this model (however they do help in the Gradient Boosting Model at the end of the notebook).
#
# In later notebooks, we will do more [feature engineering](https://docs.featuretools.com/index.html) by using the information from the other data sources. From experience, this will definitely help our model!

# %% [markdown] _uuid="b1805834b4d4eae38db4f68502aade956fc1e10f" _cell_guid="b742ed91-9dd6-4a7b-af5e-1d6e7128beb2"
# ## Model Interpretation: Feature Importances
#
# As a simple method to see which variables are the most relevant, we can look at the feature importances of the random forest. Given the correlations we saw in the exploratory data analysis, we should expect that the most important features are the `EXT_SOURCE` and the `DAYS_BIRTH`. We may use these feature importances as a method of dimensionality reduction in future work.


# %% _uuid="b912337a5f35f495398d8ae8b8576ceb7062fe50" _cell_guid="a90e9368-5f7d-4179-a5cc-1025f32c6a81"
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center',
            edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()

    return df


# %% _uuid="37309c4a94b248ad85fa7a0825f01830a818ba92" _cell_guid="1084ad42-bc44-438b-b2fd-5fd7a1c1b363"
# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)

# %% [markdown] _uuid="524c6aa12acc80e7018750e7a8897dc6b4bacf18"
# As expected, the most important features are those dealing with `EXT_SOURCE` and `DAYS_BIRTH`. We see that there are only a handful of features with a significant importance to the model, which suggests we may be able to drop many of the features without a decrease in performance (and we may even see an increase in performance.) Feature importances are not the most sophisticated method to interpret a model or perform dimensionality reduction, but they let us start to understand what factors our model takes into account when it makes predictions.

# %% _uuid="516e4b2eedeec2ff441f1ff034fbe4a73374bba2"
feature_importances_domain_sorted = plot_feature_importances(
    feature_importances_domain)

# %% [markdown] _uuid="4a2cdf24d5ecc01539d10902a9c7af6a13096086"
# We see that all four of our hand-engineered features made it into the top 15 most important! This should give us confidence that our domain knowledge was at least partially on track.

# %% [markdown] _uuid="fd407ca0f7c5c50ee71fe5c8532eabeb92c15c50"
# # Conclusions
#
# In this notebook, we saw how to get started with a Kaggle machine learning competition. We first made sure to understand the data, our task, and the metric by which our submissions will be judged. Then, we performed a fairly simple EDA to try and identify relationships, trends, or anomalies that may help our modeling. Along the way, we performed necessary preprocessing steps such as encoding categorical variables, imputing missing values, and scaling features to a range. Then, we constructed new features out of the existing data to see if doing so could help our model.
#
# Once the data exploration, data preparation, and feature engineering was complete, we implemented a baseline model upon which we hope to improve. Then we built a second slightly more complicated model to beat our first score. We also carried out an experiment to determine the effect of adding the engineering variables.
#
# We followed the general outline of a [machine learning project](https://towardsdatascience.com/a-complete-machine-learning-walk-through-in-python-part-one-c62152f39420):
#
# 1.  Understand the problem and the data
# 2. Data cleaning and formatting (this was mostly done for us)
# 3. Exploratory Data Analysis
# 4. Baseline model
# 5.  Improved model
# 6. Model interpretation (just a little)
#
# Machine learning competitions do differ slightly from typical data science problems in that we are concerned only with achieving the best performance on a single metric and do not care about the interpretation. However, by attempting to understand how our models make decisions, we can try to improve them or examine the mistakes in order to correct the errors. In future notebooks we will look at incorporating more sources of data, building more complex models (by following the code of others), and improving our scores.
#
# I hope this notebook was able to get you up and running in this machine learning competition and that you are now ready to go out on your own - with help from the community - and start working on some great problems!
#
# __Running the notebook__: now that we are at the end of the notebook, you can hit the blue Commit & Run button to execute all the code at once. After the run is complete (this should take about 10 minutes), you can then access the files that were created by going to the versions tab and then the output sub-tab. The submission files can be directly submitted to the competition from this tab or they can be downloaded to a local machine and saved. The final part is to share the share the notebook: go to the settings tab and change the visibility to Public. This allows the entire world to see your work!
#
# ### Follow-up Notebooks
#
# For those looking to keep working on this problem, I have a series of follow-up notebooks:
#
# * [Manual Feature Engineering Part One](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering)
# * [Manual Feature Engineering Part Two](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2)
# * [Introduction to Automated Feature Engineering](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics)
# * [Advanced Automated Feature Engineering](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory)
# * [Feature Selection](https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection)
# * [Intro to Model Tuning: Grid and Random Search](https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search)
#
# As always, I welcome feedback and constructive criticism. I write for Towards Data Science at https://medium.com/@williamkoehrsen/ and can be reached on Twitter at https://twitter.com/koehrsen_will
#
# Will
#

# %% [markdown] _uuid="a8bc307f9be27bfabbc3891deddbd94293ca03fa" _cell_guid="d12452cd-347e-4269-b3d4-f5f0589f4c5c"
# # Just for Fun: Light Gradient Boosting Machine
#
# Now (if you want, this part is entirely optional) we can step off the deep end and use a real machine learning model: the [gradient boosting machine](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/) using the [LightGBM library](http://lightgbm.readthedocs.io/en/latest/Quick-Start.html)! The Gradient Boosting Machine is currently the leading model for learning on structured datasets (especially on Kaggle) and we will probably need some form of this model to do well in the competition. Don't worry, even if this code looks intimidating, it's just a series of small steps that build up to a complete model. I added this code just to show what may be in store for this project, and because it gets us a slightly better score on the leaderboard. In future notebooks we will see how to work with more advanced models (which mostly means adapting existing code to make it work better), feature engineering, and feature selection. See you in the next notebook!

# %% _uuid="2719663ed461422fce26b5dd55a31ab9718df47a" _cell_guid="60208a3f-947f-42d9-8f46-2159afd2eb7d"
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc


def model(features, test_features, encoding='ohe', n_folds=5):
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """

    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    # Extract the labels for training
    labels = features['TARGET']

    # Remove the ids and target
    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])

    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        # Align the dataframes by the columns
        features, test_features = features.align(test_features,
                                                 join='inner',
                                                 axis=1)

        # No categorical indices to record
        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        # Create a label encoder
        label_encoder = LabelEncoder()

        # List for storing categorical indices
        cat_indices = []

        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(
                    np.array(features[col].astype(str)).reshape((-1, )))
                test_features[col] = label_encoder.transform(
                    np.array(test_features[col].astype(str)).reshape((-1, )))

                # Record the categorical indices
                cat_indices.append(i)

    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):

        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[
            train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[
            valid_indices]

        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000,
                                   objective='binary',
                                   class_weight='balanced',
                                   learning_rate=0.05,
                                   reg_alpha=0.1,
                                   reg_lambda=0.1,
                                   subsample=0.8,
                                   n_jobs=-1,
                                   random_state=50)

        # Train the model
        model.fit(train_features,
                  train_labels,
                  eval_metric='auc',
                  eval_set=[(valid_features, valid_labels),
                            (train_features, train_labels)],
                  eval_names=['valid', 'train'],
                  categorical_feature=cat_indices,
                  early_stopping_rounds=100,
                  verbose=200)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict_proba(
            test_features, num_iteration=best_iteration)[:,
                                                         1] / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(
            valid_features, num_iteration=best_iteration)[:, 1]

        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': test_predictions
    })

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_values
    })

    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({
        'fold': fold_names,
        'train': train_scores,
        'valid': valid_scores
    })

    return submission, feature_importances, metrics


# %% _uuid="89e02dcbb23e47e3504ed1f61431b182e2011ba5"
submission, fi, metrics = model(app_train, app_test)
print('Baseline metrics')
print(metrics)

# %% _uuid="ca59467edd1060e5f7587a77a31dcd7331ce90ec"1
fi_sorted = plot_feature_importances(fi)

# %% _uuid="d71f9d7b9b322824704eec9dc82e38a480d4f76c"
submission.to_csv('baseline_lgb.csv', index=False)

# %% [markdown] _uuid="2aca0b9ea31dfef1ca3221dc6424fe31e829cbbf"
# This submission should score about 0.735 on the leaderboard. We will certainly best that in future work!

# %% _uuid="cd53d758d2838c9b99b9ae44780514d13373b717"
app_train_domain['TARGET'] = train_labels

# Test the domain knolwedge features
submission_domain, fi_domain, metrics_domain = model(app_train_domain,
                                                     app_test_domain)
print('Baseline with domain knowledge features metrics')
print(metrics_domain)

# %% _uuid="58a2d9b330a223733e3673b24433c41122d3b611"
fi_sorted = plot_feature_importances(fi_domain)

# %% [markdown] _uuid="947f49722aa5ae1df696c311d7413e91ba51e1b9"
# Again, we see tha some of our features made it into the most important. Going forward, we will need to think about whatother domain knowledge features may be useful for this problem (or we should consult someone who knows more about the financial industry!

# %% _uuid="7dfc9123c7e231826be54a1c022e373a1ee68f51"
submission_domain.to_csv('baseline_lgb_domain_features.csv', index=False)

# %% [markdown] _uuid="033b8d192a40127ea99d6f4fb13c624dd64c6611"
# This model scores about 0.754 when submitted to the public leaderboard indicating that the domain features do improve the performance! [Feature engineering](https://en.wikipedia.org/wiki/Feature_engineering) is going to be a critical part of this competition (as it is for all machine learning problems)!

# %% _uuid="a7e9a1149953069853d4d83ec46f22084dce8711"
