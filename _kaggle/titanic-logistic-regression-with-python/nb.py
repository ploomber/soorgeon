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

# %% [markdown] _cell_guid="cb19f71d-51c8-417f-829e-3179d3319dcd" _uuid="c3b9226e142667d6b96e34daf7d6e42bea0ea1e2"
# # **Introduction**
#
# This kernel was inspired in part by the work of [SarahG](https://www.kaggle.com/sgus1318/titanic-analysis-learning-to-swim-with-python)'s analysis that I thank very much for the quality of her analysis. This work represents a deeper analysis by playing on several parameters while using only logistic regression estimator. In a future work, I will discuss other techniques. I am open to any criticism and proposal. You do not hesitate to evaluate this analysis.
# The following kernel contains the steps enumerated below for assessing the Titanic survival dataset:
#
# 1. [Import data and python packages](#t1.)
# 2. [Assess Data Quality & Missing Values](#t2.)
#     * 2.1. [Age - Missing Values](#t2.1.)
#     * 2.2. [Cabin - Missing Values](#t2.2.)
#     * 2.3. [Embarked - Missing Values](#t2.3.)
#     * 2.4. [Final Adjustments to Data](#t2.4.)
#         * 2.4.1 [Additional Variables](#t2.4.1.)
# 3. [Exploratory Data Analysis](#t3.)
# 4. [Logistic Regression and Results](#t4.)
#     * 4.1. [Feature selection](#t4.1.)
#         * 4.1.1. [Recursive feature elimination](#t4.1.1.)
#         * 4.1.2. [Feature ranking with recursive feature elimination and cross-validation](#t4.1.2.)
#     * 4.2. [Review of model evaluation procedures](#t4.2.)
#         * 4.2.1. [Model evaluation based on simple train/test split using `train_test_split()`](#t4.2.1.)
#         * 4.2.2. [Model evaluation based on K-fold cross-validation using `cross_val_score()`](#t4.2.2.)
#         * 4.2.3. [Model evaluation based on K-fold cross-validation using `cross_validate()`](#t4.2.3.)
#     * 4.3. [GridSearchCV evaluating using multiple scorers simultaneously](#t4.3.)
#     * 4.4. [GridSearchCV evaluating using multiple scorers, RepeatedStratifiedKFold and pipeline for preprocessing simultaneously](#t4.4.)

# %% [markdown] _cell_guid="33c91cae-2ff8-45a6-b8cb-671619e9c933" _uuid="0a395fd25f20834b070ef55cb8987c8c1f9b55f9"
# <a id="t1."></a>
# # 1. Import Data & Python Packages

# %% _cell_guid="de05512e-6991-44df-9599-da92a7e459ac" _uuid="d8bdd5f0320e244e4702ed8ec1c2482b022c51cd"
import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
import seaborn as sns

sns.set(style="white")  #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

import warnings

warnings.simplefilter(action='ignore')

# %% _cell_guid="e0a17223-f682-45fc-89a5-667af9782bbe" _uuid="7964157913fbcff581fc1929eed487708e81ac9c"
# Read CSV train data file into DataFrame
train_df = pd.read_csv("input/train.csv")

# Read CSV test data file into DataFrame
test_df = pd.read_csv("input/test.csv")

# preview train data
train_df.head()

# %% _cell_guid="872d0de9-a873-4b60-b1ee-d557ee39d8a1" _uuid="d38222a64d4dfd1d1ee1a7ee1f58c4aa54560de3"
print('The number of samples into the train data is {}.'.format(
    train_df.shape[0]))

# %% _cell_guid="1d969b76-ea88-4d32-a58e-f22a070258bf" _uuid="bff38fcf31baf67493513c06f0c2f6e50576ff09"
# preview test data
test_df.head()

# %% _cell_guid="254dd074-e07e-49f2-9184-80046b10b481" _uuid="62de7ddd73fed8d88ccbe1ba79e59b8e596cbb13"
print('The number of samples into the test data is {}.'.format(
    test_df.shape[0]))

# %% [markdown] _cell_guid="4cd08f1e-9cb9-4d8e-99d0-3a9ce91b91c3" _uuid="2b1d45128663b9466fe9ac0059a13cfd4bd43657"
# <font color=red>  Note: there is no target variable into test data (i.e. "Survival" column is missing), so the goal is to predict this target using different machine learning algorithms such as logistic regression. </font>

# %% [markdown] _cell_guid="6578c0da-7bcf-433d-9f28-a66d8dfa6fa3" _uuid="8660e63a62c2fcdb4f7633380166438caf5edae9"
# <a id="t2."></a>
# # 2. Data Quality & Missing Value Assessment

# %% _cell_guid="29dddd33-d995-4b0f-92ea-a361b368cc42" _uuid="d4fe22ead7e187724ca6f3ba7ba0e6412ae0e874"
# check missing values in train data
train_df.isnull().sum()

# %% [markdown] _cell_guid="7776faeb-6a8f-4460-a367-4b087d2cc089" _uuid="696b428bd3ca49421f650665267ce7ca1b358814"
# <a id="t2.1."></a>
# ## 2.1.    Age - Missing Values

# %% _cell_guid="d4ee6559-6d0c-409d-9dca-1d105a4ccd8a" _uuid="129cf984d05d9ce97c54548145e65f9e4b9b0c37"
# percent of missing "Age"
print('Percent of missing "Age" records is %.2f%%' %
      ((train_df['Age'].isnull().sum() / train_df.shape[0]) * 100))

# %% [markdown] _cell_guid="951f7bb8-779c-4eac-85a2-3fdcfdcd293e" _uuid="c8fff460fb532a063f6944450809014ca831ca52"
# ~20% of entries for passenger age are missing. Let's see what the 'Age' variable looks like in general.

# %% _cell_guid="6d65fcfa-52bf-45ab-b959-64a32c1c1976" _uuid="c6fd60f15d5e803d4dffc89e782c6fbc72445a83"
ax = train_df["Age"].hist(bins=15,
                          density=True,
                          stacked=True,
                          color='teal',
                          alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10, 85)
plt.show()

# %% [markdown] _cell_guid="e62d6951-d968-43ba-aabf-add90524d042" _uuid="24c201948b9c8c8076ab01271a4790d9db9096b5"
# Since "Age" is (right) skewed, using the mean might give us biased results by filling in ages that are older than desired. To deal with this, we'll use the median to impute the missing values.

# %% _cell_guid="1d70c27b-1e4d-4d5e-8a39-c134389d436c" _uuid="4f13840d4f9bf1b4331523c99274aa0627485e6c"
# mean age
print('The mean of "Age" is %.2f' % (train_df["Age"].mean(skipna=True)))
# median age
print('The median of "Age" is %.2f' % (train_df["Age"].median(skipna=True)))

# %% [markdown] _cell_guid="dea7b01c-c8c1-401f-a336-36ee73de2222" _uuid="e1a08114e302ddc90266e5f065b3f0b5a200bc89"
# <a id="t2.2."></a>
# ## 2.2. Cabin - Missing Values

# %% _cell_guid="1a1ad808-0a63-43ac-b757-71195880ed4f" _uuid="1acbce9c6bc5d586dda3e47b7506067a85524e66"
# percent of missing "Cabin"
print('Percent of missing "Cabin" records is %.2f%%' %
      ((train_df['Cabin'].isnull().sum() / train_df.shape[0]) * 100))

# %% [markdown] _cell_guid="eda8c434-63ff-4875-8566-2e194c0d3f66" _uuid="b6e037c7ac5ec476516031a06b042d8b9999ba44"
# 77% of records are missing, which means that imputing information and using this variable for prediction is probably not wise.  We'll ignore this variable in our model.

# %% [markdown] _cell_guid="0e696cff-ca80-4cb5-862c-ee80f4b1ab1f" _uuid="d575319b1f528c7a153d8ab680282048cb163b14"
# <a id="t2.3."></a>
# ## 2.3. Embarked - Missing Values

# %% _cell_guid="f21c2b55-2126-439d-8b1d-e96dafc97d81" _uuid="92ab9e62fb62f2a0fb9972baf6ada444187540e6"
# percent of missing "Embarked"
print('Percent of missing "Embarked" records is %.2f%%' %
      ((train_df['Embarked'].isnull().sum() / train_df.shape[0]) * 100))

# %% [markdown] _cell_guid="d03a4187-c527-4f71-8260-0495f4523e9e" _uuid="dc97b80524057522f024d0ae6f1abe77cb994903"
# There are only 2 (0.22%) missing values for "Embarked", so we can just impute with the port where most people boarded.

# %% _cell_guid="22924bc4-5dfa-4df7-b0d0-de3ede9c58b7" _uuid="f2a915f45264f8a580de6cc382d96b370eb75730"
print(
    'Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):'
)
print(train_df['Embarked'].value_counts())
sns.countplot(x='Embarked', data=train_df, palette='Set2')
plt.show()

# %% _cell_guid="def67427-3257-4dce-872e-7f5b4202d18a" _uuid="c57a9f8a54efa382bc94b695c9664330d01709ea"
print('The most common boarding port of embarkation is %s.' %
      train_df['Embarked'].value_counts().idxmax())

# %% [markdown] _cell_guid="c4c55f99-ce99-44f9-b7a8-d4d623ae9295" _uuid="19cfaae8c484dcb1d00f69b2771e86dc249e9793"
# By far the most passengers boarded in Southhampton, so we'll impute those 2 NaN's w/ "S".

# %% [markdown] _cell_guid="684c308f-25ae-4039-9332-ddb58953a054" _uuid="3609e785d210d5a8110f7ce550e61007d066449b"
# <a id="t2.4."></a>
# ## 2.4. Final Adjustments to Data (Train & Test)

# %% [markdown] _cell_guid="b3025cdc-fe9f-43b6-bda1-e45c1f25e77c" _uuid="06d2762ccec3f11564870fe941fc9ac45d71662f"
# Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:
# * If "Age" is missing for a given row, I'll impute with 28 (median age).
# * If "Embarked" is missing for a riven row, I'll impute with "S" (the most common boarding port).
# * I'll ignore "Cabin" as a variable. There are too many missing values for imputation. Based on the information available, it appears that this value is associated with the passenger's class and fare paid.

# %% _cell_guid="bc0d7121-1008-4890-9043-07eba1524e15" _uuid="feeed4b6775f88edf5de12b0ee6ee73c16eba61d"
train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(),
                              inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)

# %% _cell_guid="0cfe1c08-71a6-493e-803d-db255af01697" _uuid="d6be29651bb903964e02d3a7bcc7033513eb76c9"
# check missing values in adjusted train data
train_data.isnull().sum()

# %% _cell_guid="10dcfe1b-34f1-4bd8-b937-5ae8daf4a378" _uuid="3ee37b1151416aeeec8ebd7b94bb0184aabc57cd"
# preview adjusted train data
train_data.head()

# %% _cell_guid="dda26046-b93b-49ee-a52e-35355ecb425c" _uuid="293aec20df86ef529d10ae1f051dfe921ba07b88"
plt.figure(figsize=(15, 8))
ax = train_df["Age"].hist(bins=15,
                          density=True,
                          stacked=True,
                          color='teal',
                          alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax = train_data["Age"].hist(bins=15,
                            density=True,
                            stacked=True,
                            color='orange',
                            alpha=0.5)
train_data["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10, 85)
plt.show()

# %% [markdown] _cell_guid="6925fcc2-977b-4369-85e1-77a9210326a7" _uuid="d8280757e6bc627821fb0540c87ccd6ca110f1e0"
# <a id="t2.4.1."></a>
# ## 2.4.1. Additional Variables

# %% [markdown] _cell_guid="5cf98f33-fdd5-4a16-b6bf-fa36bc8b84e0" _uuid="3bfdee842f11d27ca490f466c45ef9bf3673e7ae"
# According to the Kaggle data dictionary, both SibSp and Parch relate to traveling with family.  For simplicity's sake (and to account for possible multicollinearity), I'll combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone.

# %% _cell_guid="759c3c8e-8db6-41d9-a1a2-058a15b338a6" _uuid="d1f5815ba663f7e8cc17d7efcff73653af5b1bdb"
## Create categorical variable for traveling alone
train_data['TravelAlone'] = np.where(
    (train_data["SibSp"] + train_data["Parch"]) > 0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)

# %% [markdown] _cell_guid="e4a22367-b719-4204-952f-d2e9a3b8075e" _uuid="ca53796bf788bd3b015f1a79a97e050bafa2c770"
# I'll also create categorical variables for Passenger Class ("Pclass"), Gender ("Sex"), and Port Embarked ("Embarked").

# %% _cell_guid="f95361e8-2533-4731-a7ab-a99cf686ed50" _uuid="4494fcbf9faa90151e20042f74d73395fac3cc8e"
#create categorical variables and drop some variables
training = pd.get_dummies(train_data, columns=["Pclass", "Embarked", "Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()

# %% [markdown] _cell_guid="768cf074-6ecb-47eb-9b42-8b9079ffb811" _uuid="6a8e533e77c7f1f1a68d136119f93972447a31cf"
# ### Now, apply the same changes to the test data. <br>
# I will apply to same imputation for "Age" in the Test data as I did for my Training data (if missing, Age = 28).  <br> I'll also remove the "Cabin" variable from the test data, as I've decided not to include it in my analysis. <br> There were no missing values in the "Embarked" port variable. <br> I'll add the dummy variables to finalize the test set.  <br> Finally, I'll impute the 1 missing value for "Fare" with the median, 14.45.

# %% _cell_guid="501f9a53-881d-4440-9366-7aae67eb358b" _uuid="d80416a026d17ccac3bf793408dd5f4f1e17bf63"
test_df.isnull().sum()

# %% _cell_guid="8b9ef076-3669-4339-8d10-0d8783a92e07" _uuid="145675b90aa2befa533c640aaedd4bf8069b12d4"
test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone'] = np.where(
    (test_data["SibSp"] + test_data["Parch"]) > 0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass", "Embarked", "Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()

# %% [markdown] _cell_guid="1430d510-1c8d-4544-8009-3911fff7afbb" _uuid="4e26c19bf719b7086addc0e1981c00836a19f189"
# <a id="t3."></a>
# # 3. Exploratory Data Analysis

# %% [markdown] _cell_guid="2655428b-d69d-4c0f-85ff-e31ada8e37b9" _uuid="32e9c04a3281fb1aa8c77e1406c56cd820459202"
# <a id="t3.1."></a>
# ## 3.1. Exploration of Age

# %% _cell_guid="9f9ca9e5-50a0-4487-ba53-815dda90af1c" _uuid="790e8d7ca89d19e276b3398e299c42893a796b79"
plt.figure(figsize=(15, 8))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1],
                 color="darkturquoise",
                 shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0],
            color="lightcoral",
            shade=True)
plt.legend(['Survived', 'Died'])
plt.title(
    'Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10, 85)
plt.show()

# %% [markdown] _cell_guid="8e304d72-27f3-41cf-863f-63872f4c37df" _uuid="6c5625b454f5e01dd6b6d843d801851c14c64d1e"
# The age distribution for survivors and deceased is actually very similar.  One notable difference is that, of the survivors, a larger proportion were children.  The passengers evidently made an attempt to save children by giving them a place on the life rafts.

# %% _cell_guid="d2aa9f59-c433-4258-b8db-225b63a5eab6" _uuid="9cf1794d9db2fdc314c20ca97a76e9470e81a354"
plt.figure(figsize=(20, 8))
avg_survival_byage = final_train[["Age",
                                  "Survived"]].groupby(['Age'],
                                                       as_index=False).mean()
g = sns.barplot(x='Age',
                y='Survived',
                data=avg_survival_byage,
                color="LightSeaGreen")
plt.show()

# %% [markdown] _cell_guid="0b636440-ab38-46a8-8cc9-9421683d5c0b" _uuid="67051cf653243b3103c9f8015c501d89d92bd3bc"
# Considering the survival rate of passengers under 16, I'll also include another categorical variable in my dataset: "Minor"

# %% _cell_guid="1655b49b-b33f-4236-8b31-d995ef26c6f6" _uuid="8918defa6e17b83c700ea45357ebd67a3a22f02f"
final_train['IsMinor'] = np.where(final_train['Age'] <= 16, 1, 0)

final_test['IsMinor'] = np.where(final_test['Age'] <= 16, 1, 0)

# %% [markdown] _cell_guid="a643b196-91c6-4b12-9463-0f984fbfc91a" _uuid="337b3ced0c6423cf1d126f23a7e60c0181af6a47"
# <a id="t3.2."></a>
# ## 3.2. Exploration of Fare

# %% _cell_guid="9f31ffe1-7cd8-4169-b193-ed44e56d0bd4" _uuid="4a1c521f08460f6983eca0c4e01294fb7c86e4f9"
plt.figure(figsize=(15, 8))
ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1],
                 color="darkturquoise",
                 shade=True)
sns.kdeplot(final_train["Fare"][final_train.Survived == 0],
            color="lightcoral",
            shade=True)
plt.legend(['Survived', 'Died'])
plt.title(
    'Density Plot of Fare for Surviving Population and Deceased Population')
ax.set(xlabel='Fare')
plt.xlim(-20, 200)
plt.show()

# %% [markdown] _cell_guid="346b7322-a3e4-48df-bbb1-4d8ec7716f3f" _uuid="2717310b6c443d675c7342be0c2c18b265723273"
# As the distributions are clearly different for the fares of survivors vs. deceased, it's likely that this would be a significant predictor in our final model.  Passengers who paid lower fare appear to have been less likely to survive.  This is probably strongly correlated with Passenger Class, which we'll look at next.

# %% [markdown] _cell_guid="cf585311-4029-4be4-8af2-3eea8258801a" _uuid="4524affda51265ea23fa923e2ea7f93d7bb91875"
# <a id="t3.3."></a>
# ## 3.3. Exploration of Passenger Class

# %% _cell_guid="676548e8-6dd4-4180-800c-7b164acb3877" _uuid="08fd677214959e0b938a0f8a94b63ab548673ea5"
sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")
plt.show()

# %% [markdown] _cell_guid="193233f8-b220-4cae-aa0f-f822316d5623" _uuid="8ddb19191253a6e09dfcb0beff2b3690f1052d52"
# Unsurprisingly, being a first class passenger was safest.

# %% [markdown] _cell_guid="c59f8e8f-e8c2-40fb-b9c8-12dddd6d318f" _uuid="2fc06b75321946b721852f78431435f9ba5fef39"
# <a id="t3.4."></a>
# ## 3.4. Exploration of Embarked Port

# %% _cell_guid="6e5bec50-2f5e-433e-9130-c56956fddad3" _uuid="a9f0598701c7c5224eaa73dafa869af73beffe18"
sns.barplot('Embarked', 'Survived', data=train_df, color="teal")
plt.show()

# %% [markdown] _cell_guid="88d78820-35a5-48fd-a234-9f3ca3fca779" _uuid="2f6a0329cf0c7b771a707ec790efc065924e1ee2"
# Passengers who boarded in Cherbourg, France, appear to have the highest survival rate.  Passengers who boarded in Southhampton were marginally less likely to survive than those who boarded in Queenstown.  This is probably related to passenger class, or maybe even the order of room assignments (e.g. maybe earlier passengers were more likely to have rooms closer to deck). <br> It's also worth noting the size of the whiskers in these plots.  Because the number of passengers who boarded at Southhampton was highest, the confidence around the survival rate is the highest.  The whisker of the Queenstown plot includes the Southhampton average, as well as the lower bound of its whisker.  It's possible that Queenstown passengers were equally, or even more, ill-fated than their Southhampton counterparts.

# %% [markdown] _cell_guid="9e6dc87e-ba59-4004-8145-79709328fe27" _uuid="92bacce85a7dec5509217b9570bc2a2fea6a8452"
# <a id="t3.5."></a>
# ## 3.5. Exploration of Traveling Alone vs. With Family

# %% _cell_guid="67017a88-93d4-412b-9adf-8b4d1d9b9db0" _uuid="e0c3dc16292ef0bcabf0fc680d821ef654084ab4"
sns.barplot(x='TravelAlone',
            y='Survived',
            data=final_train,
            color="mediumturquoise")
plt.show()

# %% [markdown] _cell_guid="e9e68cef-5e74-46aa-8343-39afbbf00efe" _uuid="f160bd7399e024ae669d55f09caf6e7902768851"
# Individuals traveling without family were more likely to die in the disaster than those with family aboard. Given the era, it's likely that individuals traveling alone were likely male.

# %% [markdown] _cell_guid="201b4c9d-b9f0-4ae9-8580-0b4e24ee62be" _uuid="693c25c25f3590f0b027725471ddd74d56f154af"
# <a id="t3.6."></a>
# ## 3.6. Exploration of Gender Variable

# %% _cell_guid="7b416e59-8616-4a44-93e1-a8005eff78a9" _uuid="354794315925dff1e96229cc737eaf299aaea17a"
sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")
plt.show()

# %% [markdown] _cell_guid="490ed298-f0e4-466b-acc8-81280315e6a2" _uuid="80c02b9fe2151c443f189cbe44c9cacf7e5c44a4"
# This is a very obvious difference.  Clearly being female greatly increased your chances of survival.

# %% [markdown] _cell_guid="c833cbf5-74db-44ff-90fa-b600ff0a09d7" _uuid="39dbc095f99dcec6d25a7a4561e81bb641078622"
# <a id="t4."></a>
# # 4. Logistic Regression and Results

# %% [markdown] _cell_guid="b70cda8a-e8d9-44a6-b9f0-2b365fdf3428" _uuid="136cf9e02ea1ab48a397f534b491fb2d9dbb5684"
# <a id="t4.1."></a>
# ## 4.1. Feature selection
#
# <a id="t4.1.1."></a>
# ### 4.1.1. Recursive feature elimination
#
# Given an external estimator that assigns weights to features, recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a `coef_ attribute` or through a `feature_importances_` attribute. Then, the least important features are pruned from current set of features.That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
#
# References: <br>
# http://scikit-learn.org/stable/modules/feature_selection.html <br>

# %% _cell_guid="11a2a468-20df-40cd-a4ba-4ae7bd2fc403" _uuid="64befdf1182c2b4e845f488f5bfd0e19ce3dc17a"
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

cols = [
    "Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2", "Embarked_C",
    "Embarked_S", "Sex_male", "IsMinor"
]
X = final_train[cols]
y = final_train['Survived']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(model, n_features_to_select=8)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))

# %% [markdown] _cell_guid="29281bd5-b954-4f3d-87e3-7416b1ec8c6b" _uuid="626da3348b48ced3564e6e05bdb0c3b4bd1402e6"
# <a id="t4.1.2."></a>
# ### 4.1.2. Feature ranking with recursive feature elimination and cross-validation
#
# RFECV performs RFE in a cross-validation loop to find the optimal number or the best number of features. Hereafter a recursive feature elimination applied on logistic regression with automatic tuning of the number of features selected with cross-validation.

# %% _cell_guid="7239aa6f-7fd2-4b75-a387-f6624f1c338c" _uuid="53d79f38cfe33d75d6ff869a443b9a29c93b4cbd"
from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(),
              step=1,
              cv=10,
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# %% [markdown] _cell_guid="b1b3b56f-2f5f-47d6-9375-62c11e49ce79" _uuid="e9d52d5b182c0a01218982e844e53d5278e0d98a"
# As we see, eight variables were kept.

# %% _cell_guid="08986ec4-79ff-466b-b763-61bf84a0879b" _uuid="3f6950a7c24c629b72e17e54c556f3c183b3f779"
Selected_features = [
    'Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 'Embarked_S',
    'Sex_male', 'IsMinor'
]
X = final_train[Selected_features]

plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()

# %% [markdown] _cell_guid="a7455afe-9716-4189-b207-f1cc9facce12" _uuid="46b76691c5f109b17f805f233a5ad5ba900b353b"
# <a id="t4.2."></a>
# ## 4.2. Review of model evaluation procedures
#
# Motivation: Need a way to choose between machine learning models
# * Goal is to estimate likely performance of a model on out-of-sample data
#
# Initial idea: Train and test on the same data
# * But, maximizing training accuracy rewards overly complex models which overfit the training data
#
# Alternative idea: Train/test split
# * Split the dataset into two pieces, so that the model can be trained and tested on different data
# * Testing accuracy is a better estimate than training accuracy of out-of-sample performance
# * Problem with train/test split
#     * It provides a high variance estimate since changing which observations happen to be in the testing set can significantly change testing accuracy
#     * Testing accuracy can change a lot depending on a which observation happen to be in the testing set
#
# Reference: <br>
# http://www.ritchieng.com/machine-learning-cross-validation/ <br>

# %% [markdown] _cell_guid="b894002e-07cf-4d02-b708-a2ac387eed54" _uuid="e35125f8aa230d4875541aa4f6b5964d2f14a6a3"
# <a id="t4.2.1."></a>
# ### 4.2.1. Model evaluation based on simple train/test split using `train_test_split()` function

# %% _cell_guid="84233f59-f3c7-4ea0-884d-96f8ad4d5b10" _uuid="46336228eeb864bc82e6739768122579d1c9634c"
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# create X (features) and y (response)
X = final_train[Selected_features]
y = final_train['Survived']

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=2)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__ +
      " accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__ +
      " log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__ + " auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)
             )  # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr,
         tpr,
         color='coral',
         label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] +
      "guarantees a sensitivity of %.3f " % tpr[idx] +
      "and a specificity of %.3f" % (1 - fpr[idx]) +
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx]) * 100))

# %% [markdown] _cell_guid="6292c3f2-6be9-45e4-be33-1dca41e604a7" _uuid="ba0017b461cea0b8849746e76475598bbba7c9ce"
# <a id="t4.2.2."></a>
# ### 4.2.2. Model evaluation based on K-fold cross-validation using `cross_val_score()` function

# %% _cell_guid="32a611ae-b2b7-43e0-8fa8-3cc56e351bf6" _uuid="7f0aba7b861c3fa1748060b4733778851fb00a31"
# 10-fold cross-validation logistic regression
logreg = LogisticRegression()
# Use cross_val_score function
# We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data
# cv=10 for 10 folds
# scoring = {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric - althought they are many
scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')
print('K-fold cross-validation results:')
print(logreg.__class__.__name__ +
      " average accuracy is %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__ +
      " average log_loss is %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__ + " average auc is %2.3f" % scores_auc.mean())

# %% [markdown] _cell_guid="bf358561-2762-4b24-b33b-bfda3c5d725f" _uuid="9debd9d4281cf7893159aa2cb1f7993652d60228"
# <a id="t4.2.3."></a>
# ### 4.2.3. Model evaluation based on K-fold cross-validation using `cross_validate()` function

# %% _cell_guid="9ea95aac-00b2-413e-ab86-c6b8782c40ef" _uuid="90840c89d42e9284c9480a256dc259778c3a5b1b"
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'log_loss': 'neg_log_loss',
    'auc': 'roc_auc'
}

modelCV = LogisticRegression()

results = cross_validate(modelCV,
                         X,
                         y,
                         cv=10,
                         scoring=list(scoring.values()),
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__ + " average %s: %.3f (+/-%.3f)" %
          (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values(
          ))[sc]].mean() if list(scoring.values())[sc] == 'neg_log_loss' else
           results['test_%s' % list(scoring.values())[sc]].mean(),
           results['test_%s' % list(scoring.values())[sc]].std()))

# %% [markdown] _cell_guid="96fc0718-8678-44b5-ba1f-9155303a4ffe" _uuid="c29fd32bbe1fb2740587524d8ddbf08d85832d9c"
# <font color=bleu>What happens when we add the feature "Fare"?<font>

# %% _cell_guid="80b5d5c3-ead3-48f7-84eb-42c801e0370a" _uuid="0fdf4a257a49b0b47552a4ed307c86852c4ed271"
cols = [
    "Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2", "Embarked_C",
    "Embarked_S", "Sex_male", "IsMinor"
]
X = final_train[cols]

scoring = {
    'accuracy': 'accuracy',
    'log_loss': 'neg_log_loss',
    'auc': 'roc_auc'
}

modelCV = LogisticRegression()

results = cross_validate(modelCV,
                         final_train[cols],
                         y,
                         cv=10,
                         scoring=list(scoring.values()),
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__ + " average %s: %.3f (+/-%.3f)" %
          (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values(
          ))[sc]].mean() if list(scoring.values())[sc] == 'neg_log_loss' else
           results['test_%s' % list(scoring.values())[sc]].mean(),
           results['test_%s' % list(scoring.values())[sc]].std()))

# %% [markdown] _cell_guid="e6d3b029-550e-4142-8d9a-c9670b8cd529" _uuid="99b2a4c72f6c50fd11abda7a6463aca5154cacd7"
# <font color=red>We notice that the model is slightly deteriorated. The "Fare" variable does not carry any useful information. Its presence is just a noise for the logistic regression model.<font>

# %% [markdown] _cell_guid="f485ca4c-2172-4383-979e-21e78a66192c" _uuid="ec44fbaddbc23f03a8ac470391f41ce35f40cf72"
# <a id="t4.3."></a>
# ## 4.3. GridSearchCV evaluating using multiple scorers simultaneously

# %% _cell_guid="4a39cb49-b446-4ffa-88a2-e37b627eb5e5" _uuid="765695a9712d3fe1ecff10f17dcc077a80ed7682"
from sklearn.model_selection import GridSearchCV

X = final_train[Selected_features]

param_grid = {'C': np.arange(1e-05, 3, 0.1)}
scoring = {
    'Accuracy': 'accuracy',
    'AUC': 'roc_auc',
    'Log_loss': 'neg_log_loss'
}

gs = GridSearchCV(LogisticRegression(),
                  return_train_score=True,
                  param_grid=param_grid,
                  scoring=scoring,
                  cv=10,
                  refit='Accuracy')

gs.fit(X, y)
results = gs.cv_results_

print('=' * 20)
print("best params: " + str(gs.best_estimator_))
print("best params: " + str(gs.best_params_))
print('best score:', gs.best_score_)
print('=' * 20)

plt.figure(figsize=(10, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, param_grid['C'].max())
ax.set_ylim(0.35, 0.95)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (
            sample, scorer)] if scoring[scorer] == 'neg_log_loss' else results[
                'mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis,
                        sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0,
                        color=color)
        ax.plot(X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[
        scorer] == 'neg_log_loss' else results['mean_test_%s' %
                                               scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([
        X_axis[best_index],
    ] * 2, [0, best_score],
            linestyle='-.',
            color=color,
            marker='x',
            markeredgewidth=3,
            ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()

# %% [markdown] _cell_guid="4c32848c-e879-4ff0-9da4-7c572e80369e" _uuid="d5e88e25cf05dc5c2ad7dcb574a91115c186047e"
# <a id="t4.4."></a>
# ## 4.4. GridSearchCV evaluating using multiple scorers, RepeatedStratifiedKFold and pipeline for preprocessing simultaneously
#
# We can applied many tasks together for more in-depth evaluation like gridsearch using cross-validation based on k-folds repeated many times, that can be scaled or no with respect to many scorers and tunning on parameter for a given estimator!

# %% _cell_guid="a777c645-7413-4e1d-855a-0b416be1b48e" _uuid="514d0e3aabe57aaad38d10d6101dbdac5af89ca4"
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

#Define simple model
###############################################################################
C = np.arange(1e-05, 5.5, 0.1)
scoring = {
    'Accuracy': 'accuracy',
    'AUC': 'roc_auc',
    'Log_loss': 'neg_log_loss'
}
log_reg = LogisticRegression()

#Simple pre-processing estimators
###############################################################################
std_scale = StandardScaler(with_mean=False, with_std=False)
#std_scale = StandardScaler()

#Defining the CV method: Using the Repeated Stratified K Fold
###############################################################################

n_folds = 5
n_repeats = 5

rskfold = RepeatedStratifiedKFold(n_splits=n_folds,
                                  n_repeats=n_repeats,
                                  random_state=2)

#Creating simple pipeline and defining the gridsearch
###############################################################################

log_clf_pipe = Pipeline(steps=[('scale', std_scale), ('clf', log_reg)])

log_clf = GridSearchCV(estimator=log_clf_pipe,
                       cv=rskfold,
                       scoring=scoring,
                       return_train_score=True,
                       param_grid=dict(clf__C=C),
                       refit='Accuracy')

log_clf.fit(X, y)
results = log_clf.cv_results_

print('=' * 20)
print("best params: " + str(log_clf.best_estimator_))
print("best params: " + str(log_clf.best_params_))
print('best score:', log_clf.best_score_)
print('=' * 20)

plt.figure(figsize=(10, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, C.max())
ax.set_ylim(0.35, 0.95)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_clf__C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (
            sample, scorer)] if scoring[scorer] == 'neg_log_loss' else results[
                'mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis,
                        sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0,
                        color=color)
        ax.plot(X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[
        scorer] == 'neg_log_loss' else results['mean_test_%s' %
                                               scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([
        X_axis[best_index],
    ] * 2, [0, best_score],
            linestyle='-.',
            color=color,
            marker='x',
            markeredgewidth=3,
            ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()

# %% _cell_guid="e52c3bfb-4325-4c27-9f8e-5514dae799c2" _uuid="b3d12ca129d816d2434e74ed6574f829f826c3fc"
final_test['Survived'] = log_clf.predict(final_test[Selected_features])
final_test['PassengerId'] = test_df['PassengerId']

submission = final_test[['PassengerId', 'Survived']]

submission.to_csv("submission.csv", index=False)

submission.tail()
