# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predicting median house value
#
# ## Load

# %%
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
plt.style.use('ggplot')
mpl.rcParams['figure.figsize'] = (12, 8)

# %%
ca_housing = datasets.fetch_california_housing(as_frame=True)
df = ca_housing['frame']

# %%
df.head()

# %% [markdown]
# ## Clean

# %%
sns.histplot(df.HouseAge)

# %%
# let's say we're only interested in newer homes, so we define this filtering
# rule
df = df[df.HouseAge <= 30]

# %%
sns.histplot(x=df.AveBedrms)

# %%
sns.boxplot(x=df.AveBedrms)

# %%
# let's also remove big houses
df = df[df.AveBedrms <= 4]

# %%
# distribution of our target variable
sns.histplot(df.MedHouseVal)

# %% [markdown]
# ## Train test split

# %%
from sklearn.model_selection import train_test_split  # noqa

# %%
X = df.drop('MedHouseVal', axis='columns')
y = df.MedHouseVal

# %%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)

# %% [markdown]
# ## Linear regression

# %%
# TODO: define a function here and use it in different sections

# %%
from sklearn.linear_model import LinearRegression  # noqa

# %%
lr = LinearRegression()

# %%
lr.fit(X_train, y_train)

# %%
y_pred = lr.predict(X_test)

# %%
sns.scatterplot(x=y_test, y=y_pred)

# %% [markdown]
# ## Random Forest Regressor

# %%
from sklearn.ensemble import RandomForestRegressor  # noqa

# %%
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
sns.scatterplot(x=y_test, y=y_pred)
