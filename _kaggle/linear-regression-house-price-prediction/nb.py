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
# # 📈 Linear Regression with Python
#
# > Linear Regression is the simplest algorithm in machine learning, it can be trained in different ways. In this notebook we will cover the following linear algorithms:
#
# > 1. Linear Regression
# > 2. Robust Regression
# > 3. Ridge Regression
# > 4. LASSO Regression
# > 5. Elastic Net
# > 6. Polynomial Regression
# > 7. Stochastic Gradient Descent
# > 8. Artificial Neaural Networks
#
# # 💾 Data
#
# > We are going to use the `USA_Housing` dataset. Since house price is a continues variable, this is a regression problem. The data contains the following columns:
#
# > * '`Avg. Area Income`': Avg. Income of residents of the city house is located in.
# > * '`Avg. Area House Age`': Avg Age of Houses in same city
# > * '`Avg. Area Number of Rooms`': Avg Number of Rooms for Houses in same city
# > * '`Avg. Area Number of Bedrooms`': Avg Number of Bedrooms for Houses in same city
# > * '`Area Population`': Population of city hou  se is located in
# > * '`Price`': Price that the house sold at
# > * '`Address`': Address for the house
#

# %% _kg_hide-input=true _kg_hide-output=true

# %% [markdown]
# # 📤 Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas

# sns.set_style("whitegrid")
# plt.style.use("fivethirtyeight")

# %% [markdown]
# ## 💾 Check out the Data

# %%
USAhousing = pd.read_csv('input/USA_Housing.csv')
USAhousing.head()

# %%
USAhousing.info()

# %%
USAhousing.describe()

# %%
USAhousing.columns

# %% [markdown]
# # 📊 Exploratory Data Analysis (EDA)
#
# Let's create some simple plots to check out the data!

# %%
sns.pairplot(USAhousing)

# %%
USAhousing.hvplot.hist(by='Price', subplots=False, width=1000)

# %%
USAhousing.hvplot.hist("Price")

# %%
USAhousing.hvplot.scatter(x='Avg. Area House Age', y='Price')

# %%
USAhousing.hvplot.scatter(x='Avg. Area Income', y='Price')

# %%
USAhousing.columns

# %%
sns.heatmap(USAhousing.corr(), annot=True)

# %% [markdown]
# # 📈 Training a Linear Regression Model
#
# > Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.
#
# ## X and y arrays

# %%
X = USAhousing[[
    'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms', 'Area Population'
]]
y = USAhousing['Price']

# %% [markdown]
# ## 🧱 Train Test Split
#
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

# %%
from sklearn import metrics
from sklearn.model_selection import cross_val_score


def cross_val(model, X, y):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()


def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')


def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# %% [markdown]
# # 📦 Preparing Data For Linear Regression
# > Linear regression is been studied at great length, and there is a lot of literature on how your data must be structured to make best use of the model.
#
# > As such, there is a lot of sophistication when talking about these requirements and expectations which can be intimidating. In practice, you can uses these rules more as rules of thumb when using Ordinary Least Squares Regression, the most common implementation of linear regression.
#
# > Try different preparations of your data using these heuristics and see what works best for your problem.
# - **Linear Assumption.** Linear regression assumes that the relationship between your input and output is linear. It does not support anything else. This may be obvious, but it is good to remember when you have a lot of attributes. You may need to transform data to make the relationship linear (e.g. log transform for an exponential relationship).
# - **Remove Noise.** Linear regression assumes that your input and output variables are not noisy. Consider using data cleaning operations that let you better expose and clarify the signal in your data. This is most important for the output variable and you want to remove outliers in the output variable (y) if possible.
# - **Remove Collinearity.** Linear regression will over-fit your data when you have highly correlated input variables. Consider calculating pairwise correlations for your input data and removing the most correlated.
# - **Gaussian Distributions.** Linear regression will make more reliable predictions if your input and output variables have a Gaussian distribution. You may get some benefit using transforms (e.g. log or BoxCox) on you variables to make their distribution more Gaussian looking.
# - **Rescale Inputs:** Linear regression will often make more reliable predictions if you rescale input variables using standardization or normalization.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('std_scalar', StandardScaler())])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# %% [markdown]
# # ✔️ Linear Regression

# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train, y_train)

# %% [markdown]
# ## ✔️ Model Evaluation
#
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# %%
# print the intercept
print(lin_reg.intercept_)

# %%
coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df

# %% [markdown]
# > Interpreting the coefficients:
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52**.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164883.28**.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$122368.67**.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$2233.80**.
# - Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.15**.
#
# Does this make sense? Probably not because I made up this data.

# %% [markdown]
# ## ✔️ Predictions from our Model
#
# Let's grab predictions off our test set and see how well it did!

# %%
pred = lin_reg.predict(X_test)

# %%
pd.DataFrame({
    'True Values': y_test,
    'Predicted Values': pred
}).hvplot.scatter(x='True Values', y='Predicted Values')

# %% [markdown]
# **Residual Histogram**

# %%
pd.DataFrame({'Error Values': (y_test - pred)}).hvplot.kde()

# %% [markdown]
# ## ✔️ Regression Evaluation Metrics
#
#
# Here are three common evaluation metrics for regression problems:
#
# > - **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
#
# > - **Mean Squared Error** (MSE) is the mean of the squared errors:
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
#
# > - **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
#
# > 📌 Comparing these metrics:
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
#
# > All of these are **loss functions**, because we want to minimize them.

# %%
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df = pd.DataFrame(
    data=[[
        "Linear Regression", *evaluate(y_test, test_pred),
        cross_val(LinearRegression(), X=X, y=y)
    ]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df

# %% [markdown]
# # ✔️ Robust Regression
#
# > Robust regression is a form of regression analysis designed to overcome some limitations of traditional parametric and non-parametric methods. Robust regression methods are designed to be not overly affected by violations of assumptions by the underlying data-generating process.
#
# > One instance in which robust estimation should be considered is when there is a strong suspicion of `heteroscedasticity`.
#
# > A common situation in which robust estimation is used occurs when the data contain outliers. In the presence of outliers that do not come from the same data-generating process as the rest of the data, least squares estimation is inefficient and can be biased. Because the least squares predictions are dragged towards the outliers, and because the variance of the estimates is artificially inflated, the result is that outliers can be masked. (In many situations, including some areas of geostatistics and medical statistics, it is precisely the outliers that are of interest.)

# %% [markdown]
# ## Random Sample Consensus - RANSAC
#
# > Random sample consensus (`RANSAC`) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates. Therefore, it also can be interpreted as an outlier detection method.
#
# > A basic assumption is that the data consists of "inliers", i.e., data whose distribution can be explained by some set of model parameters, though may be subject to noise, and "outliers" which are data that do not fit the model. The outliers can come, for example, from extreme values of the noise or from erroneous measurements or incorrect hypotheses about the interpretation of data. RANSAC also assumes that, given a (usually small) set of inliers, there exists a procedure which can estimate the parameters of a model that optimally explains or fits this data.

# %%
from sklearn.linear_model import RANSACRegressor

model = RANSACRegressor(base_estimator=LinearRegression(), max_trials=100)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[[
        "Robust Regression", *evaluate(y_test, test_pred),
        cross_val(RANSACRegressor(), X=X, y=y)
    ]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # ✔️ Ridge Regression
#
# > Source: [scikit-learn](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
#
# > Ridge regression addresses some of the problems of **Ordinary Least Squares** by imposing a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares,
#
# $$\min_{w}\big|\big|Xw-y\big|\big|^2_2+\alpha\big|\big|w\big|\big|^2_2$$
#
# > $\alpha>=0$ is a complexity parameter that controls the amount of shrinkage: the larger the value of $\alpha$, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.
#
# > Ridge regression is an L2 penalized model. Add the squared sum of the weights to the least-squares cost function.
# ***

# %%
from sklearn.linear_model import Ridge

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[[
        "Ridge Regression", *evaluate(y_test, test_pred),
        cross_val(Ridge(), X=X, y=y)
    ]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # ✔️ LASSO Regression
#
# > A linear model that estimates sparse coefficients.
#
# > Mathematically, it consists of a linear model trained with $\ell_1$ prior as regularizer. The objective function to minimize is:
#
# $$\min_{w}\frac{1}{2n_{samples}} \big|\big|Xw - y\big|\big|_2^2 + \alpha \big|\big|w\big|\big|_1$$
#
# > The lasso estimate thus solves the minimization of the least-squares penalty with $\alpha \big|\big|w\big|\big|_1$ added, where $\alpha$ is a constant and $\big|\big|w\big|\big|_1$ is the $\ell_1-norm$ of the parameter vector.
# ***

# %%
from sklearn.linear_model import Lasso

model = Lasso(
    alpha=0.1,
    precompute=True,
    #               warm_start=True,
    positive=True,
    selection='random',
    random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[[
        "Lasso Regression", *evaluate(y_test, test_pred),
        cross_val(Lasso(), X=X, y=y)
    ]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # ✔️ Elastic Net
#
# > A linear regression model trained with L1 and L2 prior as regularizer.
#
# > This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge.
#
# > Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
#
# > A practical advantage of trading-off between Lasso and Ridge is it allows Elastic-Net to inherit some of Ridge’s stability under rotation.
#
# > The objective function to minimize is in this case
#
# $$\min_{w}{\frac{1}{2n_{samples}} \big|\big|X w - y\big|\big|_2 ^ 2 + \alpha \rho \big|\big|w\big|\big|_1 +
# \frac{\alpha(1-\rho)}{2} \big|\big|w\big|\big|_2 ^ 2}$$
# ***

# %%
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1,
                   l1_ratio=0.9,
                   selection='random',
                   random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[[
        "Elastic Net Regression", *evaluate(y_test, test_pred),
        cross_val(ElasticNet(), X=X, y=y)
    ]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # ✔️ Polynomial Regression
# > Source: [scikit-learn](http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
#
# ***
#
# > One common pattern within machine learning is to use linear models trained on nonlinear functions of the data. This approach maintains the generally fast performance of linear methods, while allowing them to fit a much wider range of data.
#
# > For example, a simple linear regression can be extended by constructing polynomial features from the coefficients. In the standard linear regression case, you might have a model that looks like this for two-dimensional data:
#
# $$\hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2$$
#
# > If we want to fit a paraboloid to the data instead of a plane, we can combine the features in second-order polynomials, so that the model looks like this:
#
# $$\hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2$$
#
# > The (sometimes surprising) observation is that this is still a linear model: to see this, imagine creating a new variable
#
# $$z = [x_1, x_2, x_1 x_2, x_1^2, x_2^2]$$
#
# > With this re-labeling of the data, our problem can be written
#
# $$\hat{y}(w, x) = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3 + w_4 z_4 + w_5 z_5$$
#
# > We see that the resulting polynomial regression is in the same class of linear models we’d considered above (i.e. the model is linear in w) and can be solved by the same techniques. By considering linear fits within a higher-dimensional space built with these basis functions, the model has the flexibility to fit a much broader range of data.
# ***

# %%
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

X_train_2_d = poly_reg.fit_transform(X_train)
X_test_2_d = poly_reg.transform(X_test)

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train_2_d, y_train)

test_pred = lin_reg.predict(X_test_2_d)
train_pred = lin_reg.predict(X_train_2_d)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[["Polynomail Regression", *evaluate(y_test, test_pred), 0]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # ✔️ Stochastic Gradient Descent
#
# > Gradient Descent is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Sescent is to tweak parameters iteratively in order to minimize a cost function. Gradient Descent measures the local gradient of the error function with regards to the parameters vector, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum.

# %%
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter_no_change=250,
                       penalty=None,
                       eta0=0.0001,
                       max_iter=100000)
sgd_reg.fit(X_train, y_train)

test_pred = sgd_reg.predict(X_test)
train_pred = sgd_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[["Stochastic Gradient Descent", *evaluate(y_test, test_pred), 0]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # ✔️ Artficial Neural Network

# %% _kg_hide-output=true
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()

model.add(Dense(X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer=Adam(0.00001), loss='mse')

r = model.fit(X_train,
              y_train,
              validation_data=(X_test, y_test),
              batch_size=1,
              epochs=1)

# %%
pd.DataFrame({
    'True Values': y_test,
    'Predicted Values': pred
}).hvplot.scatter(x='True Values', y='Predicted Values')

# %%
pd.DataFrame(r.history)

# %%
pd.DataFrame(r.history).hvplot.line(y=['loss', 'val_loss'])

# %%
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[["Artficial Neural Network", *evaluate(y_test, test_pred), 0]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # ✔️ Random Forest Regressor

# %%
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=1000)
rf_reg.fit(X_train, y_train)

test_pred = rf_reg.predict(X_test)
train_pred = rf_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[["Random Forest Regressor", *evaluate(y_test, test_pred), 0]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # ✔️ Support Vector Machine

# %%
from sklearn.svm import SVR

svm_reg = SVR(kernel='rbf', C=1000000, epsilon=0.001)
svm_reg.fit(X_train, y_train)

test_pred = svm_reg.predict(X_test)
train_pred = svm_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
results_df_2 = pd.DataFrame(
    data=[["SVM Regressor", *evaluate(y_test, test_pred), 0]],
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# # 📊 Models Comparison

# %%
results_df.set_index('Model', inplace=True)
results_df['R2 Square'].plot(kind='barh', figsize=(12, 8))

# %% [markdown]
# # 📝 Summary
# In this notebook you discovered the linear regression algorithm for machine learning.
#
# You covered a lot of ground including:
# > - The common linear regression models (Ridge, Lasso, ElasticNet, ...).
# > - The representation used by the model.
# > - Learning algorithms used to estimate the coefficients in the model.
# > - Rules of thumb to consider when preparing data for use with linear regression.
# > - How to evaluate a linear regression model.
#
#
# # 🔗 References:
# - [Scikit-learn library](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
# - [Linear Regression for Machine Learning by Jason Brownlee PhD](https://machinelearningmastery.com/linear-regression-for-machine-learning/)
