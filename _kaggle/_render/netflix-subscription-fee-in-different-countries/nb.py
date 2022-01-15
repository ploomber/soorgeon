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
# # Which Countries Pay The Most and Least for Netflix in 2021?

# %%
#import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# %% [markdown]
# # Data Extraction

# %%
#read dataset
df = pd.read_csv(
    '../input/netflix-subscription-price-in-different-countries/Netflix subscription fee Dec-2021.csv'
)

# %%
#show head of dataset
df.head()

# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %%
#check rows and columns of dataset
df.shape

# %%
#check all columns
df.columns

# %%
#rename attribute columns of dataset
df = df.rename(
    columns={
        'Country_code': 'Country_Code',
        'Total Library Size': 'Library_Size',
        'No. of TV Shows': 'No_TV_Shows',
        'No. of Movies': 'No_Movies',
        'Cost Per Month - Basic ($)': 'Basic_Cost_Per_Month',
        'Cost Per Month - Standard ($)': 'Standard_Cost_Per_Month',
        'Cost Per Month - Premium ($)': 'Premium_Cost_Per_Month'
    })
df.head()

# %%
#check type of dataset
df.dtypes

# %%
#check missing value of dataset
df.isnull().sum()

# %%
#describe all columns
df.describe(include='object')

# %%
#check correlation of each variable
df.corr()

# %% [markdown]
# ## heatmap

# %%
#visualize correlation of each variable using pearson correlation
sns.heatmap(df.corr(), vmax=0.9, linewidths=0.9, cmap='YlGnBu')
plt.title('Pearson Correlation', fontsize=15, pad=12)
plt.show()

# %%
#check unique of country code column
df['Country_Code'].unique()

# %%
#check number of country code
df['Country_Code'].value_counts()

# %%
#check unique of country column
df['Country'].unique()

# %%
#check number of country
df['Country'].value_counts()

# %% [markdown]
# ## outliers

# %%
#visualize the outlier of each variable
chart = df.boxplot()
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.ylabel('Count', fontsize=12)
plt.show()

print('Maximum of library size :', df['Library_Size'].max())
print('Minimum of library size :', df['Library_Size'].min())
print('Median of library size :', df['Library_Size'].median())
print('Average of library size :', df['Library_Size'].mean())
print('Total of library size :', df['Library_Size'].sum())
print('\n')
print('Maximum of number TV shows :', df['No_TV_Shows'].max())
print('Minimum of number TV shows :', df['No_TV_Shows'].min())
print('Median of number TV shows :', df['No_TV_Shows'].median())
print('Average of number TV shows :', df['No_TV_Shows'].mean())
print('Total of number TV shows :', df['No_TV_Shows'].sum())
print('\n')
print('Maximum of number movies :', df['No_Movies'].max())
print('Minimum of number movies :', df['No_Movies'].min())
print('Median of number movies :', df['No_Movies'].median())
print('Average of number movies :', df['No_Movies'].mean())
print('Total of number movies :', df['No_Movies'].sum())
print('\n')
print('Maximum of basic cost per month :', df['Basic_Cost_Per_Month'].max())
print('Minimum of basic cost per month :', df['Basic_Cost_Per_Month'].min())
print('Median of basic cost per month :', df['Basic_Cost_Per_Month'].median())
print('Average of basic cost per month :', df['Basic_Cost_Per_Month'].mean())
print('Total of basic cost per month :', df['Basic_Cost_Per_Month'].sum())
print('\n')
print('Maximum of standard cost per month :',
      df['Standard_Cost_Per_Month'].max())
print('Minimum of standard cost per month :',
      df['Standard_Cost_Per_Month'].min())
print('Median of standard cost per month :',
      df['Standard_Cost_Per_Month'].median())
print('Average of standard cost per month :',
      df['Standard_Cost_Per_Month'].mean())
print('Total of standard cost per month :',
      df['Standard_Cost_Per_Month'].sum())
print('\n')
print('Maximum of premium cost per month :',
      df['Premium_Cost_Per_Month'].max())
print('Minimum of premium cost per month :',
      df['Premium_Cost_Per_Month'].min())
print('Median of premium cost per month :',
      df['Premium_Cost_Per_Month'].median())
print('Average of premium cost per month :',
      df['Premium_Cost_Per_Month'].mean())
print('Total of premium cost per month :', df['Premium_Cost_Per_Month'].sum())

# %% [markdown]
# ## analyze

# %%
#analyze of library size under 5195 based on country and country code
df[df['Library_Size'] < 5195.0][['Country_Code', 'Country', 'Library_Size']]

# %%
#analyze of library size over 5195 based on country and country code
df[df['Library_Size'] > 5195.0][['Country_Code', 'Country', 'Library_Size']]

# %%
#analyze of number TV shows under 3512 based on country and country code
df[df['No_TV_Shows'] < 3512.0][['Country_Code', 'Country', 'No_TV_Shows']]

# %%
#analyze of number TV shows over 3512 based on country and country code
df[df['No_TV_Shows'] > 3512.0][['Country_Code', 'Country', 'No_TV_Shows']]

# %%
#analyze of number movies under 1841 based on country and country code
df[df['No_Movies'] < 1841.0][['Country_Code', 'Country', 'No_Movies']]

# %%
#analyze of number movies over 1841 based on country and country code
df[df['No_Movies'] > 1841.0][['Country_Code', 'Country', 'No_Movies']]

# %%
#analyze of basic cost per month under 8.99 based on country and country code
df[df['Basic_Cost_Per_Month'] < 8.99][[
    'Country_Code', 'Country', 'Basic_Cost_Per_Month'
]]

# %%
#analyze of basic cost per month over 8.99 based on country and country code
df[df['Basic_Cost_Per_Month'] > 8.99][[
    'Country_Code', 'Country', 'Basic_Cost_Per_Month'
]]

# %%
#analyze of standard cost per month under 11.49 based on country and country code
df[df['Standard_Cost_Per_Month'] < 11.49][[
    'Country_Code', 'Country', 'Standard_Cost_Per_Month'
]]

# %%
#analyze of standard cost per month over 11.49 based on country and country code
df[df['Standard_Cost_Per_Month'] > 11.49][[
    'Country_Code', 'Country', 'Standard_Cost_Per_Month'
]]

# %%
#analyze of premium cost per month under 14.45 based on country and country code
df[df['Premium_Cost_Per_Month'] < 14.45][[
    'Country_Code', 'Country', 'Premium_Cost_Per_Month'
]]

# %%
#analyze of premium cost per month over 14.45 based on country and country code
df[df['Premium_Cost_Per_Month'] > 14.45][[
    'Country_Code', 'Country', 'Premium_Cost_Per_Month'
]]

# %% [markdown]
# ## top 20 countries

# %%
#visualize top 20 of country based on total basic cost per month
plt.figure(figsize=(10, 5))
top_20_country = df['Country'][:20]
chart = df.groupby(top_20_country)['Basic_Cost_Per_Month'].sum().sort_values(
    ascending=False).plot(kind='bar', color='maroon')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title('Top 20 of Country based on Total Basic Cost Per Month',
          fontsize=15,
          pad=12)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Basic Cost Per Month ($)', fontsize=12)
plt.show()

# %% [markdown]
# ## top 20 standard cost

# %%
#visualize top 20 of country based on total standard cost per month
plt.figure(figsize=(10, 5))
chart = df.groupby(top_20_country)['Standard_Cost_Per_Month'].sum(
).sort_values(ascending=False).plot(kind='bar', color='lightseagreen')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title('Top 20 of Country based on Total Standard Cost Per Month',
          fontsize=15,
          pad=12)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Standard Cost Per Month ($)', fontsize=12)
plt.show()

# %% [markdown]
# ## top 20 premium cost

# %%
#visualize top 20 of country based on total premium cost per month
plt.figure(figsize=(10, 5))
chart = df.groupby(top_20_country)['Premium_Cost_Per_Month'].sum().sort_values(
    ascending=False).plot(kind='bar', color='peru')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title('Top 20 of Country based on Total Premium Cost Per Month',
          fontsize=15,
          pad=12)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Premium Cost Per Month ($)', fontsize=12)
plt.show()

# %% [markdown]
# ## report

# %%
#profile report of dataset
ProfileReport(df)
