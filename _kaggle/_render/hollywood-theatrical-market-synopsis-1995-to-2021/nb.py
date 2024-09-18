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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import cufflinks as cf
from scipy import stats

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
cf.go_offline()
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## **Importing Data**

# %%
PopularCreativeTypes = pd.read_csv(
    "/kaggle/input/hollywood-theatrical-market-synopsis-1995-to-2021/PopularCreativeTypes.csv"
)
HighestGrossers = pd.read_csv(
    "/kaggle/input/hollywood-theatrical-market-synopsis-1995-to-2021/HighestGrossers.csv"
)
Data_Annual_Ticket = pd.read_csv(
    "/kaggle/input/hollywood-theatrical-market-synopsis-1995-to-2021/AnnualTicketSales.csv",
    thousands=",",
)

# %% [markdown]
# ## **Cleaning and Tiding Data**

# %%
Data_Annual_Ticket["TICKETS SOLD"] = Data_Annual_Ticket["TICKETS SOLD"].replace(",", "")

Data_Annual_Ticket["TOTAL BOX OFFICE"] = Data_Annual_Ticket[
    "TOTAL BOX OFFICE"
].str.replace(",", "")
Data_Annual_Ticket["TOTAL BOX OFFICE"] = Data_Annual_Ticket[
    "TOTAL BOX OFFICE"
].str.replace("$", "")

Data_Annual_Ticket["TOTAL INFLATION ADJUSTED BOX OFFICE"] = Data_Annual_Ticket[
    "TOTAL INFLATION ADJUSTED BOX OFFICE"
].str.replace(",", "")
Data_Annual_Ticket["TOTAL INFLATION ADJUSTED BOX OFFICE"] = Data_Annual_Ticket[
    "TOTAL INFLATION ADJUSTED BOX OFFICE"
].str.replace("$", "")

Data_Annual_Ticket["AVERAGE TICKET PRICE"] = Data_Annual_Ticket[
    "AVERAGE TICKET PRICE"
].str.replace("$", "")

Data_Annual_Ticket = Data_Annual_Ticket.drop(labels="Unnamed: 5", axis=1)

# %%
Data_Annual_Ticket.head(5)

# %% [markdown]
# ## **Changing the type of Data(object to float)**

# %%
Data_Annual_Ticket["TICKETS SOLD"] = Data_Annual_Ticket["TICKETS SOLD"].astype(float)
Data_Annual_Ticket["TOTAL BOX OFFICE"] = Data_Annual_Ticket["TOTAL BOX OFFICE"].astype(
    float
)

# %% [markdown]
# ## **Using bar chart to illustrate the total box office each year**

# %%
px.bar(
    Data_Annual_Ticket,
    x="YEAR",
    y="TOTAL BOX OFFICE",
    title="Total Box Office vs. Year",
)

# %% [markdown]
# ## **Calculating the total box office if last two years were normal years (*using linear regression*)**

# %%
x = list(range(0, (2020 - 1995)))
y = list(Data_Annual_Ticket["TOTAL BOX OFFICE"])
y.reverse()
y.pop()
y.pop()
slope, intercept, r, p, std_err = stats.linregress(x, y)
x1 = list(range(0, (2022 - 1995)))
y1 = [slope * x + intercept for x in x1]
y1.reverse()
Data_Annual_Ticket["TOTAL BOX OFFICE WITHOUT COVID"] = y1
Data_Annual_Ticket["Diff"] = (
    Data_Annual_Ticket["TOTAL BOX OFFICE WITHOUT COVID"]
    - Data_Annual_Ticket["TOTAL BOX OFFICE"]
)

# %% [markdown]
# ## **Illustrate the difference between total box office with covid and without covid**

# %%
px.line(
    Data_Annual_Ticket,
    x="YEAR",
    y=["TOTAL BOX OFFICE", "TOTAL BOX OFFICE WITHOUT COVID"],
    labels={"YEAR": "Years", "value": "Total Sale"},
    title="TOTAL BOX OFFICE vs TOTAL BOX OFFICE WITHOUT COVID",
)

# %% [markdown]
# ## **Calculate that how much does covid-19 affect on last two years**

# %%
px.bar(
    Data_Annual_Ticket,
    x="YEAR",
    y="Diff",
    labels={"YEAR": "Year", "Diff": "Financial Loss"},
    title="Financial Loss (just last two years are important)",
    barmode="group",
)

# %% [markdown]
#
## # **How much does covid-19 affect on total box ofice in last two years in percent?**

# %%
Data_Annual_Ticket["Percentage of Financial Loss"] = (
    (
        Data_Annual_Ticket["TOTAL BOX OFFICE WITHOUT COVID"]
        - Data_Annual_Ticket["TOTAL BOX OFFICE"]
    )
    / Data_Annual_Ticket["TOTAL BOX OFFICE WITHOUT COVID"]
    * 100
)

px.bar(
    Data_Annual_Ticket,
    x="YEAR",
    y="Percentage of Financial Loss",
    labels={
        "YEAR": "Year",
        "Percentage of Financial Loss": "Percentage of Financial Loss %",
    },
    title="Financial Loss % (just last two years are important) ",
)

# %% [markdown]
# ## **Now Visualizing the Highest Grossers**

# %% [markdown]
# ## **Cleaning and Tiding Data**

# %%
HighestGrossers["TOTAL IN 2019 DOLLARS"] = HighestGrossers[
    "TOTAL IN 2019 DOLLARS"
].str.replace(",", "")
HighestGrossers["TOTAL IN 2019 DOLLARS"] = HighestGrossers[
    "TOTAL IN 2019 DOLLARS"
].str.replace("$", "")

HighestGrossers["TICKETS SOLD"] = HighestGrossers["TICKETS SOLD"].str.replace(",", "")

HighestGrossers["TOTAL IN 2019 DOLLARS"] = HighestGrossers[
    "TOTAL IN 2019 DOLLARS"
].astype(float)
HighestGrossers["TICKETS SOLD"] = HighestGrossers["TICKETS SOLD"].astype(float)

# %%
HighestGrossers.head(5)

# %% [markdown]
# ## **Because of the inflation we just used TOTAL IN 2019 DOLLARS column**

# %% [markdown]
# ## **We use pie chart to illustrate the percentage of different thing**

# %%
px.pie(
    HighestGrossers,
    values="TOTAL IN 2019 DOLLARS",
    names="DISTRIBUTOR",
    title="Percentage of Each Distributors in Total Ticket Sale",
    color_discrete_sequence=px.colors.sequential.RdBu,
    height=600,
)

# %%
px.pie(
    HighestGrossers,
    values="TOTAL IN 2019 DOLLARS",
    names="MPAA RATING",
    title="Percentage of Each MPAA Rating in Total Ticket Sale",
    color_discrete_sequence=px.colors.sequential.RdBu,
    height=600,
)

# %% [markdown]
# ## **using bar chart to state the sum of total ticket sale each distributor and each genre**

# %%
df_g = (
    HighestGrossers.groupby(by=["DISTRIBUTOR", "GENRE"])["TICKETS SOLD"]
    .sum()
    .unstack()
    .iplot(kind="bar")
)

# %% [markdown]
# ## **using bar chart to state the count of total ticket sale each distributor and each genre**

# %%
df_g = (
    HighestGrossers.groupby(by=["DISTRIBUTOR", "GENRE"])["TICKETS SOLD"]
    .count()
    .unstack()
    .iplot(kind="bar")
)

# %% [markdown]
# ## **doing the same thing to the MPAA rating**
#

# %%
df_g = (
    HighestGrossers.groupby(by=["DISTRIBUTOR", "MPAA RATING"])["TICKETS SOLD"]
    .sum()
    .unstack()
    .iplot(kind="bar")
)

# %%
df_g = (
    HighestGrossers.groupby(by=["DISTRIBUTOR", "MPAA RATING"])["TICKETS SOLD"]
    .count()
    .unstack()
    .iplot(kind="bar")
)

# %% [markdown]
# ## **now visualising the Popular Creative Types**

# %%
PopularCreativeTypes.head(5)

# %%
PopularCreativeTypes["TOTAL GROSS"] = PopularCreativeTypes["TOTAL GROSS"].str.replace(
    ",", ""
)
PopularCreativeTypes["TOTAL GROSS"] = PopularCreativeTypes["TOTAL GROSS"].str.replace(
    "$", ""
)

PopularCreativeTypes["AVERAGE GROSS"] = PopularCreativeTypes[
    "AVERAGE GROSS"
].str.replace(",", "")
PopularCreativeTypes["AVERAGE GROSS"] = PopularCreativeTypes[
    "AVERAGE GROSS"
].str.replace("$", "")

PopularCreativeTypes["MARKET SHARE"] = PopularCreativeTypes["MARKET SHARE"].str.replace(
    "%", ""
)

PopularCreativeTypes["MOVIES"] = PopularCreativeTypes["MOVIES"].str.replace(",", "")

# %%
PopularCreativeTypes = PopularCreativeTypes.drop(index=9, axis=0)

# %%
PopularCreativeTypes["MOVIES"] = PopularCreativeTypes["MOVIES"].astype(float)
PopularCreativeTypes["TOTAL GROSS"] = PopularCreativeTypes["TOTAL GROSS"].astype(float)
PopularCreativeTypes["AVERAGE GROSS"] = PopularCreativeTypes["AVERAGE GROSS"].astype(
    float
)
PopularCreativeTypes["MARKET SHARE"] = PopularCreativeTypes["MARKET SHARE"].astype(
    float
)

# %%
px.pie(
    PopularCreativeTypes,
    values="TOTAL GROSS",
    names="CREATIVE TYPES",
    title="Percentage of Creative Types in Total Gross",
    color_discrete_sequence=px.colors.sequential.RdBu,
    height=600,
)

# %%
px.bar(
    PopularCreativeTypes,
    x="TOTAL GROSS",
    y="CREATIVE TYPES",
    title="Total Gross of Different type",
)

# %%
px.pie(
    PopularCreativeTypes,
    values="AVERAGE GROSS",
    names="CREATIVE TYPES",
    title="Percentage of Creative Types in Average Gross",
    color_discrete_sequence=px.colors.sequential.RdBu,
    height=600,
)

# %%
px.bar(
    PopularCreativeTypes,
    x="AVERAGE GROSS",
    y="CREATIVE TYPES",
    title="Average Gross in Different type",
)

# %%
px.pie(
    PopularCreativeTypes,
    values="MOVIES",
    names="CREATIVE TYPES",
    title="Percentage of Number of Muvies in Each Types",
    color_discrete_sequence=px.colors.sequential.RdBu,
    height=600,
)

# %%
px.bar(
    PopularCreativeTypes,
    x="MOVIES",
    y="CREATIVE TYPES",
    title="Number of Muvies in Different type",
)

# %%
