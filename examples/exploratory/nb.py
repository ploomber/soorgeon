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
# # Some exploratory data analysis
#
# ## Load

# %%
import seaborn as sns
from sklearn.datasets import load_iris

# %%
df = load_iris(as_frame=True)['data']

# %% [markdown]
# ## Clean

# %%
df

# %%
df.shape

# %%
df = df[df['petal length (cm)'] > 2]

# %%
df.shape

# %% [markdown]
# ## Plot

# %%
sns.histplot(df['petal length (cm)'])
