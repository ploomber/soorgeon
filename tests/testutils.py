import jupytext
import parso


def get_first_leaf_with_value(code, value):
    leaf = parso.parse(code).get_first_leaf()

    while leaf:
        if leaf.value == value:
            return leaf

        leaf = leaf.get_next_leaf()

    raise ValueError(f"could not find leaf with value {value}")


def _read(nb_str):
    return jupytext.reads(nb_str, fmt="py:light")


exploratory = """# # Exploratory data analysis
#
# ## Load

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

df = load_iris(as_frame=True)['data']

# ## Clean

df = df[df['petal length (cm)'] > 2]

# ## Plot

sns.histplot(df['petal length (cm)'])
"""

mixed = """# # Cell 0

1 + 1 # Cell 1

# ## Cell 2

2 + 2 # Cell 3

# ## Cell 4
"""
