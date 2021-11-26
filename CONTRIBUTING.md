# Contributing to Soorgeon

Hi! Thanks for considering a contribution to Soorgeon. We're building Soorgeon to help data scientists convert their monolithic notebooks into maintainable pipelines. Our command-line interface takes a notebook as an input and generates a [Ploomber pipeline](https://github.com/ploomber/ploomber) as output ([see demo here](https://www.youtube.com/watch?v=EJecqsZBr3Q)), and we need your help to ensure our tool is robust to real-world code (which is messy!).

We're downloading publicly available notebooks from Kaggle and testing if our tools can successfully refactor them.

This guide explains what the process looks like: from finding a candidate notebook to merging your changes. Hence, whenever we publish a new Soorgeon version, we'll test with all the contributed notebooks.

## Adding new test notebooks

### 1. Find a candidate notebook

[Look for notebooks in Kaggle](https://www.kaggle.com/code) that run fast (ideally, <1 minute), use small datasets (<20 MB), have lots of code (the longer, the better), and have executed recently (no more than three months ago).

Here's is a sample notebook that has all those characteristics: [kaggle.com/yuyougnchan/look-at-this-note-feature-engineering-is-easy](https://www.kaggle.com/yuyougnchan/look-at-this-note-feature-engineering-is-easy/notebook).

### 2. Open an issue to suggest a notebook

[Open an issue](https://github.com/ploomber/soorgeon/issues/new?title=Notebook%20suggestion) and share the URL with us, we'll take a quick look and let you know what we think.

### 3. Configure development environment

If we move forward, you can setup the development environment with:

```python
pip install ".[dev]"
```

*Note: We recommmend you run the command above in a virtual environment.*

### 4. Configure Kaggle CLI

You must have an account in Kaggle to continue, once you create one, [follow the instructions](https://github.com/Kaggle/kaggle-api#api-credentials) to configure the CLI client.

### 5. Download the notebook file `.ipynb`

Download the notebook with the following command:

```sh
python -m soorgeon._kaggle notebook user/notebook-name

# example
python -m soorgeon._kaggle notebook yuyougnchan/look-at-this-note-feature-engineering-is-easy
```

Note that the command above converts the notebook (`.ipynb`) to `.py` using `%%` cell separators. We prefer `.py` over `.ipynb` since they play better with git.

### 6. Download data

If you go to the data section in the notebook, you'll see a list (right side) of input datasets (look for the label `Input (X.X MB)`). Sometimes, authors may include many datasets, but the notebook may only use a few of them, so please check in the notebook contents which ones are actually used, we want to download as little data as possible to make testing fast.

Our example notebook takes us to this URL: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

We know this is a competition because the URL has the form: `kaggle.com/c/{name}`

#### Downloading a competitions dataset

To download a competition dataset, execute the following:

```sh
# Note: make sure to execute this in the _kaggle/{notebook-name}/ directory

python -m soorgeon._kaggle competition {name}

# example
python -m soorgeon._kaggle competition house-prices-advanced-regression-techniques
```

#### Downloading a user's dataset

Other notebooks use datasets that are not part of a competition. For example, [this notebook](https://www.kaggle.com/karnikakapoor/customer-segmentation-clustering), uses [this dataset](https://www.kaggle.com/imakash3011/customer-personality-analysis).

The URL is different, it has the format: `kaggle.com/{user}/{dataset}`

To download a dataset like that:

```sh
# Note: make sure to execute this in the _kaggle/{notebook-name}/ directory
python -m soorgeon._kaggle dataset user/notebook-name

# example
python -m soorgeon._kaggle dataset imakash3011/customer-personality-analysis
```

#### Final layout

Your layout should look like this:

```txt
_kaggle/
    {notebook-name}/
        nb.py
        input/
            data.csv
```

### 7. Notebook edits

The `nb.py` file may contain paths to files that are different from our setup,
so locate all the relevant lines (e.g., `df = pd.read_csv('/path/to/data.csv')`) and add
a relative path to the `input/` directory (e.g. `df = pd.read_csv('inputs/data.csv')`).

If you find any calls to pip like: `! pip install {package}`, remove them.

### 8. Test the notebook

Test it:

```sh
python -m soorgeon._kaggle test nb.py
```

A few things may go wrong, so you may have to do some edits to `nb.py`.

#### If missing dependencies

Add a `requirements.txt` under `_kaggle/{notebook-name}` and add all the
dependencies:

```txt
# requirements.txt
scikit-learn
pandas
matplotlib
```

If the notebook is old, you may encounter problems if the API for a specific library changed since the notebook ran. We recommend using notebooks that were executed recently because fixing these API incompatibility issues requires a trial and error process of looking at the library's changelog and figuring out either what version to use or how to fix the code. Hence, it works with the latest version.

If you encounter issues like this, let us know by adding a comment in the issue you opened in Step 2.

### 9. Running Soorgeon

Let's now check if ``sorgeon`` can handle the notebook:

```sh
soorgeon refactor nb.py
```

If the command above throws a `Looks like the following functions are using global variables...` error, [click here](doc/fn-global.md) to see fixing instructions.

Add a comment on the issue you created in `Step 2` if the command throws a different error or if you cannot fix the global variables issue. Please include the entire error traceback in the Github's issue.

### 10. Testing the generated pipeline

To ensure that the generate pipeline works, execute the following commands:

```
ploomber status
ploomber plot
ploomber build
```

Add a comment on the issue you created in `Step 2` if any command throws an error. Please include the entire error traceback in the Github's issue.

### 11. Registering the notebook

Add a new entry to `_kaggle/index.yaml`:

(if using a dataset from a competition)

```yaml
- url: https://www.kaggle.com/{user}/{notebook-name}
  data: https://www.kaggle.com/c/{competition-name}
```

(if using a user's dataset)

```yaml
- url: https://www.kaggle.com/{user}/{notebook-name}
  data: https://www.kaggle.com/{user-another}/{dataset-name}
```

Then, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) to merge your changes.


Thanks a lot for helping us make Soorgeon better! Happy notebook refactoring!