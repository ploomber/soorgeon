# Soorgeon

<p align="center">
  <a href="https://ploomber.io/community">Join our community</a>
  |
  <a href="https://www.getrevue.co/profile/ploomber">Newsletter</a>
  |
  <a href="mailto:contact@ploomber.io">Contact us</a>
  |
  <a href="https://ploomber.io/">Blog</a>
  |  
  <a href="https://www.ploomber.io">Website</a>
  |
  <a href="https://www.youtube.com/channel/UCaIS5BMlmeNQE4-Gn0xTDXQ">YouTube</a>
</p>


![header](_static/header.png)

Convert monolithic Jupyter notebooks into [Ploomber](https://github.com/ploomber/ploomber) pipelines.

https://user-images.githubusercontent.com/989250/150660392-559eca67-b630-4ef2-b660-4f5ddb5a8d65.mp4

[3-minute video tutorial](https://www.youtube.com/watch?v=EJecqsZBr3Q).

Try the interactive demo:

<p align="center">
  <a href="https://mybinder.org/v2/gh/ploomber/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fploomber%252Fprojects%26urlpath%3Dlab%252Ftree%252Fprojects%252Fguides/refactor%252FREADME.ipynb%26branch%3Dmaster"> <img src="_static/open-jupyterlab.svg" alt="Open JupyerLab"> </a>
</p>


*Note: Soorgeon is in alpha, [help us make it better](CONTRIBUTING.md).*

## Install

*Compatible with Python 3.7 and higher.*

```sh
pip install soorgeon
```

## Usage

### [Optional] Testing if the notebook runs

Before refactoring, you can optionally test if the original notebook or script runs without exceptions:

```sh
# works with ipynb files
soorgeon test path/to/notebook.ipynb

# and notebooks in percent format
soorgeon test path/to/notebook.py
```

Optionally, set the path to the output notebook:

```sh
soorgeon test path/to/notebook.ipynb path/to/output.ipynb

soorgeon test path/to/notebook.py path/to/output.ipynb
```

### Refactoring

To refactor your notebook:

```sh
# refactor notebook
soorgeon refactor nb.ipynb

# all variables with the df prefix are stored in csv files
soorgeon refactor nb.ipynb --df-format csv
# all variables with the df prefix are stored in parquet files
soorgeon refactor nb.ipynb --df-format parquet

# store task output in 'some-directory' (if missing, this defaults to 'output')
soorgeon refactor nb.ipynb --product-prefix some-directory

# generate tasks in .py format
soorgeon refactor nb.ipynb --file-format py

# use alternative serializer (cloudpickle or dill) if notebook 
# contains variables that cannot be serialized using pickle 
soorgeon refactor nb.ipynb --serializer cloudpickle
soorgeon refactor nb.ipynb --serializer dill
```

To learn more, check out our [guide](doc/guide.md).

### Cleaning

Soorgeon has a `clean` command that applies
[black](https://github.com/psf/black) <!--and [isort](https://github.com/PyCQA/isort)-->for `.ipynb` and `.py` files:

```
soorgeon clean path/to/notebook.ipynb
```

or

```
soorgeon clean path/to/script.py
```

## Linting

Soorgeon has a `lint` command that can apply [flake8]:

```
soorgeon lint path/to/notebook.ipynb
```

or

```
soorgeon lint path/to/script.py
```

## Examples

```sh
git clone https://github.com/ploomber/soorgeon
```

Exploratory data analysis notebook:

```sh
cd soorgeon/examples/exploratory
soorgeon refactor nb.ipynb

# to run the pipeline
pip install -r requirements.txt
ploomber build
```

Machine learning notebook:

```sh
cd soorgeon/examples/machine-learning
soorgeon refactor nb.ipynb

# to run the pipeline
pip install -r requirements.txt
ploomber build
```

To learn more, check out our [guide](doc/guide.md).

## Community

* [Join us on Slack](https://ploomber.io/community)
* [Newsletter](https://www.getrevue.co/profile/ploomber)
* [YouTube](https://www.youtube.com/channel/UCaIS5BMlmeNQE4-Gn0xTDXQ)
* [Contact the development team](mailto:contact@ploomber.io)


## About Ploomber

Ploomber is a big community of data enthusiasts pushing the boundaries of Data Science and Machine Learning tooling.

Whatever your skillset is, you can contribute to our mission. So whether you're a beginner or an experienced professional, you're welcome to join us on this journey!

[Click here to know how you can contribute to Ploomber.](https://github.com/ploomber/contributing/blob/main/README.md)


## Telemetry

We collect anonymous statistics to understand and improve usage. For details, [see here](https://docs.ploomber.io/en/latest/community/user-stats.html)
