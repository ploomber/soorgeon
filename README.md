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

Convert monolithic Jupyter notebooks into [Ploomber](https://github.com/ploomber/ploomber) pipelines. [3-minute demo](https://www.youtube.com/watch?v=EJecqsZBr3Q).

Try the interactive demo:

<p align="center">
  <a href="https://mybinder.org/v2/gh/ploomber/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fploomber%252Fprojects%26urlpath%3Dlab%252Ftree%252Fprojects%252Fguides/refactor%252FREADME.ipynb%26branch%3Dmaster"> <img src="_static/open-jupyterlab.svg" alt="Open JupyerLab"> </a>
</p>


*Note: Soorgeon is in alpha, [help us make it better](CONTRIBUTING.md).*

## Install

```sh
pip install soorgeon
```

## Examples

```sh
git clone https://github.com/ploomber/soorgeon
```

Exploratory daya analysis notebook:

```sh
cd examples/exploratory
soorgeon refactor nb.ipynb

# to run the pipeline
pip install ploomber
ploomber build
```

Machine learning notebook:

```sh
cd examples/machine-learning
soorgeon refactor nb.ipynb

# to run the pipeline
pip install ploomber
ploomber build
```

To learn more, check out our [guide](doc/guide.md).

## Community

* [Join us on Slack](https://ploomber.io/community)
* [Newsletter](https://www.getrevue.co/profile/ploomber)
* [YouTube](https://www.youtube.com/channel/UCaIS5BMlmeNQE4-Gn0xTDXQ)
* [Contact the development team](mailto:contact@ploomber.io)
