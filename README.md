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

*Note: Soorgeon is in alpha, [help us make it better](CONTRIBUTING.md).*

## Install

```sh
pip install soorgeon
```

## Examples

```sh
git clone https://github.com/ploomber/soorgeon
```

Exploratory notebook:

```sh
cd examples/exploratory
soorgeon refactor nb.ipynb
```

Machine learning:

```sh
cd examples/machine-learning
soorgeon refactor nb.ipynb
```

## Known limitations

* If a function depends on global variables, [manual editing is required](doc/fn-global.md).
* Variables required from one task to the next one should be pickable (i.e., `pickle.dumps(obj)` should work)


## Community

* [Join us on Slack](https://ploomber.io/community)
* [Newsletter](https://www.getrevue.co/profile/ploomber)
* [YouTube](https://www.youtube.com/channel/UCaIS5BMlmeNQE4-Gn0xTDXQ)
* [Contact the development team](mailto:contact@ploomber.io)
