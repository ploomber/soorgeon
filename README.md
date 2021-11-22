# soorgeon

Convert monolithic Jupyter notebooks into mainaintable pipelines.

## Examples

```sh
cd examples
soorgeon refactor exploratory/nb.py
```

```sh
cd examples
soorgeon refactor machine-learning/nb.py
```

## Known limitations

* If a function depend on global variables, manual editing is required. [#12](https://github.com/ploomber/soorgeon/issues/12).
* Variables required from one task to the next one should be pickable (i.e., `pickle.dumps(obj)` should work)