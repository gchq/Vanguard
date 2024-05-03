# Vanguard Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to generate HTML documentation. Notebooks in the examples directory (above) are converted to HTML and included in the docs.

## Building the Docs

To build the docs, first install the documentation requirements:

```shell
$ python -m pip install -r requirements-docs.txt --no-deps
```

Then run the following command:

```shell
$ python -m sphinx -b html -aE docs/source docs/build
```

The built documentation files can then be found in `docs/build`.


## Checking External Links

To check external links are correctly resolved, run the following:

```shell
$ python -m sphinx -b linkcheck docs/source docs/build
```
