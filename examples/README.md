# Vanguard Example Notebooks

Vanguard contains a number of example notebooks, contained in the `examples/notebooks` folder. These are designed to showcase certain features of Vanguard within the context of a data science problem.  To run them, you will need to first install the requirements:

```shell
$ pip install -r requirements-notebook.txt --no-deps
```

If you are in a virtual environment, you can then run the following to add the `vanguard` kernel to Jupyter, which makes running the notebooks as frictionless as possible:

```shell
$ ipython kernel install --name vanguard --user
```

> **Warning**: Certain notebooks can take a long time to run, even on a GPU.  To see fully rendered examples, please visit the documentation.
