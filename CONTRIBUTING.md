# Setting up a development environment

To install the code in development ("editable") mode inside a virtual environment:

```bash
(myenv) $ cd quaternions
(myenv) $ pip install --editable .[dev]
```

This will install quaternions, its requirements and the testing dependencies.

To run the tests:

```bash
(myenv) $ python -m unittest discover tests/
```

Or, alternatively, use `$ pytest`.
