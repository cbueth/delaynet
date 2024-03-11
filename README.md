# `DelayNet` — Delay Propagation in Transportation Networks

[//]: # ([![Dev]&#40;https://img.shields.io/badge/docs-dev-blue.svg&#41;]&#40;https://cbueth.github.io/DelayDynamics/&#41;)
[![Tests](https://github.com/cbueth/delaynet/actions/workflows/test.yml/badge.svg)](https://github.com/cbueth/delaynet/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/cbueth/delaynet/graph/badge.svg?token=G3MEQR5N1Y)](https://codecov.io/gh/cbueth/delaynet)
[![Lint](https://github.com/cbueth/delaynet/actions/workflows/lint.yml/badge.svg)](https://github.com/cbueth/delaynet/actions/workflows/lint.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python package to build networks from delay data.

---

For details on how to use this package, see the
[Guide](https://cbueth.github.io/delaynet/guide/), [examples](examples/), or
the [Documentation](https://cbueth.github.io/delaynet/).

## Setup

This package can be installed from PyPI using pip:

```bash
pip install delaynet  # when public on PyPI
```

This will automatically install all the necessary dependencies as specified in the
`pyproject.toml` file. It is recommended to use a virtual environment, e.g. using
`conda`, `mamba` or `micromamba` (they can be used interchangeably).

```bash
micromamba create -n delay_net -c conda-forge python=3.11
micromamba activate delay_net
pip install delaynet  # or `micromamba install delaynet` when on conda-forge
```

## Development Setup

For development, we recommend using `micromamba` to create a virtual
environment and installing the package in editable mode.
After cloning the repository, navigate to the root folder and
create the environment with the wished python version and the dependencies.

```bash
micromamba create -n delay_net -c conda-forge python=3.11
micromamba activate delay_net
```
Using `pip` to install the package in editable mode will also install the
development dependencies.

```bash
pip install -e ".[all]"
```

To let `micromamba` handle the dependencies, use the `requirements.txt` file

```bash
micromamba install --file requirements.txt
pip install --no-build-isolation --no-deps -e .
```

Now, the package can be imported and used in the python environment, from anywhere on
the system, if the environment is activated.

### Testing

The tests are specified using the `pytest` signature, see [`tests/`](tests/) folder, and
can be run using a test runner of choice.
A pipeline is set up, see [`.github/workflows/test.yml`](.github/workflows/lint.yml).

### Linting

The code is linted using `pylint` and `black`. From the repository root, run:

```bash
pylint delaynet/
black delaynet/
```