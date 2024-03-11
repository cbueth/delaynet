---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell} ipython3
:tags: [remove-input]
# Import path of delaynet, from here one directory up
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
```

# Getting Started

This package can be installed from PyPI using pip:

```bash
pip install delaynet
```

This will automatically install all the necessary dependencies as specified in the
[`pyproject.toml`](https://github.com/cbueth/delaynet/blob/main/pyproject.toml) file.
It is recommended to use a virtual environment, e.g. using
[`conda`](https://conda.io/projects/conda/en/latest),
[`mamba`](https://mamba.readthedocs.io/en/latest) or
[`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
(they can be used interchangeably).

```bash
micromamba create -n delay_net -c conda-forge python=3.11
micromamba activate delay_net
pip install delaynet
```

## Usage

The package can be used as a library. The most common functions are exposed in the
top-level namespace, e.g. {py:func}`~delaynet.normalisation.normalise` and
{py:func}`~delaynet.connectivity.connectivity`. For example:

```python
import delaynet as dn

ts_norm = dn.normalise(ts, norm='Z-Score')
conn = dn.connectivity(ts1, ts2, metric='Granger Causality')
```

```{code-cell}
:tags: [remove-input, remove-output]
import delaynet as dn
```

To quickly find the string specifiers for these functions, use the
{py:func}`~delaynet.normalisation.show_norms` and
{py:func}`~delaynet.connectivity.show_connectivity_metrics` functions.
These are case-insensitive.
Each documentation of these methods can be found in a respective submodule,
{py:mod}`delaynet.norms` or {py:mod}`delaynet.connectivities`.

```{code-cell}
:tags: [scroll-output]
dn.show_norms()
dn.show_connectivity_metrics()
```

For more insight into the package, read the [Guide](guide/index.myst)
or the [API Reference](api/index.rst).

## Development Setup

For development, we recommend using
[`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
to create a virtual environment and installing the package in editable mode.
After cloning the repository, navigate to the root folder and
create the environment with the wished python version and the dependencies.

```bash
micromamba create -n delay_net -c conda-forge python=3.11
micromamba activate delay_net
```

Either way, install the package directly with pip in editable mode, including all
development dependencies, like so:

```bash
pip install -e ".[all]"
```

Or, to let `micromamba` handle the dependencies, use the
[`requirements.txt`](https://github.com/cbueth/delaynet/blob/main/requirements.txt)
file

```bash
micromamba install --file requirements.txt
pip install --no-build-isolation --no-deps -e .
```

Now, the package can be imported and used in the python environment, from anywhere on
the system, if the environment is activated.

### Testing

The tests are specified using the [`pytest`](https://docs.pytest.org/en/stable/)
signature, see [`tests/`](https://github.com/cbueth/delaynet/tree/main/tests) folder,
and can be run using a test runner of choice.
A pipeline is set up, see
[`.github/workflows/test.yml`](https://github.com/cbueth/delaynet/actions/workflows/test.yml).

### Linting

The code is linted using [`pylint`](https://pylint.pycqa.org/en/latest/index.html) and
[`black`](https://black.readthedocs.io/en/stable/). From the repository root, run:

```bash
pylint delaynet/
black delaynet/
```
