---
file_format: mystnb
kernelspec:
  name: python3
---

(getting_started)=
# Getting Started

```{warning}
Use {ref}`Development Setup` until public release.
```

This package can be [installed from PyPI](https://pypi.org/project/delaynet/) using pip:

```bash
pip install delaynet
```

This will automatically install all the necessary dependencies as specified in the
[`pyproject.toml`](https://github.com/cbueth/delaynet/blob/main/pyproject.toml) file.
It is recommended to use a virtual environment, e.g., using
[`conda`](https://conda.io/projects/conda/en/latest),
[`mamba`](https://mamba.readthedocs.io/en/latest) or
[`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
(they can be used interchangeably).
`delaynet` can be installed from
the [`conda-forge`](https://anaconda.org/conda-forge/delaynet) channel.

```bash
conda create -n delay_net -c conda-forge python=3.13
conda activate delay_net
conda install -c conda-forge delaynet
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

(dev_setup)=
## Development Setup

For development, we recommend using [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
to create a virtual environment (`conda` or `mamba` also work)
and installing the package in editable mode.
After cloning the repository, navigate to the root folder and
create the environment with the desired python version and the dependencies.

```bash
micromamba create -n delay_net -c conda-forge python=3.13
micromamba activate delay_net
```

To let `micromamba` handle the dependencies, use the `requirements` files

```bash
micromamba install -f requirements.txt
pip install --no-build-isolation --no-deps -e .
```

Alternatively, if you prefer to use `pip` instead of `micromamba`,
installing the package in editable mode will also install the development dependencies.

```bash
pip install -e ".[all]"
```

Now, the package can be imported and used in the python environment, from anywhere on
the system, if the environment is activated.
For new changes, the repository only needs to be updated, but the package does not need
to be reinstalled.

### Testing

The package can be tested using pytest and coverage.
To run the tests, execute the following command:

```bash
pytest --cov delaynet/ tests/
```

### Pre-Commit

The project uses `pre-commit` for code formatting and linting.
They are installed with the development dependencies.
To install and run the pre-commit hooks, use:

```bash
pre-commit run --all-files
```
