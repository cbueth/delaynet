<div style="text-align: center; max-width: 700px; margin: 0 auto;">
  <a href="https://delaynet.readthedocs.io/">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/cbueth/delaynet/refs/heads/main/docs/_static/dn_banner.png">
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cbueth/delaynet/refs/heads/main/docs/_static/dn_banner_dark.png">
      <img src="https://raw.githubusercontent.com/cbueth/delaynet/refs/heads/main/docs/_static/dn_banner.png" style="max-width: 80%; height: auto;" alt="delaynet logo">
    </picture>
  </a>
</div>

<div align="center">

<a href="">[![Documentation](https://readthedocs.org/projects/delaynet/badge/)](https://delaynet.readthedocs.io/)</a>
<a href="">[![PyPI Version](https://badge.fury.io/py/delaynet.svg)](https://pypi.org/project/delaynet/)</a>
<a href="">[![Python Version](https://img.shields.io/pypi/pyversions/delaynet)](https://pypi.org/project/delaynet/)</a>
<a href="">[![Anaconda Version](https://anaconda.org/conda-forge/delaynet/badges/version.svg)](https://anaconda.org/conda-forge/delaynet)</a>

</div>

<div align="center">

<a href="">[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)</a>
<a href="">[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-1.2-4baaaa.svg)](CODE_OF_CONDUCT.md)</a>

</div>

<div align="center">

<a href="">[![pipeline status](https://gitlab.ifisc.uib-csic.es/carlson/delaynet/badges/main/pipeline.svg)](https://gitlab.ifisc.uib-csic.es/carlson/delaynet/-/commits/main)</a>
<a href="">[![coverage report](https://gitlab.ifisc.uib-csic.es/carlson/delaynet/badges/main/coverage.svg)](https://gitlab.ifisc.uib-csic.es/carlson/delaynet/-/jobs)</a>

</div>

Python package to reconstruct and analyse delay functional networks from time series.
It provides tools for data preparation and detrending, multiple connectivity measures
(e.g. Granger causality, transfer entropy, correlations), optimal-lag network
reconstruction, and network analysis.

---

For details on how to use this package, see the
[Guide](https://delaynet.readthedocs.io/en/latest/guide/) or
the [Documentation](https://delaynet.readthedocs.io/en/latest/).

## Setup

This package can be installed from PyPI using pip:

```bash
pip install delaynet  # when public on PyPI
```

This will automatically install all the necessary dependencies as specified in the
`pyproject.toml` file. It is recommended to use a virtual environment, e.g., using
`conda`, `mamba` or `micromamba` (they can be used interchangeably).

```bash
micromamba create -n delay_net -c conda-forge python
micromamba activate delay_net
pip install delaynet  # or `micromamba install delaynet` when on conda-forge
```

### Quickstart

```python
import numpy as np
import delaynet as dn

# Generate toy data: 5 nodes, 300 time points
rng = np.random.default_rng(1520)
data = rng.standard_normal((300, 5))

# Compute a connectivity p-value and lag for one pair
pval, lag = dn.connectivity(data[:, 0], data[:, 1], metric="gc", lag_steps=10)
print(f"GC p-value={pval:.3g}, best lag={lag}")

# Reconstruct a delay network (p-value matrix and lag matrix)
weights, lags = dn.reconstruct_network(data, connectivity_measure="gc", lag_steps=5)
print(weights.shape, lags.shape)
```

## Development Setup

For development, we recommend using [`uv`](https://docs.astral.sh/uv/)  or `micromamba`
to create a virtual environment.
After cloning the repository, navigate to the root folder and
create the environment.
When using `uv`, the environment can be created with the following command:

```bash
uv sync
```

Or, if you prefer to use `micromamba`,
with the desired Python version and the dependencies.

```bash
micromamba create -n delay_net -c conda-forge -f requirements.txt
micromamba activate delay_net
```

Either way, using `pip` to install the package in editable mode will also install the
development dependencies.

```bash
pip install -e ".[all]"
```

Or, to let `micromamba` handle the dependencies, use the `requirements.txt` file

```bash
micromamba install --file requirements.txt
pip install --no-build-isolation --no-deps -e .
```

Now, the package can be imported and used in the python environment, from anywhere on
the system if the environment is activated.

## Set up Jupyter kernel

If you want to use `delaynet` with its environment `delay_net` in Jupyter, run:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=delay_net
```

This allows you to run Jupyter with the kernel `delay_net` (Kernel > Change Kernel >
im_env)

## Acknowledgments

This project has received funding from the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation programme (grant agreement No
851255).
This work was partially supported by the María de Maeztu project CEX2021-001164-M funded
by the MICIU/AEI/10.13039/501100011033 and FEDER, EU.
