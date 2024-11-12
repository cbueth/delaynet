# `DelayNet` — Delay Propagation in Transportation Networks

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://carlson.pages.ifisc.uib-csic.es/delaynet/)
[![pipeline status](https://gitlab.ifisc.uib-csic.es/carlson/delaynet/badges/main/pipeline.svg)](https://gitlab.ifisc.uib-csic.es/carlson/delaynet/-/pipelines?page=1&scope=all&ref=main)
[![coverage report](https://gitlab.ifisc.uib-csic.es/carlson/delaynet/badges/main/coverage.svg)](https://gitlab.ifisc.uib-csic.es/carlson/delaynet/-/commits/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-1.2-4baaaa.svg)](CODE_OF_CONDUCT.md)

Python package to build networks from delay data.

---

For details on how to use this package, see the
[Guide](https://carlson.pages.ifisc.uib-csic.es/delaynet/guide/) or
the [Documentation](https://carlson.pages.ifisc.uib-csic.es/delaynet/).

## Setup

This package can be installed from PyPI using pip:

```bash
pip install delaynet  # when public on PyPI
```

This will automatically install all the necessary dependencies as specified in the
`pyproject.toml` file. It is recommended to use a virtual environment, e.g. using
`conda`, `mamba` or `micromamba` (they can be used interchangeably).

```bash
micromamba create -n delay_net -c conda-forge python=3.12
micromamba activate delay_net
pip install delaynet  # or `micromamba install delaynet` when on conda-forge
```

## Development Setup

For development, we recommend using `micromamba` to create a virtual
environment and installing the package in editable mode.
After cloning the repository, navigate to the root folder and
create the environment with the wished python version and the dependencies.

```bash
micromamba create -n delay_net -c conda-forge python=3.12
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

Finally, the `infomeasure` package must be installed manually, as it is also still under development, see its [Development Setup](https://carlson.pages.ifisc.uib-csic.es/infomeasure/getting_started/#development-setup).

Now, the package can be imported and used in the python environment, from anywhere on
the system, if the environment is activated.

## Set up Jupyter kernel

If you want to use `delaynet` with its environment `delay_net` in Jupyter, run:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=delay_net
```

This allows you to run Jupyter with the kernel `delay_net` (Kernel > Change Kernel >
im_env)
