# `DelayNet` — Delay Propagation in Transportation Networks

[//]: # ([![Dev]&#40;https://img.shields.io/badge/docs-dev-blue.svg&#41;]&#40;https://cbueth.github.io/DelayDynamics/&#41;)
[![Tests](https://github.com/cbueth/delaynet/actions/workflows/test.yml/badge.svg)](https://github.com/cbueth/delaynet/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/cbueth/delaynet/graph/badge.svg?token=G3MEQR5N1Y)](https://codecov.io/gh/cbueth/delaynet)
[![Lint](https://github.com/cbueth/delaynet/actions/workflows/lint.yml/badge.svg)](https://github.com/cbueth/delaynet/actions/workflows/lint.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python package to build networks from delay data.

---

## Set up

The environment to run the projects' code can be set up using the
`environment.yaml` by running:

```bash
mamba env create --file=environment.yml
```

This initializes a conda environment by the name `delay_net`, which can be
activated using `mamba activate delay_net`.
If you want to use `conda` or `micromamba`, just replace
`mamba` with the respective. For `micromamba`:

```bash
micromamba env create --file=environment.yml
```

Alternatively a version-less setup can be done by executing the following

```bash
mamba create -n delay_net -c conda-forge python=3.11 --file requirements.txt
mamba activate delay_net
mamba env export | grep -v "^prefix: " > environment.yml
```

which does not have explicit versions, but might resolve dependency issues. Using
`git diff environment.yml` the changes can be inspected.
`conda` and `micromamba` can be used analogously.

## Logging

The logging is done using the `logging` module. The logging level can be set in the
`setup.cfg` file. The logging level can be set to `DEBUG`, `INFO`, `WARNING`, `ERROR`
or `CRITICAL`. It defaults to `INFO` and a rotating file handler is set up to log
to `results/logs/delaynet.log`. The log file is rotated every megabyte, and the
last three log files are kept.

## Testing

The tests are specified using the `pytest` signature, see [`tests/`](tests/) folder, and
can be run using a test runner of choice.
A pipeline is set up, see [`.github/workflows/test.yml`](.github/workflows/lint.yml).