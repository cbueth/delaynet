---
sd_hide_title: true
site:
  options:
    hide_toc: true
---

(delaynet_docs)=

# Overview

:::{image} _static/dn_banner.png
:width: 550
:align: center
:class: only-light
:alt: delaynet logo
:target: .
:::

:::{image} _static/dn_banner_dark.png
:width: 550
:align: center
:class: only-dark
:alt: delaynet logo
:target: .
:::

```{eval-rst}
.. raw:: html

   <div style="height: 10px;"></div>
   <div style="text-align: center;">
     <a href="https://pypi.org/project/delaynet/" style="margin: 0 10px; display: inline-block;">
       <img src="https://badge.fury.io/py/delaynet.svg" alt="PyPI version" />
     </a>
     <!-- arXiv badge will be added once the preprint is available
     <a href="https://arxiv.org/abs/0000.12345" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/badge/arXiv-0000.12345-b31b1b.svg" alt="arXiv Pre-print" />
     </a>
     <a href="https://doi.org/10.5281/zenodo.15241810" style="margin: 0 10px; display: inline-block;">
       <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15241810.svg" alt="Zenodo Project" />
     </a>
     -->
     <a href="https://anaconda.org/conda-forge/delaynet" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/conda/vn/conda-forge/delaynet.svg" alt="Conda version" />
     </a>
     <a href="https://pypi.org/project/delaynet/" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/pypi/pyversions/delaynet" alt="Python version" />
     </a>
     <a href="https://pypi.org/project/delaynet/" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/pypi/l/delaynet" alt="License" />
     </a>
    </div>
   <div style="height: 20px;"></div>

```

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {material-regular}`rocket;2em` Getting Started
:link: getting_started
:link-type: ref

How to install this package and run the first calculation.\
Start your endeavour here!

+++
{ref}`Learn more » <getting_started>`
:::

:::{grid-item-card} {material-regular}`menu_book;2em` Reference Guide
:link: reference_guide
:link-type: ref

Idea, structure and theoretical background of `delaynet`.\
Explore the details here!

+++
{ref}`Learn more »<reference_guide>`
:::

:::{grid-item-card} {material-regular}`lightbulb;2em` Demos
:link: demos
:link-type: ref

A collection of short demos showcasing the capabilities of this package.
+++
{ref}`Learn more »<Demos>`
:::

::::

## What is `delaynet`?

`delaynet` is a Python package designed to facilitate the reconstruction and analysis of
delay functional networks from time series. It provides tools for data preparation and
detrending, computing multiple connectivity measures (e.g. Granger causality, transfer
entropy, correlations), reconstructing networks with optimal lag selection, and
analysing network topology.
More abstractly, it applies to any system with **information propagation** where delays
play a role.

## Setup and use

To set up `delaynet`, see the {ref}`Getting started` page, more on
the details of the inner workings can be found on the
{ref}`Reference pages <reference_guide>`.
Furthermore, you can also find the {ref}`API documentation <API Reference>`.

## How to cite

If you use `delaynet` in your research, find the `CITATION.cff` file
in [the repository](https://github.com/cbueth/delaynet) and cite it
accordingly.
GitLab provides citation metadata from the `CITATION.cff`, and you can also copy an APA
or BibTeX entry from the repository.

A preprint is being prepared and will be submitted; arXiv/DOI links will be added upon
availability.

## Contributing

If you want to contribute to the development of `delaynet`, please read the
[CONTRIBUTING.md](https://github.com/cbueth/delaynet/-/blob/main/CONTRIBUTING.md)
file.

## Acknowledgments

This project has received funding from the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation programme (grant agreement No
851255).
This work was partially supported by the María de Maeztu project CEX2021-001164-M funded
by the MICIU/AEI/10.13039/501100011033 and FEDER, EU.

```{eval-rst}
.. toctree::
   :hidden:
   :name: table_of_contents
   :caption: Table of Contents
   :maxdepth: 1
   :glob:

   getting_started
   guide/index
   demos/index
   api/index
   changelog
   bibliography
```
