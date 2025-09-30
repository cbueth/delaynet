# Demos

To get a better idea how to use `delaynet` in your code and research,
see the following notebooks which integrate the package's features.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {material-regular}`train;2em` SBB functional network (Aug 2025)
:link: demo_sbb_top50
:link-type: ref
Real-world functional delay network from Swiss long-distance trains (top 50 stations). Detrending, reconstruction (GC), pruning, and analysis with maps.
+++
{ref}`See demo »<demo_sbb_top50>`
:::

:::{grid-item-card} {material-regular}`hub;2em` Network reconstruction & metrics
:link: demo_network_reconstruction
:link-type: ref
Best-lag selection and p-value landscapes on SynthATDelays random connectivity (LC/RC/GC/MI/TE/COP).
+++
{ref}`See demo »<demo_network_reconstruction>`
:::

:::{grid-item-card} {material-regular}`science;2em` Ground-truth synthetic coupling
:link: 04_synthetic_ground_truth
:link-type: ref
Simple controlled-lag experiment (ts2 = a · ts1(t-τ) + noise). Compare LC/RC/GC/MI/TE/COP.
+++
*Under Preparation*
:::

:::{grid-item-card} {material-regular}`timeline;2em` Rolling-window topology
:link: demo_rolling_window
:link-type: ref
Evolving metrics on a stable generator with a regime change; GC reconstruction + FDR.
+++
*Under Preparation*
:::

:::{grid-item-card} {material-regular}`flight;2em` BTS air-transport demo
:link: demo_bts_mini
:link-type: ref
Small public dataset example: hourly airport delays → GC/COP networks + metrics.
+++
*Under Preparation*
:::
::::

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 2

   01_sbb_functional_network_top50
   02_network_reconstruction_and_metrics
```
