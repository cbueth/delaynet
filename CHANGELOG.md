# Changelog

## [0.3.3](https://github.com/cbueth/delaynet/compare/v0.3.2...v0.3.3) (2026-08-12)

- 🚀 **2-4 times faster network reconstruction!**
  The Granger causality hot path was rewritten from `statsmodels.OLS().f_test()` to a
  pure numpy/scipy normal-equations F-test with p-values matching to machine precision.
  Network reconstruction with the `gc` metric measures **~2.1× faster** on CodSpeed,
  and the isolated Granger F-test is **up to ~8× faster**
  ([#46](https://github.com/cbueth/delaynet/pull/46),
  [`d5e4952`](https://github.com/cbueth/delaynet/commit/d5e4952fe672a1fba3e84d95cd65b7e5b1420471)).

- ⚡ **More performance improvements**
    - Delta detrending is now O(n) via cumulative sums instead of O(n w)
      ([`8119494`](https://github.com/cbueth/delaynet/commit/81194940a7954bf392a7c0b4dd12935bd38d8595))
    - `global_efficiency` inner loop vectorised
      ([`7d522e7`](https://github.com/cbueth/delaynet/commit/7d522e7f9ab68b7d7348bf2057ca90db479471d2))
    - Continuous ordinal patterns: serial fast path for small data, decorator overhead
      bypassed
      ([`e8f6e0f`](https://github.com/cbueth/delaynet/commit/e8f6e0fac738710ace3ee01a1e04537b161f007d))
    - Gravity permutation test vectorised with `rng.permuted`
      ([`5c402c5`](https://github.com/cbueth/delaynet/commit/5c402c59b82ff849fd26c2a32c95e6c0f038df9e))
    - `print_progress` throttled to 1 Hz instead of printing every pair
      ([`37cd00d`](https://github.com/cbueth/delaynet/commit/37cd00ddf88ba2c16318197ed6ef50b2d2f9bfbb))

- 📊 **Benchmarks**: new CodSpeed benchmark suite with CI workflow
  ([#45](https://github.com/cbueth/delaynet/issues/45),
  [`c42071c`](https://github.com/cbueth/delaynet/commit/c42071c50a00526458e0f58108799582dba3935c)) and
  benchmarks for detrending, network analysis, and gravity
  ([`e6dfa1a`](https://github.com/cbueth/delaynet/commit/e6dfa1a45efa4a3a6d69c559da5ce9f289419472)).

- 🐛 **Bug fixes**
    - Normalisation: retry on `ValueError` when sampling the null ensemble, fixing a flaky
      reciprocity test
      ([`07e53d3`](https://github.com/cbueth/delaynet/commit/07e53d3e6ac7884f9e5643731f4d1ffe2acec50d))
    - Eigenvector centrality: correct directed handling for igraph `1.0.0` and warn on
      symmetric matrices in directed mode
      ([`7675b84`](https://github.com/cbueth/delaynet/commit/7675b847ecf7d559d6eb7744f483006d7edb2e06),
      [`4b479fe`](https://github.com/cbueth/delaynet/commit/4b479fe8ebb6a9c261d643abd3238624080f8b39))
    - CI: align ruff pre-commit with `uv.lock` and fix commitlint `extends`
      ([`a90ac2b`](https://github.com/cbueth/delaynet/commit/a90ac2bd180e266c3c030abc59498214e023614f))
    - Docs: remove the header search button that pushed the sticky navbar down on scroll
      ([`f72b29d`](https://github.com/cbueth/delaynet/commit/f72b29dcd0edb24feab0235bf28ab0735eedbbe1))

- 📖 **Documentation**: CodSpeed badge
  ([`0a01312`](https://github.com/cbueth/delaynet/commit/0a01312767568cf962793cdb3b359fef4a1602ea))

## 0.3.2 (2025-08-14)

- 🐛 Fix: Normalised scalar metrics returned NumPy arrays instead of Python floats; now
  scalars remain floats, vectors remain arrays. Added a user warning when normalising
  link density since the null ensemble preserves the number of links (σ=0), making the
  null distribution degenerate with undefined z-scores (NaN), so normalising density is
  not very meaningful.

## 0.3.1 (2025-08-14)

- **Network metrics normalisation**
    - Added decorator-based option to return z-scores for all network analysis metrics
      by comparing against directed Erdos–Rényi $G(n,m)$ ensembles with matching nodes
      and
      links.
    - API: `normalise: bool | None` (default None), `n_random: int = 20`,
      `random_seed: int | None`.
    - Strictly binary-only for normalisation; weighted normalisation is not supported.
    - Documented in the Network Analysis guide with rationale and examples, citing
      {cite:p}`zaninStudyingTopologyTransportation2018`.
      See in {ref}`normalisation-metrics-example`.

## 0.3.0 (2025-08-13)

- 📊 **Data Generation Integration**: Added [
  `synthatdelays`](https://pypi.org/project/synthatdelays/) data generation
  configurations for synthetic time series
  creation (`61de4ca`)

- 🔧 **Development Infrastructure**
    - Updated UV package manager to version 0.8.4 (`9478c31`)
    - Enhanced `uv`-based testing and setup instructions (`31a803d`)
    - Added Python 3.14rc1 to CI testing with allowed failures (`9478c31`)

- 🔄 **API Improvements for Entropy-based Connectivities**

    - Simplify API for mutual information and transfer entropy connectivity metrics
    - Breaking change: Replaced `mi_kwargs` and `te_kwargs` dictionary parameters with
      direct keyword argument passing

- 🧪 **Test:** Push coverage from 81% -> 99%
    - Add test suite for detrending methods and connectivity metrics (`833e5c5`)
    - Update regex in test assertion for pattern size and lag-steps (`2ef4ba3`)
    - Increase test coverage among several modules (`6c2414d`)
    - Relaxed correlation threshold in data generator tests (`ac63942`)

- 🐛 **Fixes:**
    - Specify 'spawn' start method in ProcessPoolExecutor to resolve fork warnings (
      `f2532d4`)
    - Correct z-score detrending logic for edge casesand update test cases (`f7196d9`)
    - Improve progress bar display in Jupyter notebooks by adding automatic environment
      detection and proper line handling

- ✨ **Feature:** Add progress tracking for connectivity computation in both sequential
  and parallel modes (`eaaace4`)

- ⏲️ **Network Analysis and Benchmarking**

    - **Added:** Comprehensive network analysis module with metrics for evaluating
      network properties:
        - Network pruning with statistical significance testing and multiple comparison
          corrections
        - Centrality measures (betweenness, eigenvector) to identify important nodes
        - Global network metrics (link density, global efficiency, transitivity,
          reciprocity)
        - Node isolation detection (inbound/outbound)
    - **Added:** Cross-validation and benchmarking tools comparing delaynet with
      NetworkX, demonstrating 3-10x performance improvements
    - **Enhanced:** Integration with python-igraph for efficient network computations
    - **Improved:** Weight matrix diagonal defaults and related test coverage

- 🏗 **Network Reconstruction and Documentation Overhaul**

    - **Added:** Complete network reconstruction functionality with the new
      {func}`~delaynet.network_reconstruction.reconstruct_network` function that applies
      connectivity measures to pairs of time series to build networks represented by $p$
      -value and lag matrices.
    - **Enhanced:** Documentation structure with comprehensive individual guides for
      each method:
        - Dedicated guides for each detrending method with examples and mathematical
          background
        - Individual connectivity method guides with detailed explanations and use cases
        - New network reconstruction guide with practical examples and custom metric
          support
        - Network analysis guide for interpreting and working with reconstructed
          networks
    - **Improved:** Core connectivity framework with better error handling and
      validation
    - **Streamlined:** Documentation organization by consolidating scattered guides into
      focused, method-specific documentation that's easier to navigate and understand

- 🔄 **API Refactoring: Normalization → Detrending**

    - **Breaking Change:** Renamed all "normalisation/norm" terminology to "
      detrending/detrend" for semantic accuracy
    - **Renamed:** Main API function from `normalise()` to `detrend()` and
      `show_norms()` to `show_detrending_methods()`
    - **Renamed:** Decorator from `@norm` to `@detrending_method` for consistency with
      functionality
    - **Renamed:** Package structure from `delaynet/norms/` to
      `delaynet/detrending_methods/`
    - **Updated:** All documentation, tests, and examples to use the new detrending
      terminology
    - **Rationale:** These methods perform detrending (trend removal) rather than
      mathematical normalization, making the new naming more theoretically accurate

- 🔄 **Connectivity Framework Unification** (`bbbe265`)

    - **Unified:** All connectivity measures now return consistent $p$-values for
      statistical significance testing
    - **Added:** Universal $p$-value support across all connectivity methods for
      standardized output
    - **Streamlined:** Unified `lag_steps` parameter handling across all connectivity
      measures
    - **Removed:** Ordinal Patterns Connectivity and Bi-Granger due to unclear $p$-value

- 📦 Updated infomeasure dependency to v0.5.0 and replaced deprecated p_value usage.
- 🌐 Consistently applied British English conventions across all files (behavior →
  behaviour, normalize → normalise).
- 🚀 Added parallel execution support for network reconstruction.
- 🔧 Enhanced norm decorator to support multi-dimensional arrays with axis parameter.
- 📊 Unified connectivity output format to return tuple[float, int] and replaced `offset`
  with `prop_time`.
- 🧪 Fixed test assertion issues and improved code formatting.
- 🖥️ Added Windows and macOS test runners for better cross-platform compatibility.
- 🐛 Fixed gt_bi_multi_lag returning non-optimal index and improved integer type checking
  for Windows compatibility.
- 📦 Refactored DelayNet → delaynet naming convention and optimized COP connectivity
  performance.

- 🗑️ **Symbolization Removal** (`a4db41c`)

    - **Removed:** Symbolization functionality to streamline the package and focus on
      core connectivity measures
    - **Simplified:** Codebase by eliminating unused symbolization components

- 🚀 [#22](https://github.com/cbueth/delaynet/pull/22): Enhancements and Fixes for
  Entropy-based Connectivities

    - **Added:** Conversion `to_symbolic` for connectivities, function attribute
      `entropy_like`, check for symbolic time series connectivities, and connectivity
      decorator tests.
    - **Changed:** CI now includes windows and mac test runners, scheduled tests,
      coverage combine, and updated artifact action.
    - **Fixed:** Issue with `gt_bi_multi_lag` returning non-optimal idx.

- 🧩 [#24](https://github.com/cbueth/delaynet/issues/24): COP: introduce
  `pattern_transform()` / `convolute_ts()`

    - Added `pattern_transform` for connectivity calculation.
    - Support for single/multiple time series and patterns.
    - ⚡️ Performance improvements and new unit tests.

- 📦 [#27](https://github.com/cbueth/delaynet/issues/27): Integrate [
  `infomeasure`](https://infomeasure.readthedocs.io/) package

    - Switched to published [`infomeasure`](https://infomeasure.readthedocs.io/)
      dependency.
    - Updated install and cache locations, test integration.

- 🔄 [#29](https://github.com/cbueth/delaynet/issues/29): Move development as well to
  gitlab?

    - CI/CD migrated to GitLab, deactivated GitHub Actions, code quality pipeline
      improvements.

- 🗑️ Removed `data` folder (`9541369`).
- 📝 Documentation and README updates (links, badges, citation, Python version).
- 🏷️ Added Citation, Code of Conduct, and CONTRIBUTING.md.
- 🧪 General test, CI, and dependency improvements.
- 🛠️ Minor optimizations to Z-Score detrending, COP connectivity, and code cleanup.

---

## 0.2.0 (2024-03-15)

- 🔄 Granger: Rework bidirectional version

    - Add description of GC
    - Add bibliography for doc
      [`sphinxcontrib.bibtex`](https://sphinxcontrib-bibtex.readthedocs.io/en/latest)

- 🐛 Fix: Stability of random data

    - Remove use of `numpy.random.randint()`
    - Add test `test_gen_rand_data_stability()`
    - Add fixed seed to fixture `two_fmri_time_series()`

- 🐛 Fix: Fix OS connectivity

    - Rename to fit with US English: synchronisation → synchronization

- 📝 EX: Add example comparing connectivities with fMRI data
- ✏️ Typo: Correct fMRI typo
- 🐛 Fix: Random time series indexing
- 📈 Z-Score: Added `max_period`, exclude current datapoint

    - Added `max_periods` parameter to Z-Score detrending function to limit the number
      of periods considered in calculations.
    - Excluded the current point from mean and standard deviation calculations.

- 📐 Sig: Make time series positional only
- 🧪 Test: Add automatic tests for all detrending methods and connectivities
    - Uses generated data
    - Approaches not all functioning yet

- 📚 Add data generation methods

    - Generate fMRI time series
    - Wrapper for all approaches
    - Increased `max-args = 8`
    - Corrected argument order

- 📁 Ignore built folder, used by pip
- 📚 Doc: Changed setup modality compatible with `pip` and `micromamba`
- 🧪 Tests: Add python `3.10` and `3.12` compatibility
- 🔄 CI/CD: Change environment caching runner

---

## 0.1.0 (2024-02-16)

- 🏗 Setup main functionality
- 📚️ Setup documentation pages
- 🖍 Linting and formatting
- CI/CD pipeline: linting, building, testing, docs
- See changes before in the repository up until the tag [
  `v0.1.0`](https://github.com/cbueth/delaynet/releases/tag/v0.1.0).
