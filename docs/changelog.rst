*********
Changelog
*********

Version 0.2.1 (2024-0X-XX)
**************************

* `PR 22 <https://github.com/cbueth/delaynet/pull/22>`_: Enhancements and Fixes for Entropy-based Connectivities

  - **Added**: Conversion `to_symbolic` for connectivities, function attribute `entropy_like`, check for symbolic time series connectivities, and connectivity decorator tests.
  - **Changed**: CI now includes windows and mac test runners, scheduled tests, coverage combine, and updated artifact action.
  - **Fixed**: Issue with `gt_bi_multi_lag` returning non-optimal idx.

Version 0.2.0 (2024-03-15)
**************************

* 🔄 Granger: Rework bidirectional version

  - Add description of GC
  - Add bibliography for doc
    `sphinxcontrib.bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest>`_

* 🐛 Fix: Stability of random data

  - Remove use of :func:`numpy.random.randint()`
  - Add test `test_gen_rand_data_stability()`
  - Add fixed seed to fixture `two_fmri_time_series()`

* 🐛 Fix: Fix OS connectivity

  - Rename to fit with US english: synchronisation -> synchronization

* 📝 EX: Add example comparing connectivities with fMRI data
* ✏️ Typo: Correct fMRI typo
* 🐛 Fix: Random time series indexing
* 📈 Z-Score: Added `max_period`, exclude current datapoint

  - Added `max_periods` parameter to Z-Score normalization function to limit the number of periods considered in calculations.
  - Excluded the current point from mean and standard deviation calculations.

* 📐 Sig: Make time series positional only
* 🧪 Test: Add automatic tests for all norms and connectivities
  - Uses generated data
  - Approaches not all functioning yet

* 📚 Add data generation methods

  - Generate fMRI time series
  - Wrapper for all approaches
  - Increased `max-args = 8`
  - Corrected argument order

* 📁 Ignore built folder, uses by pip
* 📚 Doc: Changed setup modality compatible with `pip` and `micromamba`
* 🧪 Tests: Add python `3.10` and `3.12` compatibility
* 🔄 CI/CD: Change environment caching runner

Version 0.1.0 (2024-02-16)
**************************

* 🏗 Setup main functionality
* 📚️ Setup documentation pages
* 🖍 Linting and formatting
* CI/CD pipeline: linting, building, testing, docs
* See changes before in the repository up until the tag `v0.1.0
  <https://github.com/cbueth/delaynet/releases/tag/v0.1.0>`_.