|Documentation Status|

An efficient and light-weight ordered set of 32 bits integers.
This is a Python wrapper for the C library `CRoaring <https://github.com/RoaringBitmap/CRoaring>`__.

Example
-------

You can use a bitmap nearly as the classical Python set in your code:

.. code:: python

    from pyroaring import BitMap
    bm1 = BitMap()
    bm1.add(3)
    bm1.add(18)
    print("has 3:", 3 in bm1)
    print("has 4:", 4 in bm1)
    bm2 = BitMap([3, 27, 42])
    print("bm1       = %s" % bm1)
    print("bm2       = %s" % bm2)
    print("bm1 & bm2 = %s" % (bm1&bm2))
    print("bm1 | bm2 = %s" % (bm1|bm2))

Output:

::

    has 3: True
    has 4: False
    bm1       = BitMap([3, 18])
    bm2       = BitMap([3, 27, 42])
    bm1 & bm2 = BitMap([3])
    bm1 | bm2 = BitMap([3, 18, 27, 42])

Installation from Pypi
----------------------

Supported systems: Linux, MacOS or Windows, Python 3.7 or higher. Note that pyroaring might still work with older Python
versions, but they are not tested anymore.

To install pyroaring on your local account, use the following command:

.. code:: bash

    pip install pyroaring --user

For a system-wide installation, use the following command:

.. code:: bash

    pip install pyroaring

Naturally, the latter may require superuser rights (consider prefixing
the commands by ``sudo``).

If you want to use Python 3 and your system defaults on Python 2.7, you
may need to adjust the above commands, e.g., replace ``pip`` by ``pip3``.

Installation from conda-forge
-----------------------------

Conda users can install the package from `conda-forge`:

.. code:: bash

   conda install -c conda-forge pyroaring

(Supports Python 3.6 or higher; Mac/Linux/Windows)

Installation from Source
---------------------------------

If you want to compile (and install) pyroaring by yourself, for instance
to modify the Cython sources you can follow the following instructions.
Note that these examples will install in your currently active python
virtual environment. Installing this way will require an appropriate
C compiler to be installed on your system.

First clone this repository.

.. code:: bash

    git clone https://github.com/Ezibenroc/PyRoaringBitMap.git

To install from Cython via source, for example during development run the following from the root of the above repository:

.. code:: bash

    python -m pip install .

This will automatically install Cython if it not present for the build, cythonise the source files and compile everything for you.

If you just want to recompile the package in place for quick testing you can
try the following:

.. code:: bash

    python setup.py build_clib
    python setup.py build_ext -i

Note that the build_clib compiles croaring only, and only needs to be run once.

Then you can test the new code using tox - this will install all the other
dependencies needed for testing and test in an isolated environment:

.. code:: bash

    python -m pip install tox
    tox

If you just want to run the tests directly from the root of the repository:

.. code:: bash

    python -m pip install hypothesis
    # This will test in three ways: via installation from source,
    # via cython directly, and creation of a wheel
    python test.py


Package pyroaring as an sdist and wheel. Note that building wheels that have
wide compatibility can be tricky - for releases we rely on `cibuildwheel <https://cibuildwheel.readthedocs.io/en/stable/>`_
to do the heavy lifting across platforms.

.. code:: bash

    python -m pip install build
    python -m build .

For all the above commands, two environment variables can be used to control the compilation.

- ``DEBUG=1`` to build pyroaring in debug mode.
- ``ARCHI=<cpu-type>`` to build pyroaring for the given platform. The platform may be any keyword
  given to the ``-march`` option of gcc (see the
  `documentation <https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html>`__).
  Note that cross-compiling for a 32-bit architecture from a 64-bit architecture is not supported.

Example of use:

.. code:: bash

    DEBUG=1 ARCHI=x86-64 python setup.py build_ext


Optimizing the builds for your machine (x64)
--------------------------------------------

For recent Intel and AMD (x64) processors under Linux, you may get better performance by requesting that
CRoaring be built for your machine, specifically, when building from source.
Be mindful that when doing so, the generated binary may only run on your machine.


.. code:: bash

    ARCHI=native pip install pyroaring  --no-binary :all:

This approach may not work under macOS.


Development Notes
-----------------

Updating CRoaring
=================

The download_amalgamation.py script can be used to download a specific version
of the official CRoaring amalgamation:

.. code:: bash

    python download_amalgamation.py v0.7.2

This will update roaring.c and roaring.h. This also means that the dependency
is vendored in and tracked as part of the source repository now. Note that the
__croaring_version__ in version.pxi will need to be updated to match the new
version.


Tracking Package and CRoaring versions
======================================

The package version is maintained in the file `pyroaring/version.pxi` - this
can be manually incremented in preparation for releases. This file is read
from in setup.py to specify the version.

The croaring version is tracked in `pyroaring/croaring_version.pxi` - this is
updated automatically when downloading a new amalgamation.


Benchmark
---------

``Pyroaring`` is compared with the built-in ``set`` and other implementations:

- A `Python wrapper <https://github.com/sunzhaoping/python-croaring>`__ of CRoaring called ``python-croaring``
- A `Cython implementation <https://github.com/andreasvc/roaringbitmap>`__ of Roaring bitmaps called ``roaringbitmap``
- A Python implementation of `ordered sets <https://github.com/grantjenks/sorted_containers>`__ called ``sortedcontainers``

The script ``quick_bench.py`` measures the time of different set
operations. It uses randomly generated sets of size 1e6 and density
0.125. For each operation, the average time (in seconds) of 30 tests
is reported.

The results have been obtained with:

- CPU Intel Xeon CPU E5-2630 v3
- CPython version 3.5.3
- gcc version 6.3.0
- Cython version 0.28.3
-  pyroaring commit
   `dcf448a <https://github.com/Ezibenroc/PyRoaringBitMap/tree/dcf448a166b535b35693071254d0042633671194>`__
-  python-croaring commit
   `3aa61dd <https://github.com/sunzhaoping/python-croaring/tree/3aa61dde6b4a123665ca5632eb5b089ec0bc5bc4>`__
-  roaringbitmap commit
   `502d78d <https://github.com/andreasvc/roaringbitmap/tree/502d78d2e5d65967ab61c1a759cac53ddfefd9a2>`__
-  sortedcontainers commit
   `7d6a28c <https://github.com/grantjenks/python-sortedcontainers/tree/7d6a28cdcba2f46eb2ef6cb1cc33cd8de0e8f27f>`__

===============================  ===========  =================  ===============  ==========  ==================
operation                          pyroaring    python-croaring    roaringbitmap         set    sortedcontainers
===============================  ===========  =================  ===============  ==========  ==================
range constructor                   3.09e-04           1.48e-04         8.72e-05    7.29e-02            2.08e-01
ordered list constructor            3.45e-02           6.93e-02         1.45e-01    1.86e-01            5.74e-01
list constructor                    1.23e-01           1.33e-01         1.55e-01    1.12e-01            5.12e-01
ordered array constructor           5.06e-03           6.42e-03         2.89e-01    9.82e-02            3.01e-01
array constructor                   1.13e-01           1.18e-01         4.63e-01    1.45e-01            5.08e-01
element addition                    3.08e-07           8.26e-07         2.21e-07    1.50e-07            1.18e-06
element removal                     3.44e-07           8.17e-07         2.61e-07    1.78e-07            4.26e-07
membership test                     1.24e-07           1.00e-06         1.50e-07    1.00e-07            5.72e-07
union                               1.61e-04           1.96e-04         1.44e-04    2.15e-01            1.11e+00
intersection                        9.08e-04           9.48e-04         9.26e-04    5.22e-02            1.65e-01
difference                          1.57e-04           1.97e-04         1.43e-04    1.56e-01            4.84e-01
symmetric diference                 1.62e-04           2.01e-04         1.44e-04    2.62e-01            9.13e-01
equality test                       7.80e-05           7.82e-05         5.89e-05    1.81e-02            1.81e-02
subset test                         7.92e-05           8.12e-05         8.22e-05    1.81e-02            1.81e-02
conversion to list                  4.71e-02           2.78e-01         4.35e-02    5.77e-02            5.32e-02
pickle dump & load                  4.02e-04           6.27e-04         5.08e-04    2.41e-01            5.75e-01
"naive" conversion to array         5.12e-02           2.92e-01         4.75e-02    1.20e-01            1.18e-01
"optimized" conversion to array     1.27e-03           3.40e-02       nan         nan                 nan
selection                           1.77e-06           5.33e-05         1.14e-06  nan                   1.64e-05
contiguous slice                    9.38e-05           9.51e-05         6.99e-05  nan                   2.04e-02
slice                               2.88e-03           3.04e-01         1.00e-01  nan                   4.74e-01
small slice                         8.93e-05           3.00e-01         3.60e-03  nan                   1.79e-02
===============================  ===========  =================  ===============  ==========  ==================

.. |Documentation Status| image:: https://readthedocs.org/projects/pyroaringbitmap/badge/?version=stable
   :target: http://pyroaringbitmap.readthedocs.io/en/stable/?badge=stable
