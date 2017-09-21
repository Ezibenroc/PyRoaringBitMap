|Build Status|
|Documentation Status|

An efficient and light-weight ordered set of 32 bits integers.
This is a Python wrapper for the C library `CRoaring <https://github.com/RoaringBitmap/CRoaring>`__.

The wrapping used to be done with ``Ctypes``. We recently switched to
``Cython`` for the following reasons:

-  Much better performances for short function calls (e.g.
   addition/deletion of a single element).
-  Easier installation, no need to install manually the C library, it is
   now distributed with PyRoaring.
-  Extensibility, it will be easier to write efficient codes with Cython
   (e.g. some data structures based on roaring bitmaps).

If for some reason you wish to keep using the old version, based on
``Ctypes``, use `PyRoaring
0.0.7 <https://github.com/Ezibenroc/PyRoaringBitMap/tree/0.0.7>`__.

Requirements
------------

-  Environment like Linux and MacOS
-  Python 2.7, or Python 3.3 or better
-  A recent C compiler like GCC
-  The package manager ``pip``
-  The Python package ``hypothesis`` (optional, for testing)
-  The Python package ``Cython`` (optional, for compiling pyroaring from
   the sources)
-  The Python package ``wheel`` (optional, to build a wheel for the library)

Installation
------------

To install pyroaring on your local account, use the following command:

.. code:: bash

    pip install pyroaring --user # installs PyRoaringBitMap

For a system-wide installation, use the following command:

.. code:: bash

    pip install pyroaring

Naturally, the latter may require superuser rights (consider prefixing
the commands by ``sudo``).

If you want to use Python 3 and your system defaults on Python 2.7, you
may need to adjust the above commands, e.g., replace ``pip`` by ``pip3``.

Manual compilation / installation
---------------------------------

If you want to compile (and install) pyroaring by yourself, for instance
to modify the Cython sources or because you do not have ``pip``, follow
these steps. Note that the Python package ``Cython`` is required.

Clone this repository.

.. code:: bash

    git clone https://github.com/Ezibenroc/PyRoaringBitMap.git
    cd PyRoaringBitMap

Build pyroaring locally, e.g. to test a new feature you made.

.. code:: bash

    python setup.py build_ext -i
    python test.py # run the tests, optionnal but recommended

Install pyroaring (use this if you do not have ``pip``).

.. code:: bash

    python setup.py install # may require superuser rights, add option --user if you wish to install it on your local account 

Package pyroaring.

.. code:: bash

    python setup.py sdist
    pip install dist/pyroaring-0.1.?.tar.gz # optionnal, to install the package

Build a wheel.

.. code:: bash

    python setup.py bdist_wheel

For all the above commands, two environment variables can be used to control the compilation.

- ``DEBUG=1`` to build pyroaring in debug mode.
- ``ARCHI=<cpu-type>`` to build pyroaring for the given platform. The platform may be any keyword
  given to the ``-march`` option of gcc (see the
  `documentation <https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html>`__).
  Note that cross-compiling for a 32-bit architecture from a 64-bit architecture is not supported.

Example of use:

.. code:: bash

    DEBUG=1 ARCHI=x86-64 python setup.py build_ext

Utilization
-----------

First, you can run the tests to make sure everything is ok:

.. code:: bash

    pip install hypothesis --user
    python test.py

You can use a bitmap nearly as the classical Python set in your code:

.. code:: python

    from pyroaring import BitMap
    bm1 = BitMap()
    bm1.add(3)
    bm1.add(18)
    bm2 = BitMap([3, 27, 42])
    print("bm1       = %s" % bm1)
    print("bm2       = %s" % bm2)
    print("bm1 & bm2 = %s" % (bm1&bm2))
    print("bm1 | bm2 = %s" % (bm1|bm2))

Output:

::

    bm1       = BitMap([3, 18])
    bm2       = BitMap([3, 27, 42])
    bm1 & bm2 = BitMap([3])
    bm1 | bm2 = BitMap([3, 18, 27, 42])

Benchmark
---------

``Pyroaring`` is compared with the built-in ``set`` and other implementations:

- A `Python wrapper <https://github.com/sunzhaoping/python-croaring>`__ of CRoaring called ``python-croaring``
- A `Cython implementation <https://github.com/andreasvc/roaringbitmap>`__ of Roaring bitmaps called ``roaringbitmap``
- A Python implemenntation of `ordered sets <https://github.com/grantjenks/sorted_containers>`__ called ``sortedcontainers``

The script ``quick_bench.py`` measures the time of different set
operations. It uses randomly generated sets of size 1e6 and density
0.125. For each operation, the average time (in seconds) of 30 tests
is reported.

The results have been obtained with:

- CPU Intel i7-7820HQ
- CPython version 3.5.3
- gcc version 6.3.0
- pyroaring commit ``6c86765d0357492895fee99de8841ce42340f879``
- python-croaring commit ``3aa61dde6b4a123665ca5632eb5b089ec0bc5bc4``
- roaringbitmap commit ``a32915f262eb4e39b854d942e005dc7381796808``
- sortedcontainers commit ``53fd6c54aebe5b969adc87d4b5e6331be1e32079``

===============================  ===========  =================  ===============  ==========  ==================
operation                          pyroaring    python-croaring    roaringbitmap         set    sortedcontainers
===============================  ===========  =================  ===============  ==========  ==================
range constructor                   1.08e-04           1.14e-04         8.89e-05    4.18e-02            1.33e-01
ordered list constructor            2.58e-02           5.25e-02         1.01e-01    1.23e-01            3.88e-01
list constructor                    9.18e-02           1.05e-01         1.26e-01    6.80e-02            3.47e-01
ordered array constructor           4.07e-03           5.05e-03         2.19e-01    6.30e-02            2.13e-01
array constructor                   8.55e-02           9.11e-02         3.88e-01    1.05e-01            3.63e-01
element addition                    1.48e-07           5.23e-07         1.45e-07    1.06e-07            9.74e-07
element removal                     1.40e-07           5.41e-07         1.26e-07    1.02e-07            4.41e-07
membership test                     7.39e-08           6.59e-07         8.03e-08    5.90e-08            3.74e-07
union                               1.03e-04           1.37e-04         1.02e-04    1.03e-01            7.09e-01
intersection                        8.44e-04           8.05e-04         7.90e-04    3.73e-02            1.20e-01
difference                          1.02e-04           1.40e-04         9.97e-05    9.24e-02            3.16e-01
symmetric diference                 1.02e-04           1.36e-04         9.81e-05    1.34e-01            5.94e-01
equality test                       5.36e-05           5.62e-05         4.30e-05    1.56e-02            1.53e-02
subset test                         6.97e-05           6.00e-05         5.91e-05    1.54e-02            1.55e-02
conversion to list                  3.37e-02           2.12e-01         3.18e-02    3.50e-02            3.83e-02
pickle dump & load                  2.25e-04           3.56e-04         2.47e-04    1.46e-01            3.88e-01
"naive" conversion to array         3.35e-02           2.25e-01         3.16e-02    6.39e-02            6.10e-02
"optimized" conversion to array     1.20e-03           2.42e-02       nan         nan                 nan
selection                           7.69e-07           3.76e-05         1.13e-06  nan                   8.57e-06
slice                               3.23e-03           2.35e-01         1.23e-01  nan                   5.76e-01
===============================  ===========  =================  ===============  ==========  ==================

.. |Build Status| image:: https://travis-ci.org/Ezibenroc/PyRoaringBitMap.svg?branch=master
   :target: https://travis-ci.org/Ezibenroc/PyRoaringBitMap
.. |Documentation Status| image:: https://readthedocs.org/projects/pyroaringbitmap/badge/?version=stable
   :target: http://pyroaringbitmap.readthedocs.io/en/stable/?badge=stable
