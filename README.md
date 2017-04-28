[![Build Status](https://travis-ci.org/Ezibenroc/PyRoaringBitMap.svg?branch=master)](https://travis-ci.org/Ezibenroc/PyRoaringBitMap)

This piece of code is a wrapper for the C library [CRoaring](https://github.com/RoaringBitmap/CRoaring).
It provides a very efficient way to store and manipulate sets of (unsigned 32 bits) integers.

The wrapping used to be done with ``Ctypes``. We recently switched to ``Cython`` for the following reasons:

* Much better performances for short function calls (e.g. addition/deletion of a single element).
* Easier installation, no need to install manually the C library, it is now distributed with PyRoaring.
* Extensibility, it will be easier to write efficient codes with Cython (e.g. some data structures based on roaring bitmaps).

If for some reason you wish to keep using the old version, based on ``Ctypes``, use [PyRoaring 0.0.7](https://github.com/Ezibenroc/PyRoaringBitMap/tree/0.0.7).

## Requirements

- Environment like Linux and MacOS
- Python 2.7, or Python 3.3 or better
- A recent C compiler like GCC
- The Python package ``hypothesis`` (optional, for testing)
- The Python package ``Cython`` (optional, for compiling pyroaring from the sources)

## Installation


To install pyroaring and the CRoaring library on your local account, use the following two lines:
```bash
pip install pyroaring --user # installs PyRoaringBitMap
```
To install them  system-wide, use the following lines :
```bash
pip install pyroaring
```

Naturally, the latter may require superuser rights (consider prefixing the commands by  ``sudo``).


(If you want to use Python 3 and your system defaults on Python 2.7, you may need to adjust the above commands, e.g., replace ``pip`` by ``pip3`` and python by ``python3``.)

## Manual compilation / installation

If you want to compile (and install) pyroaring by yourself, for instance to modify the Cython sources or because you do not have ``pip``, follow these steps.
Note that the Python package ``Cython`` is required.

Clone this repository.
```bash
git clone https://github.com/Ezibenroc/PyRoaringBitMap.git
cd PyRoaringBitMap
```

Build pyroaring locally, e.g. to test a new feature you made.
```bash
python setup.py build_ext -i
python test.py # run the tests, optionnal but recommended
```

Install pyroaring (use this if you do not have ``pip``).
```bash
python setup.py install # may require superuser rights, add option --user if you wish to install it on your local account 
```

Package pyroaring.
```bash
python setup.py sdist
pip install dist/pyroaring-0.0.7.tar.gz # optionnal, to install the package
```

## Utilization

First, you can run the tests to make sure everything is ok:
```bash
pip install hypothesis --user
python test.py
```

You can use a bitmap nearly as the classical Python set in your code:
```python
from pyroaring import BitMap
bm1 = BitMap()
bm1.add(3)
bm1.add(18)
bm2 = BitMap([3, 27, 42])
print("bm1       = %s" % bm1)
print("bm2       = %s" % bm2)
print("bm1 & bm2 = %s" % (bm1&bm2))
print("bm1 | bm2 = %s" % (bm1|bm2))
```

Output:
```
bm1       = BitMap([3, 18])
bm2       = BitMap([3, 27, 42])
bm1 & bm2 = BitMap([3])
bm1 | bm2 = BitMap([3, 18, 27, 42])
```

## Benchmark

The built-in `set` is compared with this Python wrapper of `CRoaring` (designated as `pyroaring` in the following) and a [Cython implementation](https://github.com/andreasvc/roaringbitmap) of Roaring bitmaps (designated as `cyroaring` in the following).

### Quick benchmarks for common operations

The script ``quick_bench.sh`` measures the time of different set operations. It uses sets initialized to ``range(b, 100000000, 8)`` with ``b`` equal to 0 or 1. It is far from being exhaustive, but rather a quick overview of how the three classes compare to each other.

| Operation           | Pyroaring | Cyroaring |      set |
| ------------------- | --------- | --------- | -------- |
| Empty constructor   |  0.000134 |  0.000154 | 7.75e-05 |
| Range constructor   |      4.51 |      4.15 |      756 |
| List constructor    |       153 |       124 |      584 |
| Element addition    |  7.09e-05 |  7.06e-05 |  6.5e-05 |
| Test for membership |  3.03e-05 |  3.28e-05 |  2.6e-05 |
| Conversion to list  |       513 |       486 |      146 |
| Equality test       |      1.47 |      1.35 |      289 |
| Subset test         |      1.54 |      1.46 |      283 |
| Union               |      3.18 |       3.4 |      811 |
| Intersection        |      2.58 |      2.52 |      132 |
| Symetric difference |      3.13 |      3.19 |      927 |
| Pickle dump & load  |      17.4 |      17.3 | 1.29e+03 |
| Selection           |   0.00754 |   0.00135 |       NA |
| Slice               |       614 |  2.88e+03 |       NA |

### Complete benchmark for the union

The performances of the `union` operation have been measured more carefully. Full results can be found [here](https://github.com/Ezibenroc/roaring_analysis/blob/master/python_analysis.ipynb).

Three interesting plots:

![Plot of the performances for sparse data (density of 0.04)](benchmark_sparse.png)

![Plot of the performances for dense data (density of 0.5)](benchmark_dense.png)

![Plot of the performances for very dense data (density of 0.999)](benchmark_very_dense.png)

To sum up, both Roaring bitmap implementations are several orders of magnitude faster than the built-in set, regardless of the density of the data.

For sparse data, `pyroaring` is faster than `cyroaring`, for very dense data `cyroaring` is faster. Otherwise, they are similar.
