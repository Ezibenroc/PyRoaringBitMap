[![Build Status](https://travis-ci.org/Ezibenroc/PyRoaringBitMap.svg?branch=master)](https://travis-ci.org/Ezibenroc/PyRoaringBitMap)

This piece of code is a wrapper for the C library [CRoaring](https://github.com/RoaringBitmap/CRoaring).
It provides a very efficient way to store and manipulate sets of (unsigned 32 bits) integers.

## Requirements

- Environment like Linux and MacOS
- Python 2.7, or Python 3.3 or better
- A recent C compiler like GCC
- Numpy (optional, for benchmarking)
- The Python package ``hypothesis`` (optional, for testing)

## Installation


To install pyroaring and the CRoaring library on your local account, use the following two lines:
```bash
pip install pyroaring --user # installs PyRoaringBitMap
```
```bash
python -m pyroaring install --user # installs the CRoaring library
```
Then restart your shell.

To install them  system-wide, use the following lines :

```bash
pip install pyroaring
```
```bash
python -m pyroaring install
```

Naturally, the latter may require superuser rights (consider prefixing the commands by  ``sudo``).


(If you want to use Python 3 and your system defaults on Python 2.7, you may need to adjust the above commands, e.g., replace ``pip`` by ``pip3`` and python by ``python3``.)

#### Manual Installation of CRoaring

If you have troubles with the installation scripts or just want to install croaring by yourself, you can use the following steps.

First, get cmake, clone the repository and compile the project:
```bash
sudo apt-get install cmake
git clone https://github.com/RoaringBitmap/CRoaring.git
cd CRoaring && mkdir -p build && cd build && cmake .. && make
```

Then, either set the environment variable `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
```

Or install the library to your system:
```bash
sudo make install
```

## Utilization

First, you can run the tests to make sure everything is ok:
```bash
./test.py
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
Warning: when creating a new `BitMap` instance from a Python `list`, we truncate all integers to the least significant bits (values between 0 and 2^32).

## Benchmark

#### Utilization

With `benchmark.py`, you can compare the performances of objects implementing the set interface. Currently, we compare `BitMap` with `set`.

The syntax is the following:

```bash
./benchmark.py -l load_file.pickle -d dump_file.pickle -p plot_file.tex
```

It will launch the benchmark. The program will not stop until you press `Ctrl-C` or send the signal `SIGINT`.

* Benchmark results will be loaded from the file `load_file.pickle`. It is useful to resume a previous test.
* Benchmark results will be dumped into file `dump_file.pickle` after its execution. This file can then be loaded in a future execution with option `-l`.
* Benchmark results will be plotted into file `plot_file.tex`. It is a standalone Latex file using `pgfplots`. It can then be compiled using the command `pdflatex plot_file.tex`.

All these options are optional.

The benchmark will run operations of the set interface with randomly generated operands of given size. To do so, it picks randomly the right number of elements in a large enough range of possible elements. It tests for sparse operands (the range of possible elements is 16 times larger than the desired size) and dense operands (the range is 2 times larger).

#### Results

You can find results of the benchmark in file [plots.pdf](plots.pdf).

Experimental settings:

    CPU                : Intel i7-5600U
    RAM                : 16GB
    OS                 : GNU/Linux, LMDE2, kernel 4.6.0
    CRoaring version   : 0.2.1
    PyRoaring version  : 0.0.2
    Number of measures : 30

These results show the speedup obtained by using a roaring bitmap instead of the built-in Python set. We plot the average of the ratios `set/BitMap`.

All operations, except the list constructor, are faster with a roaring bitmap than with a set when there is at least 100 elements in the operand(s), regardless of their sparsity. The speedup increases with the size of the operands. The operations are much faster with dense operand(s). We can observe speedup of approximately 1000 for operations on dense operands with one million of elements.
