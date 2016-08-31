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


