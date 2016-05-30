[![Build Status](https://travis-ci.org/Ezibenroc/PyRoaringBitMap.svg?branch=master)](https://travis-ci.org/Ezibenroc/PyRoaringBitMap)

This piece of code is a wrapper for the C library [CRoaring](https://github.com/RoaringBitmap/CRoaring).
It provides a very efficient way to store and manipulate sets of (unsigned 32 bits) integers.

## Installation

Install [CRoaring](https://github.com/RoaringBitmap/CRoaring), the C library for Roaring bitmap.

Add the path to this library in your environment:
```bash
export LD_LIBRARY_PATH=/path/to/your/CRoaring/build:$LD_LIBRARY_PATH
```
(consider adding this to your `.zshrc` or `.bashrc` if you do not want to type the command each time you use the library).

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

**Warning:** An important feature which still misses is a method to remove an element.

Moreover, it seems that the creation of a new `BitMap` instance is very slow. This is mainly due to the verification of the data consistency (i.e. checking that we have integers between 0 and 2^32-1) and the conversion from the Python data structure to a C array. We should find a way to improve that in a near future.
