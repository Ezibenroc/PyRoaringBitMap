# Benchmarking pyroaring

See the notebook: [Performance evaluation of different pyroaring versions](version_evaluation.ipynb)

It is entirely based on measuring the duration of individual operations, like the union of two bitmaps. For a quick
test, it is also possible with such one-liners (adapting the initialization and operation to your needs):
```
python -c 'import timeit ; print(timeit.timeit(setup="from pyroaring import BitMap ; bm1 = BitMap(range(0, 1000000, 2)) ; bm2 = BitMap(range(1, 1000000, 2))", stmt="bm3 = bm1 | bm2"))'
```
