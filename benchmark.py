#! /usr/bin/env python3

import time
import random
import numpy
import abc
import pickle
import argparse
import sys
from pyroaring import BitMap

class AbstractBenchMark(metaclass=abc.ABCMeta):
    max_int = 2**32
    sizes = [2**n for n in range(0, 22)]
    classes = [BitMap, set]
    agregators=[min, max, numpy.mean, numpy.median]

    @classmethod
    def get_data(cls, size, bound):
        assert size < cls.max_int
        assert bound < cls.max_int
        assert size < bound
        return random.sample(range(0, bound), size)

    @classmethod
    def sparse_data(cls, size):
        bound = size*16
        return cls.get_data(size, bound)

    @classmethod
    def dense_data(cls, size):
        bound = size*2
        return cls.get_data(size, bound)

    def __init__(self):
        self.results = {data_func : dict() for data_func in [self.sparse_data, self.dense_data]}
        for data_func, d in self.results.items():
            for size in self.sizes:
                d[size] = {cls:list() for cls in self.classes}

    def compute_ratios(self):
        self.ratios = {data_func : dict() for data_func in self.results.keys()}
        for data_func, d in self.ratios.items():
            for size in self.sizes:
                d[size] = {cls:list() for cls in self.classes[1:]}
                for cls in self.classes[1:]:
                    r = list()
                    result = self.results[data_func][size]
                    for i in range(len(result[cls])):
                        r.append(result[cls][i]/result[self.classes[0]][i])
                    d[size][cls] = numpy.mean(r)

    @abc.abstractmethod
    def run(self, data_func, size):
        pass

    def run_all(self):
        for data_func in self.results.keys():
            for size in self.sizes:
                times = self.run(data_func, size)
                for i, cls in enumerate(self.classes):
                    self.results[data_func][size][cls].append(times[i])

    def print_results(self):
        print('\n# %s' % self.__class__.__name__)
        for data_func, d in sorted(self.ratios.items(), key = lambda x: x[0].__name__):
            print('\t%s' % data_func.__name__)
            for size, d_size in sorted(d.items()):
                print('\t\tSize=%d' % size)
                for test_class, ratio in sorted(d_size.items(), key = lambda x: x[0].__name__):
                    print('\t\t\tRatio %s/%s: %f' % (test_class.__name__, self.classes[0].__name__, ratio))

    @staticmethod
    def _results_to_tex(f, results): # results has the form { size1 : result1, size2 : result2, ...}
        f.write('\t\t\t\\addplot table {\n')
        for size, value in sorted(results.items()):
            f.write('\t\t\t\t%d %f\n' % (size, value))
        f.write('\t\t\t};\n')

    @staticmethod
    def add_legend(f, legend):
        f.write('\t\t\t\\addlegendentry{%s}\n' % legend.replace('_', '\\_'))

    def _plot_func(self, f):
        f.write('\t\\begin{tikzpicture}\n')
        f.write('\t\t\\begin{axis}[\n')
        f.write('\t\t\txlabel=size,\n')
        f.write('\t\t\txmode=log,\n')
        f.write('\t\t\tymode=log,\n')
        f.write('\t\t\tylabel=speedup,\n')
        f.write('\t\t\tlegend style={at={(0, 1)},anchor=north west,fill=none}\n')
        f.write('\t\t\t]\n')
        for data_func, d_func in sorted(self.ratios.items(), key=lambda x: x[0].__name__):
            results = dict()
            # First, we "inverse" the dict...
            for size, d_size in d_func.items():
                for test_class, value in d_size.items():
                    try:
                        results[test_class][size] = value
                    except KeyError:
                        results[test_class] = {size : value}
            for test_class in self.classes[1:]:
                self._results_to_tex(f, results[test_class])
        # Finally, the legend
        for data_func in sorted(self.ratios, key=lambda x: x.__name__):
            for test_class in self.classes[1:]:
                self.add_legend(f, 'ratio %s/%s: %s' % (test_class.__name__, self.classes[0].__name__, data_func.__name__))
        f.write('\t\t\\end{axis}\n')
        f.write('\t\\end{tikzpicture}\n')

    def plot_results(self, f):
        f.write('% Do not forget \\usepackage{pgfplots}\n')
        f.write('\\begin{figure}\n')
        self._plot_func(f)
        f.write('\t\\caption{%s}\n' % self.__class__.__name__)
        f.write('\\end{figure}\n')

class AbstractConstructorBenchMark(AbstractBenchMark):

    @abc.abstractmethod
    def get_values(self, data_func, size):
        pass

    def run_cls(self, cls, values):
        begin = time.time()
        result = cls(values)
        return time.time()-begin

    def run(self, data_func, size):
        values = self.get_values(data_func, size)
        assert len(values) == size
        return [self.run_cls(cls, values) for cls in self.classes]

class ListConstructor(AbstractConstructorBenchMark):

    def get_values(self, data_func, size):
        return data_func(size)

class RangeConstructor(AbstractConstructorBenchMark):

    def get_values(self, data_func, size):
        start = random.randint(0, self.max_int-size)
        max_step = (self.max_int-start)//size
        if data_func == self.sparse_data:
            step = random.randint(max_step//2, max_step)
        elif data_func == self.dense_data:
            step = random.randint(1, min(max_step, 4))
        else:
            assert False
        end = start + step*size
        return range(start, end, step)

class CopyConstructor(AbstractConstructorBenchMark):

    def get_values(self, data_func, size):
        return data_func(size)

    def run_cls(self, cls, values):
        values = cls(values)
        return super().run_cls(cls, values)

class AbstractBinaryOpBenchMark(AbstractBenchMark):

    @abc.abstractmethod
    def do_binary_op(self, values1, values2):
        pass

    def run_cls(self, cls, values1, values2):
        values1 = cls(values1)
        values2 = cls(values2)
        begin = time.time()
        self.do_binary_op(values1, values2)
        return time.time()-begin

    def run(self, data_func, size):
        values1 = data_func(size)
        values2 = data_func(size)
        return [self.run_cls(cls, values1, values2) for cls in self.classes]

class BinaryUnion(AbstractBinaryOpBenchMark):

    def do_binary_op(self, values1, values2):
        values1 | values2

class BinaryIntersection(AbstractBinaryOpBenchMark):

    def do_binary_op(self, values1, values2):
        values1 & values2

class BinarySymmetricDifference(AbstractBinaryOpBenchMark):

    def do_binary_op(self, values1, values2):
        values1 ^ values2

class BinaryUnionInPlace(AbstractBinaryOpBenchMark):

    def do_binary_op(self, values1, values2):
        values1 |= values2

class BinaryIntersectionInPlace(AbstractBinaryOpBenchMark):

    def do_binary_op(self, values1, values2):
        values1 &= values2

class BinarySymmetricDifferenceInPlace(AbstractBinaryOpBenchMark):

    def do_binary_op(self, values1, values2):
        values1 ^= values2

class AbstractManyOpBenchMark(AbstractBenchMark):

    @abc.abstractmethod
    def do_many_op(self, values):
        pass

    def run_cls(self, cls, values_list):
        values_list = [cls(v) for v in values_list]
        begin = time.time()
        self.do_many_op(cls, *values_list)
        return time.time()-begin

    def run(self, data_func, size):
        nb = 20#random.randint(2, 100)
        values_list = [data_func(size) for _ in range(nb)]
        return [self.run_cls(cls, values_list) for cls in self.classes]

class ManyUnion(AbstractManyOpBenchMark):

    def do_many_op(self, cls, *values):
        cls.union(*values)

classes = [ListConstructor, RangeConstructor, CopyConstructor,
    BinaryUnion, BinaryIntersection, BinarySymmetricDifference,
    BinaryUnionInPlace, BinaryIntersectionInPlace, BinarySymmetricDifferenceInPlace,
    ManyUnion]

class Runer:
    def __init__(self):
        self.benchmarks = [cls() for cls in classes]
        self.nb_measures = 0

    def run(self):
        begin = time.time()
        while True:
            print('Number of measures: %d' % self.nb_measures)
            try:
                for bench in self.benchmarks:
                    print('\t%s' % bench.__class__.__name__)
                    bench.run_all()
            except KeyboardInterrupt:
                end = time.time()
                break
            self.nb_measures += 1
        for bench in self.benchmarks:
            bench.compute_ratios()
        print('\nFinished after a time of %fs.' % (end-begin))

    def print_results(self):
        for bench in self.benchmarks:
            bench.print_results()

    def plot_results(self, file_name):
        with open(file_name, 'w') as f:
            f.write('%% Number of measures: %d\n' % self.nb_measures)
            f.write('\\documentclass{article}\n')
            f.write('\\usepackage{tikz, pgfplots}\n')
            f.write('\\begin{document}\n')
            f.write('\n')
            for bench in self.benchmarks:
                bench.plot_results(f)
                f.write('\n')
            f.write('\\end{document}')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
            description='Benchmark for pyroaring')
    parser.add_argument('-p', '--plot', type=str,
            default=None, help='Latex file to generate.')
    parser.add_argument('-d', '--dump', type=str,
            default=None, help='Pickle file to dump.')
    parser.add_argument('-l', '--load', type=str,
            default=None, help='Pickle file to load. If specified, the computation will not be done, the results will be loaded from the file instead.')
    args = parser.parse_args()

    if args.load is None:
        runer = Runer()
    else:
        with open(args.load, 'rb') as f:
            runer = pickle.load(f)
        print('Loaded results from file %s\n' % args.load)
    print('Now launching the benchmark. You can stop it properly at any time by pressing Ctrl-C (results will then be written and plotted).\n')
    runer.run()
    runer.print_results()
    print('')
    if args.plot is not None:
        runer.plot_results(args.plot)
        print('\nPlots written in file %s' % args.plot)
    if args.dump is not None:
        with open(args.dump, 'wb') as f:
            pickle.dump(runer, f)
        print('Results dumped in file %s' % args.dump)
