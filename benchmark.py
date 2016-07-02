#! /usr/bin/env python3

import time
import random
import numpy
import abc
import pickle
import argparse
from pyroaring import BitMap

class AbstractBenchMark(metaclass=abc.ABCMeta):
    max_int = 2**32
    classes = [BitMap, set]
    nb_repeat = 20
    agregators=[min, max, numpy.mean, numpy.median]

    def __init__(self, sample_sizes):
        self.sample_sizes = sample_sizes

    def iter_sizes(self):
        try:
            for size in self.sample_sizes:
                assert isinstance(size, int) and size > 0 and size < self.max_int
                yield size
        except TypeError: # not iterable
            size = self.sample_sizes
            assert isinstance(size, int) and size > 0 and size < self.max_int
            yield size

    @abc.abstractmethod
    def run(self, size):
        '''
            Return a list of floats. Result[i] is the run-time of the class self.classes[i].
        '''
        pass

    def run_all(self, size):
        return [self.run(size) for _ in range(self.nb_repeat)]

    def run_and_agregate(self, r_dict=None):
        r_dict = r_dict or dict()
        r_func = r_dict[self.__class__.__name__] = dict()
        for size in self.iter_sizes():
            r_size = r_func[size] = dict()
            results = self.run_all(size)
            for i, cls in enumerate(self.classes):
                r_class = r_size[cls.__name__] = [r[i] for r in results]
        return r_dict

    @classmethod
    def print_results(cls, comparison_class, results, agregators_to_print=None):
        if agregators_to_print is None:
            agregators_to_print = cls.agregators
        for func, d_func in sorted(results.items(), key=(lambda t: '#'+t[0] if 'Constructor' in t[0] else t[0])):
            print('\n# %s' % func)
            for size, d_size in sorted(d_func.items()):
                print('\tSize=%d' % size)
                for test_class, values in sorted(d_size.items()):
                    print('\t\tClass %s' % test_class)
                    for agregator in agregators_to_print:
                        value = agregator(values)
                        print('\t\t\t%s: %.4f s.' % (agregator.__name__.ljust(6), value))
                    for agregator in agregators_to_print:
                        if test_class != comparison_class:
                            print('\t\t\tRatio of %s with %s: %.4f' % (agregator.__name__.ljust(6), comparison_class, agregator(values)/agregator(d_size[comparison_class])))

    @staticmethod
    def _results_to_tex_noagreg(f, results): # results has the form { sizeA : [vA1, vA2, ..., vAn], sizeB : [vB1, vB2, ..., vBm], ...}
        f.write('\t\t\t\\addplot [only marks] table {\n')
        for size, values in sorted(results.items()):
            for value in values:
                f.write('\t\t\t\t%d %f\n' % (size, value))
        f.write('\t\t\t};\n')

    @staticmethod
    def _results_to_tex_agreg(f, results, agregator): # results has the form { sizeA : [vA1, vA2, ..., vAn], sizeB : [vB1, vB2, ..., vBm], ...}
        f.write('\t\t\t\\addplot table {\n')
        for size, values in sorted(results.items()):
            value = agregator(values)
            f.write('\t\t\t\t%d %f\n' % (size, value))
        f.write('\t\t\t};\n')

    @staticmethod
    def add_legend(f, legend):
        f.write('\t\t\t\\addlegendentry{%s}\n' % legend)

    @classmethod
    def _plot_func(cls, f, d_func, print_all_points, agregators_to_print):
        f.write('\t\\begin{tikzpicture}\n')
        f.write('\t\t\\begin{loglogaxis}[\n')
        f.write('\t\t\txlabel=size,\n')
        f.write('\t\t\tylabel=run time(s),\n')
        f.write('\t\t\tlegend style={at={(0, 1)},anchor=north west,fill=none}\n')
        f.write('\t\t\t]\n')
        results = dict()
        # First, we "inverse" the dict...
        for size, d_size in d_func.items():
            for test_class, values in d_size.items():
                try:
                    results[test_class][size] = values
                except KeyError:
                    results[test_class] = {size : values}
        # Then, we print the points
        for test_class, d_size in sorted(results.items()):
            if print_all_points:
                cls._results_to_tex_noagreg(f, d_size)
            for agregator in agregators_to_print:
                cls._results_to_tex_agreg(f, d_size, agregator)
        # Finally, the legend
        for test_class, d_size in sorted(results.items()):
            if print_all_points:
                cls.add_legend(f, '%s: all points' % test_class)
            for agregator in agregators_to_print:
                cls.add_legend(f, '%s: %s' % (test_class, agregator.__name__))
        f.write('\t\t\\end{loglogaxis}\n')
        f.write('\t\\end{tikzpicture}\n')

    @classmethod
    def plot_results(cls, f, results, print_all_points=True, agregators_to_print=None):
        if agregators_to_print is None:
            agregators_to_print = cls.agregators
        f.write('% Do not forget \\usepackage{pgfplots}\n')
        for func, d_func in sorted(results.items(), key=(lambda t: '#'+t[0] if 'Constructor' in t[0] else t[0])):
            f.write('\\begin{figure}\n')
            cls._plot_func(f, d_func, print_all_points, agregators_to_print)
            f.write('\t\\caption{%s}\n' % func)
            f.write('\\end{figure}\n')

    @classmethod
    def get_random_range(cls, size):
        start = random.randint(0, cls.max_int-size)
        max_step = (cls.max_int-start)//size
        if random.random() < 0.5: # compact sample
            step = random.randint(1, min(max_step, 32))
        else: # sparse sample
            step = random.randint(1, max_step)
        end = start + step*size
        return range(start, end, step)


class AbstractConstructorBenchMark(AbstractBenchMark):

    @abc.abstractmethod
    def get_values(self, size):
        pass

    def run_cls(self, cls, values):
        begin = time.time()
        result = cls(values)
        return time.time()-begin

    def run(self, size):
        values = self.get_values(size)
        assert len(values) == size
        return [self.run_cls(cls, values) for cls in self.classes]

class ListConstructor(AbstractConstructorBenchMark):

    def get_values(self, size):
        if random.random() < 0.5: # compact sample
            universe = range(min(self.max_int, size*2))
        else: # sparse sample
            universe = range(self.max_int)
        return random.sample(universe, size)


class RangeConstructor(AbstractConstructorBenchMark):

    def get_values(self, size):
        return self.get_random_range(size)


class ContinuousRangeConstructor(RangeConstructor):

    def get_values(self, size):
        return range(size)

class CopyConstructor(AbstractConstructorBenchMark):

    def get_values(self, size):
        return self.get_random_range(size)

    def run_cls(self, cls, values):
        values = cls(values)
        return super().run_cls(cls, values)

class AbstractBinaryOpBenchMark(AbstractBenchMark):

    @abc.abstractmethod
    def do_binary_op(self, values1, values2):
        pass

    def get_values(self, size):
        values = self.get_random_range(size)
        assert len(values) == size
        return values

    def run_cls(self, cls, values1, values2):
        values1 = cls(values1)
        values2 = cls(values2)
        begin = time.time()
        self.do_binary_op(values1, values2)
        return time.time()-begin

    def run(self, size):
        values1 = self.get_values(size)
        values2 = self.get_values(size)
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

    def get_values(self, size):
        values = self.get_random_range(size)
        assert len(values) == size
        return values

    def run_cls(self, cls, values_list):
        values_list = [cls(v) for v in values_list]
        begin = time.time()
        self.do_many_op(cls, *values_list)
        return time.time()-begin

    def run(self, size):
        nb = random.randint(2, 100)
        values_list = [self.get_values(size) for _ in range(nb)]
        return [self.run_cls(cls, values_list) for cls in self.classes]

class ManyUnion(AbstractManyOpBenchMark):

    def do_many_op(self, cls, *values):
        cls.union(*values)

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
        total_time = time.time()
        sample_sizes = [2**n for n in range(5, 20)]
        classes = [ListConstructor, RangeConstructor, ContinuousRangeConstructor, CopyConstructor,
            BinaryUnion, BinaryIntersection, BinarySymmetricDifference,
            BinaryUnionInPlace, BinaryIntersectionInPlace, BinarySymmetricDifferenceInPlace,
            ManyUnion]
        result_dict = None
        for cls in classes:
            print('Run %s...' % cls.__name__)
            result_dict = cls(sample_sizes).run_and_agregate(result_dict)
        total_time = time.time()-total_time
    else:
        with open(args.load, 'rb') as f:
            result_dict = pickle.load(f)
        print('Loaded results from file %s' % args.load)
    AbstractBenchMark.print_results('BitMap', result_dict, agregators_to_print=[numpy.mean, numpy.median])
    print('')
    if args.plot is not None:
        with open(args.plot, 'w') as f:
            AbstractBenchMark.plot_results(f, result_dict, print_all_points=False, agregators_to_print=[numpy.mean])
        print('\nPlots written in file %s' % args.plot)
    if args.dump is not None:
        with open(args.dump, 'wb') as f:
            pickle.dump(result_dict, f)
        print('Results dumped in file %s' % args.dump)
    if args.load is None:
        print('Total time: %.2f s.' % total_time)
