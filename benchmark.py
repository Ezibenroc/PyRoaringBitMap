#! /usr/bin/env python3

import time
import random
import numpy
import abc
from pyroaring import BitMap

class AbstractBenchMark(metaclass=abc.ABCMeta):
    max_int = 2**32
    classes = [BitMap, set]
    nb_repeat = 10

    def __init__(self, sample_sizes, agregators=[min, max, numpy.mean, numpy.median]):
        self.sample_sizes = sample_sizes
        self.agregators = agregators

    def iter_sizes(self):
        try:
            for size in self.sample_sizes:
                assert size > 0 and size < self.max_int
                yield size
        except TypeError: # not iterable
            size = self.sample_sizes
            assert isinstance(size, int) and size > 0 and size < 2**32
            yield size

    @abc.abstractmethod
    def run(self, size):
        '''
            Return a list of floats. Result[i] is the run-time of the class self.classes[i].
        '''
        pass

    def run_all(self, size):
        return [self.run(size) for _ in range(self.nb_repeat)]

    def agregate(self, measures, agregate_function):
        result = []
        for i in range(len(self.classes)):
            result.append(agregate_function([m[i] for m in measures]))
        return result

    def run_and_agregate(self, r_dict=None):
        r_dict = r_dict or dict()
        r_func = r_dict[self.__class__.__name__] = dict()
        for size in self.iter_sizes():
            r_size = r_func[size] = dict()
            results = self.run_all(size)
            for i, cls in enumerate(self.classes):
                r_class = r_size[cls.__name__] = dict()
                for agregator in self.agregators:
                    result = self.agregate(results, agregator)
                    r_agr = r_class[agregator.__name__] = result[i]
        return r_dict

    @staticmethod
    def print_results(comparison_class, results, agregators_to_print=None):
        for func, d_func in sorted(results.items(), key=(lambda t: '#'+t[0] if 'Constructor' in t[0] else t[0])):
            print('\n# %s' % func)
            for size, d_size in sorted(d_func.items()):
                print('\tSize=%d' % size)
                for cls, d_cls in sorted(d_size.items()):
                    print('\t\tClass %s' % cls)
                    agregators_to_print = agregators_to_print or d_cls.keys()
                    for agregator in agregators_to_print:
                        value = d_cls[agregator]
                        print('\t\t\t%s: %.4f s.' % (agregator.ljust(6), value))
                    for agregator in agregators_to_print:
                        if cls != comparison_class:
                            print('\t\t\tRatio of %s with %s: %.4f' % (agregator.ljust(6), comparison_class, value/d_size[comparison_class][agregator]))

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

class BinaryUnionInPlace(AbstractBinaryOpBenchMark):

    def do_binary_op(self, values1, values2):
        values1 |= values2

class BinaryIntersectionInPlace(AbstractBinaryOpBenchMark):

    def do_binary_op(self, values1, values2):
        values1 &= values2

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
    total_time = time.time()
    sample_sizes = [2**n for n in range(15, 20)]
    classes = [ListConstructor, RangeConstructor, ContinuousRangeConstructor, CopyConstructor,
        BinaryUnion, BinaryIntersection, BinaryUnionInPlace, BinaryIntersectionInPlace, ManyUnion]
    result_dict = None
    for cls in classes:
        print('Run %s...' % cls.__name__)
        result_dict = cls(sample_sizes).run_and_agregate(result_dict)
    total_time = time.time()-total_time
    AbstractBenchMark.print_results('BitMap', result_dict, agregators_to_print=['median', 'mean'])
    print('\nTotal time: %.2f s.' % total_time)
