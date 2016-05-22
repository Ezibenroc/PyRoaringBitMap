#! /usr/bin/env python3

import time
import random
import numpy
from pyroaring import BitMap

class BenchMark:
    classes = [BitMap, set]
    nb_repeat = 1
    size = 10000000

    @staticmethod
    def measure_constructor(cls, values):
        begin = time.time()
        result = cls(values)
        return time.time()-begin

    @staticmethod
    def measure_constructor_classes(classes, values):
        measures = []
        for cls in classes:
            measures.append(BenchMark.measure_constructor(cls, values))
        return measures

    @staticmethod
    def measure_copy_constructor(cls, values):
        values = cls(values)
        return BenchMark.measure_constructor(cls, values)

    @staticmethod
    def measure_copy_constructor_classes(classes, values):
        measures = []
        for cls in classes:
            measures.append(BenchMark.measure_copy_constructor(cls, values))
        return measures

    @staticmethod
    def measure_binary_operation(cls, values1, values2, op):
        values1 = cls(values1)
        values2 = cls(values2)
        begin = time.time()
        result = op(values1, values2)
        return time.time()-begin

    @staticmethod
    def measure_binary_operation_classes(classes, values1, values2, op):
        measures = []
        for cls in classes:
            measures.append(BenchMark.measure_binary_operation(cls, values1, values2, op))
        return measures

    def list_constructor(self):
        values = random.sample(range(self.size), self.size//2)
        return self.measure_constructor_classes(self.classes, values)

    def get_random_range(self):
        start = random.randint(0, 2**16)
        end = random.randint(self.size//2, self.size)
        return range(start, end)

    def range_constructor(self):
        values = self.get_random_range()
        return self.measure_constructor_classes(self.classes, values)

    def copy_constructor(self):
        values = self.get_random_range()
        return self.measure_copy_constructor_classes(self.classes, values)

    def union(self):
        values1 = self.get_random_range()
        values2 = self.get_random_range()
        return self.measure_binary_operation_classes(self.classes, values1, values2, lambda s1, s2: s1|s2)

    def intersection(self):
        values1 = self.get_random_range()
        values2 = self.get_random_range()
        return self.measure_binary_operation_classes(self.classes, values1, values2, lambda s1, s2: s1&s2)

    def union_inplace(self):
        values1 = self.get_random_range()
        values2 = self.get_random_range()
        return self.measure_binary_operation_classes(self.classes, values1, values2, lambda s1, s2: s1.__ior__(s2))

    def intersection_inplace(self):
        values1 = self.get_random_range()
        values2 = self.get_random_range()
        return self.measure_binary_operation_classes(self.classes, values1, values2, lambda s1, s2: s1.__iand__(s2))

    def measure_time(self, measurer):
        measures = []
        for _ in range(self.nb_repeat):
            measures.append(measurer())
        result = []
        for i in range(len(self.classes)):
            result.append(numpy.mean([m[i] for m in measures]))
        return result

    def print_result(self, measurer):
        print(measurer.__name__)
        result = self.measure_time(measurer)
        for i, t in enumerate(result):
            print('\t%s: %.3fs, ratio: %.3f' % (self.classes[i].__name__.ljust(10), t, t/result[0]))
        print('')

    def print_results(self):
        self.print_result(self.list_constructor)
        self.print_result(self.range_constructor)
        self.print_result(self.copy_constructor)
        self.print_result(self.union)
        self.print_result(self.intersection)
        self.print_result(self.union_inplace)
        self.print_result(self.intersection_inplace)

if __name__ == '__main__':
    bench = BenchMark()
    bench.print_results()
