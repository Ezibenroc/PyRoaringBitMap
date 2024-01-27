#! /usr/bin/env python3

import os
import re
import sys
import array
import pickle
import random
import operator
import unittest
import functools

import hypothesis.strategies as st
from hypothesis import given, assume, errors, settings, Verbosity, HealthCheck

import pyroaring
from pyroaring import BitMap, FrozenBitMap


settings.register_profile("ci", settings(
    max_examples=100, deadline=None))
settings.register_profile("dev", settings(max_examples=10, deadline=None))
settings.register_profile("debug", settings(
    max_examples=10, verbosity=Verbosity.verbose, deadline=None))
try:
    env = os.getenv('HYPOTHESIS_PROFILE', 'dev')
    settings.load_profile(env)
except errors.InvalidArgument:
    sys.exit('Unknown hypothesis profile: %s.' % env)

uint18 = st.integers(min_value=0, max_value=2**18)
uint32 = st.integers(min_value=0, max_value=2**32 - 1)
uint64 = st.integers(min_value=0, max_value=2**64 - 1)
large_uint64 = st.integers(min_value=2**32, max_value=2**64 - 1)
integer = st.integers(min_value=0, max_value=2**31 - 1)
int64 = st.integers(min_value=-2**63, max_value=2**63 - 1)

range_max_size = 2**18

range_big_step = uint18.flatmap(lambda n:
                                st.builds(range, st.just(n),
                                          st.integers(
                                              min_value=n + 1, max_value=n + range_max_size),
                                          st.integers(min_value=2**8, max_value=range_max_size // 8)))

range_small_step = uint18.flatmap(lambda n:
                                  st.builds(range, st.just(n),
                                            st.integers(
                                                min_value=n + 1, max_value=n + range_max_size),
                                            st.integers(min_value=1, max_value=2**8)))

range_power2_step = uint18.flatmap(lambda n:
                                   st.builds(range, st.just(n),
                                             st.integers(
                                                 min_value=n + 1, max_value=n + range_max_size),
                                             st.integers(min_value=0, max_value=8).flatmap(
                                       lambda n: st.just(2**n),
                                   )))

hyp_range = range_big_step | range_small_step | range_power2_step | st.sampled_from(
    [range(0, 0)])  # last one is an empty range
# would be great to build a true random set, but it takes too long and hypothesis does a timeout...
hyp_set = st.builds(set, hyp_range)
hyp_array = st.builds(lambda x: array.array('I', x), hyp_range)
hyp_collection = hyp_range | hyp_set | hyp_array
hyp_many_collections = st.lists(hyp_collection, min_size=1, max_size=20)

bitmap_cls = st.sampled_from([BitMap, FrozenBitMap])


class Util(unittest.TestCase):

    comparison_set = random.sample(
        range(2**8), 100) + random.sample(range(2**31 - 1), 50)

    def compare_with_set(self, bitmap, expected_set):
        self.assertEqual(len(bitmap), len(expected_set))
        self.assertEqual(bool(bitmap), bool(expected_set))
        self.assertEqual(set(bitmap), expected_set)
        self.assertEqual(sorted(list(bitmap)), sorted(list(expected_set)))
        self.assertEqual(
            BitMap(expected_set, copy_on_write=bitmap.copy_on_write), bitmap)
        for value in self.comparison_set:
            if value in expected_set:
                self.assertIn(value, bitmap)
            else:
                self.assertNotIn(value, bitmap)

    @staticmethod
    def bitmap_sample(bitmap, size):
        indices = random.sample(range(len(bitmap)), size)
        return [bitmap[i] for i in indices]

    def assert_is_not(self, bitmap1, bitmap2):
        if isinstance(bitmap1, BitMap):
            if bitmap1:
                bitmap1.remove(bitmap1[0])
            else:
                bitmap1.add(27)
        elif isinstance(bitmap2, BitMap):
            if bitmap2:
                bitmap2.remove(bitmap1[0])
            else:
                bitmap2.add(27)
        else:  # The two are non-mutable, cannot do anything...
            return
        if bitmap1 == bitmap2:
            self.fail(
                'The two bitmaps are identical (modifying one also modifies the other).')


class BasicTest(Util):

    @given(hyp_collection, st.booleans())
    @settings(deadline=None)
    def test_basic(self, values, cow):
        bitmap = BitMap(copy_on_write=cow)
        self.assertEqual(bitmap.copy_on_write, cow)
        expected_set = set()
        self.compare_with_set(bitmap, expected_set)
        values = list(values)
        random.shuffle(values)
        size = len(values)
        for value in values[:size // 2]:
            bitmap.add(value)
            expected_set.add(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[size // 2:]:
            bitmap.add(value)
            with self.assertRaises(KeyError):
                bitmap.add_checked(value)
            expected_set.add(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[:size // 2]:
            bitmap.remove(value)
            expected_set.remove(value)
            with self.assertRaises(KeyError):
                bitmap.remove(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[size // 2:]:
            bitmap.discard(value)
            # check that we can discard element not in the bitmap
            bitmap.discard(value)
            expected_set.discard(value)
        self.compare_with_set(bitmap, expected_set)

    @given(bitmap_cls, bitmap_cls, hyp_collection, st.booleans())
    def test_bitmap_equality(self, cls1, cls2, values, cow):
        bitmap1 = cls1(values, copy_on_write=cow)
        bitmap2 = cls2(values, copy_on_write=cow)
        self.assertEqual(bitmap1, bitmap2)

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_bitmap_unequality(self, cls1, cls2, values1, values2, cow):
        assume(set(values1) != set(values2))
        bitmap1 = cls1(values1, copy_on_write=cow)
        bitmap2 = cls2(values2, copy_on_write=cow)
        self.assertNotEqual(bitmap1, bitmap2)

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_constructor_values(self, cls, values, cow):
        bitmap = cls(values, copy_on_write=cow)
        expected_set = set(values)
        self.compare_with_set(bitmap, expected_set)

    @given(bitmap_cls, bitmap_cls, hyp_collection, uint32, st.booleans(), st.booleans())
    def test_constructor_copy(self, cls1, cls2, values, other_value, cow1, cow2):
        bitmap1 = cls1(values, copy_on_write=cow1)
        # should be robust even if cow2 != cow1
        bitmap2 = cls2(bitmap1, copy_on_write=cow2)
        self.assertEqual(bitmap1, bitmap2)
        self.assert_is_not(bitmap1, bitmap2)

    @given(hyp_collection, hyp_collection, st.booleans())
    def test_update(self, initial_values, new_values, cow):
        bm = BitMap(initial_values, cow)
        expected = BitMap(bm)
        bm.update(new_values)
        expected |= BitMap(new_values, copy_on_write=cow)
        self.assertEqual(bm, expected)

    @given(hyp_collection, hyp_collection, st.booleans())
    def test_intersection_update(self, initial_values, new_values, cow):
        bm = BitMap(initial_values, cow)
        expected = BitMap(bm)
        bm.intersection_update(new_values)
        expected &= BitMap(new_values, copy_on_write=cow)
        self.assertEqual(bm, expected)

    def wrong_op(self, op):
        bitmap = BitMap()
        with self.assertRaises(OverflowError):
            op(bitmap, -3)
        with self.assertRaises(OverflowError):
            op(bitmap, 2**33)
        with self.assertRaises(TypeError):
            op(bitmap, 'bla')

    def test_wrong_add(self):
        self.wrong_op(lambda bitmap, value: bitmap.add(value))

    def test_wrong_contain(self):
        self.wrong_op(lambda bitmap, value: bitmap.__contains__(value))

    @given(bitmap_cls)
    def test_wrong_constructor_values(self, cls):
        with self.assertRaises(TypeError):  # this should fire a type error!
            cls([3, 'bla', 3, 42])
        with self.assertRaises(ValueError):
            cls(range(0, 10, 0))

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_to_array(self, cls, values, cow):
        bitmap = cls(values, copy_on_write=cow)
        result = bitmap.to_array()
        expected = array.array('I', sorted(values))
        self.assertEqual(result, expected)

    @given(bitmap_cls, st.booleans(), st.integers(min_value=0, max_value=100))
    def test_constructor_generator(self, cls, cow, size):
        def generator(n):
            for i in range(n):
                yield i
        bitmap = cls(generator(size), copy_on_write=cow)
        self.assertEqual(bitmap, cls(range(size), copy_on_write=cow))


class SelectRankTest(Util):

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_simple_select(self, cls, values, cow):
        bitmap = cls(values, copy_on_write=cow)
        values = list(bitmap)  # enforce sorted order
        for i in range(-len(values), len(values)):
            self.assertEqual(bitmap[i], values[i])

    @given(bitmap_cls, hyp_collection, uint32, st.booleans())
    def test_wrong_selection(self, cls, values, n, cow):
        bitmap = cls(values, cow)
        with self.assertRaises(IndexError):
            bitmap[len(values)]
        with self.assertRaises(IndexError):
            bitmap[n + len(values)]
        with self.assertRaises(IndexError):
            bitmap[-len(values) - 1]
        with self.assertRaises(IndexError):
            bitmap[-n - len(values) - 1]

    def check_slice(self, cls, values, start, stop, step, cow):
        bitmap = cls(values, copy_on_write=cow)
        values = list(bitmap)  # enforce sorted order
        expected = values[start:stop:step]
        expected.sort()
        observed = list(bitmap[start:stop:step])
        self.assertEqual(expected, observed)

    def slice_arg(n):
        return st.integers(min_value=-n, max_value=n)

    @given(bitmap_cls, hyp_collection, slice_arg(2**12), slice_arg(2**12), slice_arg(2**5), st.booleans())
    def test_slice_select_non_empty(self, cls, values, start, stop, step, cow):
        assume(step != 0)
        assume(len(range(start, stop, step)) > 0)
        self.check_slice(cls, values, start, stop, step, cow)

    @given(bitmap_cls, hyp_collection, slice_arg(2**12), slice_arg(2**12), slice_arg(2**5), st.booleans())
    def test_slice_select_empty(self, cls, values, start, stop, step, cow):
        assume(step != 0)
        assume(len(range(start, stop, step)) == 0)
        self.check_slice(cls, values, start, stop, step, cow)

    @given(bitmap_cls, hyp_collection, slice_arg(2**12) | st.none(), slice_arg(2**12) | st.none(), slice_arg(2**5) | st.none(), st.booleans())
    def test_slice_select_none(self, cls, values, start, stop, step, cow):
        assume(step != 0)
        self.check_slice(cls, values, start, stop, step, cow)

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_simple_rank(self, cls, values, cow):
        bitmap = cls(values, copy_on_write=cow)
        for i, value in enumerate(sorted(values)):
            self.assertEqual(bitmap.rank(value), i + 1)

    @given(bitmap_cls, hyp_collection, uint18, st.booleans())
    def test_general_rank(self, cls, values, element, cow):
        bitmap = cls(values, copy_on_write=cow)
        observed_rank = bitmap.rank(element)
        expected_rank = len([n for n in set(values) if n <= element])
        self.assertEqual(expected_rank, observed_rank)

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_min(self, cls, values, cow):
        assume(len(values) > 0)
        bitmap = cls(values, copy_on_write=cow)
        self.assertEqual(bitmap.min(), min(values))

    @given(bitmap_cls)
    def test_wrong_min(self, cls):
        bitmap = cls()
        with self.assertRaises(ValueError):
            bitmap.min()

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_max(self, cls, values, cow):
        assume(len(values) > 0)
        bitmap = cls(values, copy_on_write=cow)
        self.assertEqual(bitmap.max(), max(values))

    @given(bitmap_cls)
    def test_wrong_max(self, cls):
        bitmap = cls()
        with self.assertRaises(ValueError):
            bitmap.max()

    @given(bitmap_cls, hyp_collection, uint32, st.booleans())
    def test_next_set_bit(self, cls, values, other_value, cow):
        assume(len(values) > 0)
        bitmap = cls(values, copy_on_write=cow)
        try:
            expected = next(i for i in sorted(values) if i >= other_value)
            self.assertEqual(bitmap.next_set_bit(other_value), expected)
        except StopIteration:
            with self.assertRaises(ValueError):
                bitmap.next_set_bit(other_value)

    @given(bitmap_cls)
    def test_wrong_next_set_bit(self, cls):
        bitmap = cls()
        with self.assertRaises(ValueError):
            bitmap.next_set_bit(0)


class BinaryOperationsTest(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_binary_op(self, cls1, cls2, values1, values2, cow):
        for op in [operator.or_, operator.and_, operator.xor, operator.sub]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = cls1(values1, cow)
            self.bitmap2 = cls2(values2, cow)
            old_bitmap1 = cls1(self.bitmap1)
            old_bitmap2 = cls2(self.bitmap2)
            result_set = op(self.set1, self.set2)
            result_bitmap = op(self.bitmap1, self.bitmap2)
            self.assertEqual(self.bitmap1, old_bitmap1)
            self.assertEqual(self.bitmap2, old_bitmap2)
            self.compare_with_set(result_bitmap, result_set)
            self.assertEqual(type(self.bitmap1), type(result_bitmap))

    @given(bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_binary_op_inplace(self, cls2, values1, values2, cow):
        for op in [operator.ior, operator.iand, operator.ixor, operator.isub]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = BitMap(values1, cow)
            original = self.bitmap1
            self.bitmap2 = cls2(values2, cow)
            old_bitmap2 = cls2(self.bitmap2)
            op(self.set1, self.set2)
            op(self.bitmap1, self.bitmap2)
            self.assertIs(original, self.bitmap1)
            self.assertEqual(self.bitmap2, old_bitmap2)
            self.compare_with_set(self.bitmap1, self.set1)

    @given(bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_binary_op_inplace_frozen(self, cls2, values1, values2, cow):
        for op in [operator.ior, operator.iand, operator.ixor, operator.isub]:
            self.set1 = frozenset(values1)
            self.set2 = frozenset(values2)

            self.bitmap1 = FrozenBitMap(values1, cow)
            old_bitmap1 = FrozenBitMap(self.bitmap1)
            self.bitmap2 = cls2(values2, cow)
            old_bitmap2 = cls2(self.bitmap2)

            new_set = op(self.set1, self.set2)
            new_bitmap = op(self.bitmap1, self.bitmap2)

            self.assertEqual(self.bitmap1, old_bitmap1)
            self.assertEqual(self.bitmap2, old_bitmap2)

            self.compare_with_set(new_bitmap, new_set)


class ComparisonTest(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_comparison(self, cls1, cls2, values1, values2, cow):
        for op in [operator.le, operator.ge, operator.lt, operator.gt]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = cls1(values1, copy_on_write=cow)
            self.bitmap2 = cls2(values2, copy_on_write=cow)
            self.assertEqual(op(self.bitmap1, self.bitmap1),
                             op(self.set1, self.set1))
            self.assertEqual(op(self.bitmap1, self.bitmap2),
                             op(self.set1, self.set2))
            self.assertEqual(op(self.bitmap1 | self.bitmap2, self.bitmap2),
                             op(self.set1 | self.set2, self.set2))
            self.assertEqual(op(self.set1, self.set1 | self.set2),
                             op(self.set1, self.set1 | self.set2))

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_intersect(self, cls1, cls2, values1, values2, cow):
        bm1 = cls1(values1, copy_on_write=cow)
        bm2 = cls2(values2, copy_on_write=cow)
        self.assertEqual(bm1.intersect(bm2), len(bm1 & bm2) > 0)


class RangeTest(Util):
    @given(bitmap_cls, hyp_collection, st.booleans(), uint32, uint32)
    def test_contains_range_arbitrary(self, cls, values, cow, start, end):
        bm = cls(values)
        expected = (cls(range(start, end)) <= bm)
        self.assertEqual(expected, bm.contains_range(start, end))

    @given(bitmap_cls, st.booleans(), uint32, uint32)
    def test_contains_range(self, cls, cow, start, end):
        assume(start < end)
        self.assertTrue(cls(range(start, end)).contains_range(start, end))
        self.assertTrue(cls(range(start, end)).contains_range(start, end - 1))
        self.assertFalse(cls(range(start, end - 1)).contains_range(start, end))
        self.assertTrue(cls(range(start, end)).contains_range(start + 1, end))
        self.assertFalse(cls(range(start + 1, end)).contains_range(start, end))
        r = range(start, end)
        try:
            middle = r[len(r) // 2]  # on 32bits systems, this call might fail when len(r) is too large
        except OverflowError:
            if sys.maxsize > 2**32:
                raise
            else:
                return
        bm = cls(range(start, end)) - cls([middle])
        self.assertFalse(bm.contains_range(start, end))
        self.assertTrue(bm.contains_range(start, middle))
        self.assertTrue(bm.contains_range(middle + 1, end))

    @given(hyp_collection, st.booleans(), uint32, uint32)
    def test_add_remove_range(self, values, cow, start, end):
        assume(start < end)
        bm = BitMap(values, copy_on_write=cow)
        # Empty range
        original = BitMap(bm)
        bm.add_range(end, start)
        self.assertEqual(bm, original)
        bm.remove_range(end, start)
        self.assertEqual(bm, original)
        # Adding the range
        bm.add_range(start, end)
        self.assertTrue(bm.contains_range(start, end))
        self.assertEqual(bm.intersection_cardinality(BitMap(range(start, end), copy_on_write=cow)), end - start)
        # Empty range (again)
        original = BitMap(bm)
        bm.remove_range(end, start)
        self.assertEqual(bm, original)
        self.assertEqual(bm.intersection_cardinality(BitMap(range(start, end), copy_on_write=cow)), end - start)
        # Removing the range
        bm.remove_range(start, end)
        self.assertFalse(bm.contains_range(start, end))
        self.assertEqual(bm.intersection_cardinality(BitMap(range(start, end), copy_on_write=cow)), 0)

    @given(hyp_collection, st.booleans(), large_uint64, large_uint64)
    def test_large_values(self, values, cow, start, end):
        bm = BitMap(values, copy_on_write=cow)
        original = BitMap(bm)
        bm.add_range(start, end)
        self.assertEqual(bm, original)
        bm.remove_range(start, end)
        self.assertEqual(bm, original)
        self.assertTrue(bm.contains_range(start, end))


class CardinalityTest(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_cardinality(self, cls1, cls2, values1, values2, cow):

        for real_op, estimated_op in [
            (operator.or_, cls1.union_cardinality),
            (operator.and_, cls1.intersection_cardinality),
            (operator.sub, cls1.difference_cardinality),
            (operator.xor, cls1.symmetric_difference_cardinality),
        ]:
            self.bitmap1 = cls1(values1, copy_on_write=cow)
            self.bitmap2 = cls2(values2, copy_on_write=cow)
            real_value = len(real_op(self.bitmap1, self.bitmap2))
            estimated_value = estimated_op(self.bitmap1, self.bitmap2)
            self.assertEqual(real_value, estimated_value)

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_jaccard_index(self, cls1, cls2, values1, values2, cow):
        assume(len(values1) > 0 or len(values2) > 0)
        self.bitmap1 = cls1(values1, copy_on_write=cow)
        self.bitmap2 = cls2(values2, copy_on_write=cow)
        real_value = float(len(self.bitmap1 & self.bitmap2)) / \
            float(max(1, len(self.bitmap1 | self.bitmap2)))
        estimated_value = self.bitmap1.jaccard_index(self.bitmap2)
        self.assertAlmostEqual(real_value, estimated_value)

    @given(bitmap_cls, hyp_collection, uint32, uint32)
    def test_range_cardinality(self, cls, values, a, b):
        bm = cls(values)
        start, end = sorted([a, b])

        # make an intersection with the relevant range to test against
        test_bm = bm.intersection(BitMap(range(start, end)))

        self.assertEqual(len(test_bm), bm.range_cardinality(start, end))


class ManyOperationsTest(Util):

    @given(hyp_collection, hyp_many_collections, st.booleans())
    def test_update(self, initial_values, all_values, cow):
        self.initial_bitmap = BitMap(initial_values, copy_on_write=cow)
        self.all_bitmaps = [BitMap(values, copy_on_write=cow)
                            for values in all_values]
        self.initial_bitmap.update(*all_values)
        expected_result = functools.reduce(
            lambda x, y: x | y, self.all_bitmaps + [self.initial_bitmap])
        self.assertEqual(expected_result, self.initial_bitmap)
        self.assertEqual(type(expected_result), type(self.initial_bitmap))

    @given(hyp_collection, hyp_many_collections, st.booleans())
    def test_intersection_update(self, initial_values, all_values, cow):
        self.initial_bitmap = BitMap(initial_values, copy_on_write=cow)
        self.all_bitmaps = [BitMap(values, copy_on_write=cow)
                            for values in all_values]
        self.initial_bitmap.intersection_update(*all_values)
        expected_result = functools.reduce(
            lambda x, y: x & y, self.all_bitmaps + [self.initial_bitmap])
        self.assertEqual(expected_result, self.initial_bitmap)
        self.assertEqual(type(expected_result), type(self.initial_bitmap))

    @given(bitmap_cls, st.data(), hyp_many_collections, st.booleans())
    def test_union(self, cls, data, all_values, cow):
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values, copy_on_write=cow)
                            for i, values in enumerate(all_values)]
        result = cls.union(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x | y, self.all_bitmaps)
        self.assertEqual(expected_result, result)

    @given(bitmap_cls, st.data(), hyp_many_collections, st.booleans())
    def test_intersection(self, cls, data, all_values, cow):
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values, copy_on_write=cow)
                            for i, values in enumerate(all_values)]
        result = cls.intersection(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x & y, self.all_bitmaps)
        self.assertEqual(expected_result, result)

    @given(bitmap_cls, st.data(), hyp_many_collections, st.booleans())
    def test_difference(self, cls, data, all_values, cow):
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values, copy_on_write=cow)
                            for i, values in enumerate(all_values)]
        result = cls.difference(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x - y, self.all_bitmaps)
        self.assertEqual(expected_result, result)


class SerializationTest(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection)
    def test_serialization(self, cls1, cls2, values):
        old_bm = cls1(values)
        buff = old_bm.serialize()
        new_bm = cls2.deserialize(buff)
        self.assertEqual(old_bm, new_bm)
        self.assertIsInstance(new_bm, cls2)
        self.assert_is_not(old_bm, new_bm)

    @given(bitmap_cls, hyp_collection, st.integers(min_value=2, max_value=pickle.HIGHEST_PROTOCOL))
    def test_pickle_protocol(self, cls, values, protocol):
        old_bm = cls(values)
        pickled = pickle.dumps(old_bm, protocol=protocol)
        new_bm = pickle.loads(pickled)
        self.assertEqual(old_bm, new_bm)
        self.assert_is_not(old_bm, new_bm)


class StatisticsTest(Util):

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_basic_properties(self, cls, values, cow):
        bitmap = cls(values, copy_on_write=cow)
        stats = bitmap.get_statistics()
        self.assertEqual(stats['n_values_array_containers'] + stats['n_values_bitset_containers']
                         + stats['n_values_run_containers'], len(bitmap))
        self.assertEqual(stats['n_bytes_array_containers'],
                         2 * stats['n_values_array_containers'])
        self.assertEqual(stats['n_bytes_bitset_containers'],
                         2**13 * stats['n_bitset_containers'])
        if len(values) > 0:
            self.assertEqual(stats['min_value'], bitmap[0])
            self.assertEqual(stats['max_value'], bitmap[len(bitmap) - 1])
        self.assertEqual(stats['cardinality'], len(bitmap))
        self.assertEqual(stats['sum_value'], sum(values))

    @given(bitmap_cls)
    def test_implementation_properties_array(self, cls):
        values = range(2**16 - 10, 2**16 + 10, 2)
        stats = cls(values).get_statistics()
        self.assertEqual(stats['n_array_containers'], 2)
        self.assertEqual(stats['n_bitset_containers'], 0)
        self.assertEqual(stats['n_run_containers'], 0)
        self.assertEqual(stats['n_values_array_containers'], len(values))
        self.assertEqual(stats['n_values_bitset_containers'], 0)
        self.assertEqual(stats['n_values_run_containers'], 0)

    @given(bitmap_cls)
    def test_implementation_properties_bitset(self, cls):
        values = range(2**0, 2**17, 2)
        stats = cls(values).get_statistics()
        self.assertEqual(stats['n_array_containers'], 0)
        self.assertEqual(stats['n_bitset_containers'], 2)
        self.assertEqual(stats['n_run_containers'], 0)
        self.assertEqual(stats['n_values_array_containers'], 0)
        self.assertEqual(stats['n_values_bitset_containers'], len(values))
        self.assertEqual(stats['n_values_run_containers'], 0)

    @given(bitmap_cls)
    def test_implementation_properties_run(self, cls):
        values = range(2**0, 2**17, 1)
        stats = cls(values).get_statistics()
        self.assertEqual(stats['n_array_containers'], 0)
        self.assertEqual(stats['n_bitset_containers'], 0)
        self.assertEqual(stats['n_run_containers'], 2)
        self.assertEqual(stats['n_values_array_containers'], 0)
        self.assertEqual(stats['n_values_bitset_containers'], 0)
        self.assertEqual(stats['n_values_run_containers'], len(values))
        self.assertEqual(stats['n_bytes_run_containers'], 12)


class FlipTest(Util):

    def check_flip(self, bm_before, bm_after, start, end):
        size = 100
        iter_range = random.sample(
            range(start, end), min(size, len(range(start, end))))
        iter_before = self.bitmap_sample(bm_before, min(size, len(bm_before)))
        iter_after = self.bitmap_sample(bm_after, min(size, len(bm_after)))
        for elt in iter_range:
            if elt in bm_before:
                self.assertNotIn(elt, bm_after)
            else:
                self.assertIn(elt, bm_after)
        for elt in iter_before:
            if not (start <= elt < end):
                self.assertIn(elt, bm_after)
        for elt in iter_after:
            if not (start <= elt < end):
                self.assertIn(elt, bm_before)

    @given(bitmap_cls, hyp_collection, integer, integer, st.booleans())
    def test_flip_empty(self, cls, values, start, end, cow):
        assume(start >= end)
        bm_before = cls(values, copy_on_write=cow)
        bm_copy = cls(bm_before)
        bm_after = bm_before.flip(start, end)
        self.assertEqual(bm_before, bm_copy)
        self.assertEqual(bm_before, bm_after)

    @given(bitmap_cls, hyp_collection, integer, integer, st.booleans())
    def test_flip(self, cls, values, start, end, cow):
        assume(start < end)
        bm_before = cls(values, copy_on_write=cow)
        bm_copy = cls(bm_before)
        bm_after = bm_before.flip(start, end)
        self.assertEqual(bm_before, bm_copy)
        self.check_flip(bm_before, bm_after, start, end)

    @given(hyp_collection, integer, integer, st.booleans())
    def test_flip_inplace_empty(self, values, start, end, cow):
        assume(start >= end)
        bm_before = BitMap(values, copy_on_write=cow)
        bm_after = BitMap(bm_before)
        bm_after.flip_inplace(start, end)
        self.assertEqual(bm_before, bm_after)

    @given(hyp_collection, integer, integer, st.booleans())
    def test_flip_inplace(self, values, start, end, cow):
        assume(start < end)
        bm_before = BitMap(values, copy_on_write=cow)
        bm_after = BitMap(bm_before)
        bm_after.flip_inplace(start, end)
        self.check_flip(bm_before, bm_after, start, end)


class ShiftTest(Util):
    @given(bitmap_cls, hyp_collection, int64, st.booleans())
    def test_shift(self, cls, values, offset, cow):
        bm_before = cls(values, copy_on_write=cow)
        bm_copy = cls(bm_before)
        bm_after = bm_before.shift(offset)
        self.assertEqual(bm_before, bm_copy)
        expected = cls([val + offset for val in values if val + offset in range(0, 2**32)], copy_on_write=cow)
        self.assertEqual(bm_after, expected)


class IncompatibleInteraction(Util):

    def incompatible_op(self, op):
        for cow1, cow2 in [(True, False), (False, True)]:
            bm1 = BitMap(copy_on_write=cow1)
            bm2 = BitMap(copy_on_write=cow2)
            with self.assertRaises(ValueError):
                op(bm1, bm2)

    def test_incompatible_or(self):
        self.incompatible_op(lambda x, y: x | y)

    def test_incompatible_and(self):
        self.incompatible_op(lambda x, y: x & y)

    def test_incompatible_xor(self):
        self.incompatible_op(lambda x, y: x ^ y)

    def test_incompatible_sub(self):
        self.incompatible_op(lambda x, y: x - y)

    def test_incompatible_or_inplace(self):
        self.incompatible_op(lambda x, y: x.__ior__(y))

    def test_incompatible_and_inplace(self):
        self.incompatible_op(lambda x, y: x.__iand__(y))

    def test_incompatible_xor_inplace(self):
        self.incompatible_op(lambda x, y: x.__ixor__(y))

    def test_incompatible_sub_inplace(self):
        self.incompatible_op(lambda x, y: x.__isub__(y))

    def test_incompatible_eq(self):
        self.incompatible_op(lambda x, y: x == y)

    def test_incompatible_neq(self):
        self.incompatible_op(lambda x, y: x != y)

    def test_incompatible_le(self):
        self.incompatible_op(lambda x, y: x <= y)

    def test_incompatible_lt(self):
        self.incompatible_op(lambda x, y: x < y)

    def test_incompatible_ge(self):
        self.incompatible_op(lambda x, y: x >= y)

    def test_incompatible_gt(self):
        self.incompatible_op(lambda x, y: x > y)

    def test_incompatible_intersect(self):
        self.incompatible_op(lambda x, y: x.intersect(y))

    def test_incompatible_union(self):
        self.incompatible_op(lambda x, y: BitMap.union(x, y))
        self.incompatible_op(lambda x, y: BitMap.union(x, x, y, y, x, x, y, y))

    def test_incompatible_or_card(self):
        self.incompatible_op(lambda x, y: x.union_cardinality(y))

    def test_incompatible_and_card(self):
        self.incompatible_op(lambda x, y: x.intersection_cardinality(y))

    def test_incompatible_xor_card(self):
        self.incompatible_op(
            lambda x, y: x.symmetric_difference_cardinality(y))

    def test_incompatible_sub_card(self):
        self.incompatible_op(lambda x, y: x.difference_cardinality(y))

    def test_incompatible_jaccard(self):
        self.incompatible_op(lambda x, y: x.jaccard_index(y))


class BitMapTest(unittest.TestCase):
    @given(hyp_collection, uint32)
    def test_iter_equal_or_larger(self, values, other_value):
        bm = BitMap(values)
        bm_iter = bm.iter_equal_or_larger(other_value)
        expected = [i for i in values if i >= other_value]
        expected.sort()

        observed = list(bm_iter)
        self.assertEqual(expected, observed)

    def test_unashability(self):
        bm = BitMap()
        with self.assertRaises(TypeError):
            hash(bm)


class FrozenTest(unittest.TestCase):

    @given(hyp_collection, hyp_collection, integer)
    def test_immutability(self, values, other, number):
        frozen = FrozenBitMap(values)
        copy = FrozenBitMap(values)
        other = BitMap(other)
        with self.assertRaises(AttributeError):
            frozen.clear()
        with self.assertRaises(AttributeError):
            frozen.pop()
        with self.assertRaises(AttributeError):
            frozen.add(number)
        with self.assertRaises(AttributeError):
            frozen.update(other)
        with self.assertRaises(AttributeError):
            frozen.discard(number)
        with self.assertRaises(AttributeError):
            frozen.remove(number)
        with self.assertRaises(AttributeError):
            frozen.intersection_update(other)
        with self.assertRaises(AttributeError):
            frozen.difference_update(other)
        with self.assertRaises(AttributeError):
            frozen.symmetric_difference_update(other)
        with self.assertRaises(AttributeError):
            frozen.update(number, number + 10)
        with self.assertRaises(AttributeError):
            frozen.overwrite(other)
        self.assertEqual(frozen, copy)

    @given(hyp_collection, hyp_collection)
    def test_hash_uneq(self, values1, values2):
        """This test as a non null (but extremly low) probability to fail."""
        bitmap1 = FrozenBitMap(values1)
        bitmap2 = FrozenBitMap(values2)
        assume(bitmap1 != bitmap2)
        h1 = hash(bitmap1)
        h2 = hash(bitmap2)
        hd = hash(bitmap1 ^ bitmap2)
        hashes = [h1, h2, hd]
        nb_collisions = len(hashes) - len(set(hashes))
        self.assertGreaterEqual(1, nb_collisions)

    @given(hyp_collection)
    def test_hash_eq(self, values):
        bitmap1 = FrozenBitMap(values)
        bitmap2 = FrozenBitMap(values)
        bitmap3 = FrozenBitMap(bitmap1)
        self.assertEqual(hash(bitmap1), hash(bitmap2))
        self.assertEqual(hash(bitmap1), hash(bitmap3))

    def test_hash_eq2(self):
        """It can happen that two bitmaps hold the same values but have a different data structure. They should still
        have a same hash.
        This test compares two bitmaps with the same values, one has a run container, the other has an array container."""
        n = 100
        bm1 = FrozenBitMap(range(n))
        bm2 = BitMap()
        for i in range(n):
            bm2.add(i)
        bm2 = FrozenBitMap(bm2, optimize=False)
        self.assertEqual(bm1, bm2)
        self.assertNotEqual(bm1.get_statistics(), bm2.get_statistics())
        self.assertEqual(hash(bm1), hash(bm2))


class OptimizationTest(unittest.TestCase):

    @given(bitmap_cls)
    def test_run_optimize(self, cls):
        bm1 = BitMap()
        size = 1000
        for i in range(size):
            bm1.add(i)
        bm2 = cls(bm1, optimize=False)
        stats = bm2.get_statistics()
        self.assertEqual(bm1.get_statistics(), stats)
        self.assertEqual(stats['n_containers'], stats['n_array_containers'])
        self.assertEqual(stats['n_values_array_containers'], size)
        self.assertTrue(bm2.run_optimize())
        stats = bm2.get_statistics()
        self.assertEqual(stats['n_containers'], stats['n_run_containers'])
        self.assertEqual(stats['n_values_run_containers'], size)
        bm3 = cls(bm1)  # optimize is True by default
        self.assertEqual(stats, bm3.get_statistics())

    @given(bitmap_cls)
    def test_shrink_to_fit(self, cls):
        bm1 = BitMap()
        size = 1000
        for i in range(size):
            bm1.add(i)
        bm2 = cls(bm1, optimize=False)
        self.assertGreater(bm2.shrink_to_fit(), 0)
        self.assertEqual(bm2.shrink_to_fit(), 0)
        bm3 = cls(bm1, optimize=True)
        self.assertEqual(bm3.shrink_to_fit(), 0)


small_integer = st.integers(min_value=0, max_value=200)
small_integer_list = st.lists(min_size=0, max_size=2000, elements=small_integer)


class PythonSetEquivalentTest(unittest.TestCase):
    """
    The main goal of this class is to make sure the BitMap api is a superset of the python builtin set api.
    """

    @given(bitmap_cls, small_integer_list, st.booleans())
    def test_convert_to_set(self, BitMapClass, list1, cow):
        """
        Most of the tests depend on a working implementation for converting from BitMap to python set.
        This test sanity checks it.

        This test should be modified or removed if you want to run PythonSetEquivalentTest with integers drawn from
        a larger set than `small_integer`. It will become prohibitively time-consuming.
        """
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        b1 = BitMapClass(list1, copy_on_write=cow)

        converted_set = SetClass(b1)

        try:
            min_value = min(s1)
        except ValueError:
            min_value = 0

        try:
            max_value = max(s1) + 1
        except ValueError:
            max_value = 200 + 1

        for i in range(min_value, max_value):
            self.assertEqual(i in s1, i in converted_set)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_difference(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.difference(s2), set(b1.difference(b2)))
        self.assertEqual(SetClass.difference(s1, s2), set(BitMapClass.difference(b1, b2)))
        self.assertEqual((s1 - s2), set(b1 - b2))
        self.assertEqual(b1 - b2, b1.difference(b2))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_symmetric_difference(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.symmetric_difference(s2), set(b1.symmetric_difference(b2)))
        self.assertEqual(SetClass.symmetric_difference(s1, s2), set(BitMapClass.symmetric_difference(b1, b2)))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_union(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.union(s2), set(b1.union(b2)))
        self.assertEqual(SetClass.union(s1, s2), set(BitMapClass.union(b1, b2)))
        self.assertEqual((s1 | s2), set(b1 | b2))
        self.assertEqual(b1 | b2, b1.union(b2))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_issubset(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.issubset(s2), b1.issubset(b2))
        self.assertEqual(SetClass.issubset(s1, s2), BitMapClass.issubset(b1, b2))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_le(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.__le__(s2), b1.__le__(b2))
        self.assertEqual(SetClass.__le__(s1, s2), BitMapClass.__le__(b1, b2))
        self.assertEqual((s1 <= s2), (b1 <= b2))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_ge(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.__ge__(s2), b1.__ge__(b2))
        self.assertEqual(SetClass.__ge__(s1, s2), BitMapClass.__ge__(b1, b2))
        self.assertEqual((s1 >= s2), (b1 >= b2))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_eq(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()
        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.__eq__(s2), b1.__eq__(b2))
        self.assertEqual(SetClass.__eq__(s1, s2), BitMapClass.__eq__(b1, b2))
        self.assertEqual((s1 == s2), (b1 == b2))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_issuperset(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.issuperset(s2), b1.issuperset(b2))
        self.assertEqual(SetClass.issuperset(s1, s2), BitMapClass.issuperset(b1, b2))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_isdisjoint(self, BitMapClass, list1, list2, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.isdisjoint(s2), b1.isdisjoint(b2))
        self.assertEqual(SetClass.isdisjoint(s1, s2), BitMapClass.isdisjoint(b1, b2))

    @given(small_integer_list, st.booleans())
    def test_clear(self, list1, cow):
        b1 = BitMap(list1, copy_on_write=cow)
        b1.clear()
        self.assertEqual(len(b1), 0)

    @given(small_integer_list, st.booleans())
    def test_pop(self, list1, cow):
        b1 = BitMap(list1, copy_on_write=cow)
        starting_length = len(b1)
        if starting_length >= 1:
            popped_element = b1.pop()
            self.assertEqual(len(b1), starting_length - 1)  # length decreased by one
            self.assertFalse(popped_element in b1)  # and element isn't in the BitMap anymore
        else:
            with self.assertRaises(KeyError):
                b1.pop()

    @given(bitmap_cls, small_integer_list, st.booleans())
    def test_copy(self, BitMapClass, list1, cow):
        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = b1.copy()
        self.assertEqual(b2, b1)

    @given(small_integer_list, st.booleans())
    def test_copy_writable(self, list1, cow):
        b1 = BitMap(list1, copy_on_write=cow)
        b2 = b1.copy()

        try:
            new_element = max(b1) + 1  # doesn't exist in the set
        except ValueError:
            new_element = 1

        b2.add(new_element)

        self.assertTrue(new_element in b2)
        self.assertTrue(new_element not in b1)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_overwrite(self, BitMapClass, list1, list2, cow):
        assume(set(list1) != set(list2))
        b1 = BitMap(list1, copy_on_write=cow)
        orig1 = b1.copy()
        b2 = BitMapClass(list2, copy_on_write=cow)
        orig2 = b2.copy()
        b1.overwrite(b2)
        self.assertEqual(b1, b2)       # the two bitmaps are now equal
        self.assertNotEqual(b1, orig1)  # the first bitmap has been modified
        self.assertEqual(b2, orig2)    # the second bitmap was left untouched
        with self.assertRaises(ValueError):
            b1.overwrite(b1)

    @given(small_integer_list, small_integer_list, st.booleans())
    def test_difference_update(self, list1, list2, cow):
        s1 = set(list1)
        s2 = set(list2)
        s1.difference_update(s2)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b1.difference_update(b2)

        self.assertEqual(s1, set(b1))

    @given(small_integer_list, small_integer_list, st.booleans())
    def test_symmetric_difference_update(self, list1, list2, cow):
        s1 = set(list1)
        s2 = set(list2)
        s1.symmetric_difference_update(s2)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b1.symmetric_difference_update(b2)

        self.assertEqual(s1, set(b1))

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_dunder(self, BitMapClass, list1, list2, cow):
        """
        Tests for &|^-
        """
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        self.assertEqual(s1.__and__(s2), SetClass(b1.__and__(b2)))
        self.assertEqual(s1.__or__(s2), SetClass(b1.__or__(b2)))
        self.assertEqual(s1.__xor__(s2), SetClass(b1.__xor__(b2)))
        self.assertEqual(s1.__sub__(s2), SetClass(b1.__sub__(b2)))

    @given(small_integer_list, small_integer, st.booleans())
    def test_add(self, list1, value, cow):
        s1 = set(list1)
        b1 = BitMap(list1, copy_on_write=cow)
        self.assertEqual(s1, set(b1))

        s1.add(value)
        b1.add(value)
        self.assertEqual(s1, set(b1))

    @given(small_integer_list, small_integer, st.booleans())
    def test_discard(self, list1, value, cow):
        s1 = set(list1)
        b1 = BitMap(list1, copy_on_write=cow)
        self.assertEqual(s1, set(b1))

        s1.discard(value)
        b1.discard(value)
        self.assertEqual(s1, set(b1))

    @given(small_integer_list, small_integer, st.booleans())
    def test_remove(self, list1, value, cow):
        s1 = set(list1)
        b1 = BitMap(list1, copy_on_write=cow)
        self.assertEqual(s1, set(b1))

        s1_raised = False
        b1_raised = False
        try:
            s1.remove(value)
        except KeyError:
            s1_raised = True

        try:
            b1.remove(value)
        except KeyError:
            b1_raised = True

        self.assertEqual(s1, set(b1))
        self.assertEqual(s1_raised, b1_raised)  # Either both raised exception or neither did

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_union(self, BitMapClass, list1, list2, list3, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)
        s3 = SetClass(list3)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)
        b3 = BitMapClass(list3, copy_on_write=cow)

        self.assertEqual(SetClass.union(s1, s2, s3), SetClass(BitMapClass.union(b1, b2, b3)))
        self.assertEqual(s1.union(s2, s3), SetClass(b1.union(b2, b3)))

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_difference(self, BitMapClass, list1, list2, list3, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)
        s3 = SetClass(list3)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)
        b3 = BitMapClass(list3, copy_on_write=cow)

        self.assertEqual(SetClass.difference(s1, s2, s3), SetClass(BitMapClass.difference(b1, b2, b3)))
        self.assertEqual(s1.difference(s2, s3), SetClass(b1.difference(b2, b3)))

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_intersection(self, BitMapClass, list1, list2, list3, cow):
        if BitMapClass == BitMap:
            SetClass = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)
        s3 = SetClass(list3)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)
        b3 = BitMapClass(list3, copy_on_write=cow)

        self.assertEqual(SetClass.intersection(s1, s2, s3), SetClass(BitMapClass.intersection(b1, b2, b3)))
        self.assertEqual(s1.intersection(s2, s3), SetClass(b1.intersection(b2, b3)))

    @given(small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_intersection_update(self, list1, list2, list3, cow):
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        set.intersection_update(s1, s2, s3)
        BitMap.intersection_update(b1, b2, b3)
        self.assertEqual(s1, set(b1))

        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        s1.intersection_update(s2, s3)
        b1.intersection_update(b2, b3)

        self.assertEqual(s1, set(b1))

    @given(small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_difference_update(self, list1, list2, list3, cow):
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        set.difference_update(s1, s2, s3)
        BitMap.difference_update(b1, b2, b3)
        self.assertEqual(s1, set(b1))

        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        s1.difference_update(s2, s3)
        b1.difference_update(b2, b3)

        self.assertEqual(s1, set(b1))

    @given(small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_update(self, list1, list2, list3, cow):
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        set.update(s1, s2, s3)
        BitMap.update(b1, b2, b3)
        self.assertEqual(s1, set(b1))

        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        s1.update(s2, s3)
        b1.update(b2, b3)

        self.assertEqual(s1, set(b1))


small_list_of_uin32 = st.lists(min_size=0, max_size=400, elements=uint32)
large_list_of_uin32 = st.lists(min_size=600, max_size=1000, elements=uint32, unique=True)


class StringTest(unittest.TestCase):

    @given(bitmap_cls, small_list_of_uin32)
    def test_small_list(self, cls, collection):
        # test that repr for a small bitmap is equal to the original bitmap
        bm = cls(collection)
        self.assertEqual(bm, eval(repr(bm)))

    @settings(suppress_health_check=HealthCheck.all())
    @given(bitmap_cls, large_list_of_uin32)
    def test_large_list(self, cls, collection):
        # test that for a large bitmap the both the start and the end of the bitmap get printed

        bm = cls(collection)
        s = repr(bm)
        nondigits = set(s) - set('0123456789\n.')
        for x in nondigits:
            s = s.replace(x, ' ')

        small, large = s.split('...')
        small_ints = [int(i) for i in small.split()]
        large_ints = [int(i) for i in large.split()]

        for i in small_ints:
            self.assertIn(i, bm)

        for i in large_ints:
            self.assertIn(i, bm)

        self.assertEqual(min(small_ints), min(bm))
        self.assertEqual(max(large_ints), max(bm))


class VersionTest(unittest.TestCase):
    def assert_regex(self, pattern, text):
        matches = re.findall(pattern, text)
        if len(matches) != 1 or matches[0] != text:
            self.fail('Regex "%s" does not match text "%s".' % (pattern, text))

    def test_version(self):
        self.assert_regex(r'\d+\.\d+\.\d+(?:\.dev\d+)?', pyroaring.__version__)
        self.assert_regex(r'v\d+\.\d+\.\d+', pyroaring.__croaring_version__)


if __name__ == "__main__":
    unittest.main()
