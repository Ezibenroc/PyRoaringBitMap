#! /usr/bin/env python3

import unittest
import random
import functools
import os
from hypothesis import given
import hypothesis.strategies as st
from pyroaring import BitMap, load, dump

uint18 = st.integers(min_value=0, max_value=2**18)
uint32 = st.integers(min_value=0, max_value=2**32-1)
hyp_set = st.sets(uint32,
                 min_size=0, max_size=2**14, average_size=2**10)

range_max_size = 2**18

range_big_step = uint18.flatmap(lambda n:
    st.builds(range, st.just(n),
                     st.integers(min_value=n+1, max_value=n+range_max_size),
                     st.integers(min_value=2**8, max_value=range_max_size//8)))

range_small_step = uint18.flatmap(lambda n:
    st.builds(range, st.just(n),
                     st.integers(min_value=n+1, max_value=n+range_max_size),
                     st.integers(min_value=1, max_value=2**8)))

range_power2_step = uint18.flatmap(lambda n:
     st.builds(range, st.just(n),
                      st.integers(min_value=n+1, max_value=n+range_max_size),
                      st.integers(min_value=0, max_value=8).flatmap(
                        lambda n: st.just(2**n)
                      )))

hyp_range = range_big_step | range_small_step | range_power2_step
hyp_many_ranges = st.lists(hyp_range, min_size=1, max_size=20)
hyp_set = st.builds(set, hyp_range)
hyp_collection = hyp_range | hyp_set

class Util(unittest.TestCase):

    comparison_set = random.sample(range(2**8), 100) + random.sample(range(2**32-1), 50)

    def compare_with_set(self, bitmap, expected_set):
        self.assertEqual(len(bitmap), len(expected_set))
        self.assertEqual(set(bitmap), expected_set)
        self.assertEqual(sorted(list(bitmap)), sorted(list(expected_set)))
        self.assertEqual(BitMap(expected_set), bitmap)
        for value in self.comparison_set:
            if value in expected_set:
                self.assertIn(value, bitmap)
            else:
                self.assertNotIn(value, bitmap)

class BasicTest(Util):

    @given(hyp_range)
    def test_basic(self, values):
        bitmap = BitMap()
        expected_set = set()
        self.compare_with_set(bitmap, expected_set)
        values = list(values)
        random.shuffle(values)
        size = len(values)
        for value in values[:size//2]:
            bitmap.add(value)
            expected_set.add(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[size//2:]:
            bitmap.add(value)
            expected_set.add(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[:size//2]:
            bitmap.remove(value)
            expected_set.remove(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[size//2:]:
            bitmap.remove(value)
            expected_set.remove(value)
        self.compare_with_set(bitmap, expected_set)


    @given(hyp_range)
    def test_bitmap_equality(self, values):
        bitmap1 = BitMap(values)
        bitmap2 = BitMap(values)
        self.assertEqual(bitmap1, bitmap2)

    @given(hyp_range, hyp_range)
    def test_bitmap_unequality(self, values1, values2):
        st.assume(values1 != values2)
        bitmap1 = BitMap(values1)
        bitmap2 = BitMap(values2)
        self.assertNotEqual(bitmap1, bitmap2)

    @given(hyp_collection)
    def test_constructor_values(self, values):
        bitmap = BitMap(values)
        expected_set = set(values)
        self.compare_with_set(bitmap, expected_set)

    @given(hyp_range, uint32)
    def test_constructor_copy(self, values, other_value):
        st.assume(other_value not in values)
        bitmap1 = BitMap(values)
        bitmap2 = BitMap(bitmap1)
        self.assertEqual(bitmap1, bitmap2)
        bitmap1.add(other_value)
        self.assertNotEqual(bitmap1, bitmap2)

    def wrong_op(self, op):
        bitmap = BitMap()
        with self.assertRaises(ValueError):
            op(bitmap, -3)
        with self.assertRaises(ValueError):
            op(bitmap, 2**33)
        with self.assertRaises(ValueError):
            op(bitmap, 'bla')

    def test_wrong_add(self):
        self.wrong_op(lambda bitmap, value: bitmap.add(value))

    def test_wrong_contain(self):
        self.wrong_op(lambda bitmap, value: bitmap.__contains__(value))

    def test_wrong_constructor_values(self):
        with self.assertRaises(ValueError):
            bitmap = BitMap([3, 1, -3, 42])
        with self.assertRaises(ValueError):
            bitmap = BitMap([3, 2**33, 3, 42])
        with self.assertRaises(ValueError):
            bitmap = BitMap([3, 'bla', 3, 42])
        with self.assertRaises(ValueError):
            bitmap = BitMap(range(0, 10, 0))
        with self.assertRaises(ValueError):
            bitmap = BitMap(range(10, 0, 1))

class SelectTest(Util):

    @given(hyp_range)
    def test_simple_select(self, values):
        bitmap = BitMap(values)
        for i, value in enumerate(values):
            self.assertEqual(bitmap[i], value)

    @given(hyp_range, uint32)
    def test_wrong_selection(self, values, n):
        bitmap = BitMap(values)
        with self.assertRaises(ValueError):
            bitmap[len(values)]
        with self.assertRaises(ValueError):
            bitmap[n + len(values)]

class BinaryOperationsTest(Util):

    @given(hyp_range, hyp_range)
    def setUp(self, values1, values2):
        self.set1 = set(values1)
        self.set2 = set(values2)
        self.bitmap1 = BitMap(values1)
        self.bitmap2 = BitMap(values2)

    def do_test_binary_op(self, op):
        old_bitmap1 = BitMap(self.bitmap1)
        old_bitmap2 = BitMap(self.bitmap2)
        result_set = op(self.set1, self.set2)
        result_bitmap = op(self.bitmap1, self.bitmap2)
        self.assertEqual(self.bitmap1, old_bitmap1)
        self.assertEqual(self.bitmap2, old_bitmap2)
        self.compare_with_set(result_bitmap, result_set)

    def test_or(self):
        self.do_test_binary_op(lambda x,y : x|y)

    def test_and(self):
        self.do_test_binary_op(lambda x,y : x&y)

    def do_test_binary_op_inplace(self, op):
        old_bitmap2 = BitMap(self.bitmap2)
        op(self.set1, self.set2)
        op(self.bitmap1, self.bitmap2)
        self.assertEqual(self.bitmap2, old_bitmap2)
        self.compare_with_set(self.bitmap1, self.set1)

    def test_or_inplace(self):
        self.do_test_binary_op_inplace(lambda x,y : x.__ior__(y))

    def test_and_inplace(self):
        self.do_test_binary_op_inplace(lambda x,y : x.__iand__(y))

class ManyOperationsTest(Util):

    @given(hyp_many_ranges)
    def test_or_many(self, all_values):
        bitmaps = [BitMap(values) for values in all_values]
        bitmaps_copy = [BitMap(bm) for bm in bitmaps]
        result = BitMap.or_many(bitmaps)
        self.assertEqual(bitmaps_copy, bitmaps)
        expected_result = functools.reduce(lambda x, y: x|y, bitmaps)
        self.assertEqual(expected_result, result)

class SerializationTest(Util):

    @given(hyp_range, uint32)
    def test_serialization(self, values, other_value):
        st.assume(other_value not in values)
        old_bm = BitMap(values)
        buff = old_bm.serialize()
        new_bm = BitMap.deserialize(buff)
        self.assertEqual(old_bm, new_bm)
        old_bm.add(other_value)
        self.assertNotEqual(old_bm, new_bm)

    @given(hyp_range)
    def test_load_dump(self, values):
        filepath = 'testfile'
        old_bm = BitMap(values)
        with open(filepath, 'wb') as f:
            dump(f, old_bm)
        with open(filepath, 'rb') as f:
            new_bm = load(f)
        self.assertEqual(old_bm, new_bm)
        os.remove(filepath)

if __name__ == "__main__":
    unittest.main()
