#! /usr/bin/env python3

import unittest
import random
import functools
import os
import sys
import pickle
from hypothesis import given
import hypothesis.strategies as st
from pyroaring import BitMap, load, dump

is_python2 = sys.version_info < (3, 0)

try: # Python2 compatibility
    range = xrange
except NameError:
    pass

uint18 = st.integers(min_value=0, max_value=2**18)
uint32 = st.integers(min_value=0, max_value=2**32-1)
integer = st.integers(min_value=0, max_value=2**31-1)

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
hyp_set = st.builds(set, hyp_range) # would be great to build a true random set, but it takes too long and hypothesis does a timeout...
hyp_collection = hyp_range | hyp_set
hyp_many_collections = st.lists(hyp_collection, min_size=1, max_size=20)

class Util(unittest.TestCase):

    comparison_set = random.sample(range(2**8), 100) + random.sample(range(2**31-1), 50)

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

    @staticmethod
    def bitmap_sample(bitmap, size):
        indices = random.sample(range(len(bitmap)), size)
        return [bitmap[i] for i in indices]

class BasicTest(Util):

    @given(hyp_collection)
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


    @given(hyp_collection)
    def test_bitmap_equality(self, values):
        bitmap1 = BitMap(values)
        bitmap2 = BitMap(values)
        self.assertEqual(bitmap1, bitmap2)

    @given(hyp_collection, hyp_collection)
    def test_bitmap_unequality(self, values1, values2):
        st.assume(set(values1) != set(values2))
        bitmap1 = BitMap(values1)
        bitmap2 = BitMap(values2)
        self.assertNotEqual(bitmap1, bitmap2)

    @given(hyp_collection)
    def test_constructor_values(self, values):
        bitmap = BitMap(values)
        expected_set = set(values)
        self.compare_with_set(bitmap, expected_set)

    @given(hyp_collection, uint32)
    def test_constructor_copy(self, values, other_value):
        st.assume(other_value not in values)
        bitmap1 = BitMap(values)
        bitmap2 = BitMap(bitmap1)
        self.assertEqual(bitmap1, bitmap2)
        bitmap1.add(other_value)
        self.assertNotEqual(bitmap1, bitmap2)

    @given(hyp_collection, hyp_collection)
    def test_update(self, initial_values, new_values):
        bm = BitMap(initial_values)
        expected = BitMap(bm)
        bm.update(new_values)
        expected |= BitMap(new_values)
        self.assertEqual(bm, expected)

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
        with self.assertRaises(TypeError): # this should fire a type error!
            bitmap = BitMap([3, 'bla', 3, 42])
        with self.assertRaises(ValueError):
            bitmap = BitMap(range(0, 10, 0))
        with self.assertRaises(ValueError):
            bitmap = BitMap(range(10, 0, 1))

class SelectRankTest(Util):

    @given(hyp_collection)
    def test_simple_select(self, values):
        bitmap = BitMap(values)
        for i, value in enumerate(sorted(values)):
            self.assertEqual(bitmap[i], value)

    @given(hyp_collection, uint32)
    def test_wrong_selection(self, values, n):
        bitmap = BitMap(values)
        with self.assertRaises(ValueError):
            bitmap[len(values)]
        with self.assertRaises(ValueError):
            bitmap[n + len(values)]

    @given(hyp_collection)
    def test_simple_rank(self, values):
        bitmap = BitMap(values)
        for i, value in enumerate(sorted(values)):
            self.assertEqual(bitmap.rank(value), i+1)

    @given(hyp_collection, uint18)
    def test_general_rank(self, values, element):
        bitmap = BitMap(values)
        observed_rank = bitmap.rank(element)
        expected_rank = len([n for n in set(values) if n <= element])
        self.assertEqual(expected_rank, observed_rank)

    @given(hyp_collection)
    def test_min(self, values):
        bitmap = BitMap(values)
        self.assertEqual(bitmap.min(), min(values))

    def test_wrong_min(self):
        bitmap = BitMap()
        with self.assertRaises(ValueError):
            m = bitmap.min()

    @given(hyp_collection)
    def test_max(self, values):
        bitmap = BitMap(values)
        self.assertEqual(bitmap.max(), max(values))

    def test_wrong_max(self):
        bitmap = BitMap()
        with self.assertRaises(ValueError):
            m = bitmap.max()

class BinaryOperationsTest(Util):

    @given(hyp_collection, hyp_collection)
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

    def test_xor(self):
        self.do_test_binary_op(lambda x,y : x^y)

    def test_sub(self):
        self.do_test_binary_op(lambda x,y : x-y)

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

    def test_xor_inplace(self):
        self.do_test_binary_op_inplace(lambda x,y : x.__ixor__(y))

    def test_sub_inplace(self):
        self.do_test_binary_op_inplace(lambda x,y : x.__isub__(y))

class ComparisonTest(Util):

    def do_test(self, values1, values2, op):
        self.assertEqual(op(BitMap(values1), BitMap(values1)),
                         op(set(values1), set(values1)))
        self.assertEqual(op(BitMap(values1), BitMap(values2)),
                         op(set(values1), set(values2)))
        self.assertEqual(op(BitMap(values1)|BitMap(values2), BitMap(values2)),
                         op(set(values1)|set(values2), set(values2)))
        self.assertEqual(op(BitMap(values1), BitMap(values1)|BitMap(values2)),
                         op(set(values1), set(values1)|set(values2)))

    @given(hyp_collection, hyp_collection)
    def test_le(self, values1, values2):
        self.do_test(values1, values2, lambda x,y: x <= y)

    @given(hyp_collection, hyp_collection)
    def test_ge(self, values1, values2):
        self.do_test(values1, values2, lambda x,y: x >= y)

    @given(hyp_collection, hyp_collection)
    def test_lt(self, values1, values2):
        self.do_test(values1, values2, lambda x,y: x < y)

    @given(hyp_collection, hyp_collection)
    def test_gt(self, values1, values2):
        self.do_test(values1, values2, lambda x,y: x > y)

    @given(hyp_collection, hyp_collection)
    def test_intersect(self, values1, values2):
        bm1 = BitMap(values1)
        bm2 = BitMap(values2)
        self.assertEqual(bm1.intersect(bm2), len(bm1&bm2) > 0)

class CardinalityTest(Util):

    @given(hyp_collection, hyp_collection)
    def setUp(self, values1, values2):
        self.bitmap1 = BitMap(values1)
        self.bitmap2 = BitMap(values2)

    def do_test_cardinality(self, real_op, estimated_op):
        real_value = real_op(self.bitmap1, self.bitmap2)
        estimated_value = estimated_op(self.bitmap1, self.bitmap2)
        self.assertEqual(real_value, estimated_value)

    def test_or_card(self):
        self.do_test_cardinality(lambda x,y : len(x|y), lambda x,y: x.union_cardinality(y))

    def test_and_card(self):
        self.do_test_cardinality(lambda x,y : len(x&y), lambda x,y: x.intersection_cardinality(y))

    def test_andnot_card(self):
        self.do_test_cardinality(lambda x,y : len(x-y), lambda x,y: x.difference_cardinality(y))

    def test_xor_card(self):
        self.do_test_cardinality(lambda x,y : len(x^y), lambda x,y: x.symmetric_difference_cardinality(y))

    def test_jaccard_index(self):
        real_value = float(len(self.bitmap1&self.bitmap2))/float(max(1, len(self.bitmap1|self.bitmap2)))
        estimated_value = self.bitmap1.jaccard_index(self.bitmap2)
        self.assertAlmostEqual(real_value, estimated_value)

class ManyOperationsTest(Util):

    @given(hyp_many_collections)
    def test_union(self, all_values):
        bitmaps = [BitMap(values) for values in all_values]
        bitmaps_copy = [BitMap(bm) for bm in bitmaps]
        result = BitMap.union(*bitmaps)
        self.assertEqual(bitmaps_copy, bitmaps)
        expected_result = functools.reduce(lambda x, y: x|y, bitmaps)
        self.assertEqual(expected_result, result)

class SerializationTest(Util):

    @given(hyp_collection, uint32)
    def test_serialization(self, values, other_value):
        st.assume(other_value not in values)
        old_bm = BitMap(values)
        buff = old_bm.serialize()
        new_bm = BitMap.deserialize(buff)
        self.assertEqual(old_bm, new_bm)
        old_bm.add(other_value)
        self.assertNotEqual(old_bm, new_bm)

    @given(hyp_collection)
    def test_load_dump(self, values):
        filepath = 'testfile'
        old_bm = BitMap(values)
        with open(filepath, 'wb') as f:
            dump(f, old_bm)
        with open(filepath, 'rb') as f:
            new_bm = load(f)
        self.assertEqual(old_bm, new_bm)
        os.remove(filepath)

    @given(hyp_collection)
    def test_pickle_protocol(self, values):
        old_bm = BitMap(values)
        pickled = pickle.dumps(old_bm)
        new_bm = pickle.loads(pickled)
        self.assertEqual(old_bm, new_bm)
        self.assertTrue(old_bm is not new_bm)
        self.assertNotEqual(old_bm.__obj__, new_bm.__obj__)

class StatisticsTest(Util):

    @given(hyp_collection)
    def test_basic_properties(self, values):
        bitmap = BitMap(values)
        stats = bitmap.get_statistics()
        self.assertEqual(stats.n_values_array_containers + stats.n_values_bitset_containers
            + stats.n_values_run_containers, len(bitmap))
        self.assertEqual(stats.n_bytes_array_containers, 2*stats.n_values_array_containers)
        self.assertEqual(stats.n_bytes_bitset_containers, 2**13*stats.n_bitset_containers)
        if len(values) > 0:
            self.assertEqual(stats.min_value, bitmap[0])
            self.assertEqual(stats.max_value, bitmap[len(bitmap)-1])
        self.assertEqual(stats.cardinality, len(bitmap))
        self.assertEqual(stats.sum_value, sum(values))

    def test_implementation_properties_array(self):
        values = range(2**16-10, 2**16+10, 2)
        stats = BitMap(values).get_statistics()
        self.assertEqual(stats.n_array_containers, 2)
        self.assertEqual(stats.n_bitset_containers, 0)
        self.assertEqual(stats.n_run_containers, 0)
        self.assertEqual(stats.n_values_array_containers, len(values))
        self.assertEqual(stats.n_values_bitset_containers, 0)
        self.assertEqual(stats.n_values_run_containers, 0)

    def test_implementation_properties_bitset(self):
        values = range(2**0, 2**17, 2)
        stats = BitMap(values).get_statistics()
        self.assertEqual(stats.n_array_containers, 0)
        self.assertEqual(stats.n_bitset_containers, 2)
        self.assertEqual(stats.n_run_containers, 0)
        self.assertEqual(stats.n_values_array_containers, 0)
        self.assertEqual(stats.n_values_bitset_containers, len(values))
        self.assertEqual(stats.n_values_run_containers, 0)

    def test_implementation_properties_run(self):
        values = range(2**0, 2**17, 1)
        stats = BitMap(values).get_statistics()
        self.assertEqual(stats.n_array_containers, 0)
        self.assertEqual(stats.n_bitset_containers, 0)
        self.assertEqual(stats.n_run_containers, 2)
        self.assertEqual(stats.n_values_array_containers, 0)
        self.assertEqual(stats.n_values_bitset_containers, 0)
        self.assertEqual(stats.n_values_run_containers, len(values))
        self.assertEqual(stats.n_bytes_run_containers, 12)

class FlipTest(Util):

    def check_flip(self, bm_before, bm_after, start, end):
        size = 100
        iter_range = random.sample(range(start, end), min(size, len(range(start, end))))
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

    @given(hyp_collection, integer, integer)
    def test_flip_empty(self, values, start, end):
        st.assume(start >= end)
        bm_before = BitMap(values)
        bm_copy = BitMap(bm_before)
        bm_after = bm_before.flip(start, end)
        self.assertEqual(bm_before, bm_copy)
        self.assertEqual(bm_before, bm_after)

    @given(hyp_collection, integer, integer)
    def test_flip(self, values, start, end):
        st.assume(start < end)
        bm_before = BitMap(values)
        bm_copy = BitMap(bm_before)
        bm_after = bm_before.flip(start, end)
        self.assertEqual(bm_before, bm_copy)
        self.check_flip(bm_before, bm_after, start, end)

    @given(hyp_collection, integer, integer)
    def test_flip_inplace_empty(self, values, start, end):
        st.assume(start >= end)
        bm_before = BitMap(values)
        bm_after = BitMap(bm_before)
        bm_after.flip_inplace(start, end)
        self.assertEqual(bm_before, bm_after)

    @given(hyp_collection, integer, integer)
    def test_flip_inplace(self, values, start, end):
        st.assume(start < end)
        bm_before = BitMap(values)
        bm_after = BitMap(bm_before)
        bm_after.flip_inplace(start, end)
        self.check_flip(bm_before, bm_after, start, end)

if __name__ == "__main__":
    unittest.main()
