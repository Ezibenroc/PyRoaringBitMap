#! /usr/bin/env python3

import pytest
import unittest
import random
import functools
import operator
import os
import sys
import pickle
import re
from hypothesis import given, settings, Verbosity, errors, HealthCheck, assume
import hypothesis.strategies as st
import array
import time
from pyroaring import BitMap64, FrozenBitMap64

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
uint32 = st.integers(min_value=0, max_value=2**32-1)
uint64 = st.integers(min_value=0, max_value=2**64-1)
large_uint64 = st.integers(min_value=2**32, max_value=2**64-1)
integer = st.integers(min_value=0, max_value=2**31-1)
int64 = st.integers(min_value=-2**63, max_value=2**63-1)

range_max_size = 2**18

range_big_step = uint18.flatmap(lambda n:
                                st.builds(range, st.just(n),
                                          st.integers(
                                              min_value=n+1, max_value=n+range_max_size),
                                          st.integers(min_value=2**8, max_value=range_max_size//8)))

range_small_step = uint18.flatmap(lambda n:
                                  st.builds(range, st.just(n),
                                            st.integers(
                                                min_value=n+1, max_value=n+range_max_size),
                                            st.integers(min_value=1, max_value=2**8)))

range_power2_step = uint18.flatmap(lambda n:
                                   st.builds(range, st.just(n),
                                             st.integers(
                                                 min_value=n+1, max_value=n+range_max_size),
                                             st.integers(min_value=0, max_value=8).flatmap(
                                       lambda n: st.just(2**n)
                                   )))

range_huge_interval = uint18.flatmap(lambda n:
                                st.builds(range, st.just(n),
                                          st.integers(
                                              min_value=n+2**52, max_value=n+2**63),
                                          st.integers(min_value=2**49, max_value=2**63)))

# Build a list of values of the form a * 2**16 + b with b in [-2,+2]
# In other words, numbers that are close (or equal) to a multiple of 2**16
multiple_2p16 = st.sets(
        st.builds(
            int.__add__, st.builds(
                int.__mul__,
                st.integers(min_value=1, max_value=2**32),
                st.just(2**16)
                ),
            st.integers(min_value=-2, max_value=+2)
            ),
        max_size=100)

hyp_range = (
        range_big_step |
        range_small_step |
        range_power2_step |
        range_huge_interval |
        multiple_2p16 |
        st.sampled_from([range(0, 0)])  # last one is an empty range
)
# would be great to build a true random set, but it takes too long and hypothesis does a timeout...
hyp_set = st.builds(set, hyp_range)
hyp_array = st.builds(lambda x: array.array('Q', x), hyp_range)
# Take the union of several strategies
hyp_collection = st.recursive(hyp_set, lambda values: st.builds(set.union, values, hyp_set))
hyp_many_collections = st.lists(hyp_collection, min_size=1, max_size=20)

bitmap_cls = st.sampled_from([BitMap64, FrozenBitMap64])


class Util:

    comparison_set = random.sample(
        range(2**8), 100) + random.sample(range(2**62), 50)

    def compare_with_set(self, bitmap, expected_set):
        assert len(bitmap) == len(expected_set)
        assert bool(bitmap) == bool(expected_set)
        assert set(bitmap) == expected_set
        assert sorted(list(bitmap)) == sorted(list(expected_set))
        assert BitMap64(expected_set) == bitmap
        for value in self.comparison_set:
            if value in expected_set:
                assert value in bitmap
            else:
                assert value not in bitmap

    @staticmethod
    def bitmap_sample(bitmap, size):
        indices = random.sample(range(len(bitmap)), size)
        return [bitmap[i] for i in indices]

    def assert_is_not(self, bitmap1, bitmap2):
        if isinstance(bitmap1, BitMap64):
            if bitmap1:
                bitmap1.remove(bitmap1[0])
            else:
                bitmap1.add(27)
        elif isinstance(bitmap2, BitMap64):
            if bitmap2:
                bitmap2.remove(bitmap1[0])
            else:
                bitmap2.add(27)
        else:  # The two are non-mutable, cannot do anything...
            return
        if bitmap1 == bitmap2:
            pytest.fail(
                'The two bitmaps are identical (modifying one also modifies the other).')


class TestBasic(Util):

    @given(hyp_collection)
    @settings(deadline=None)
    def test_basic(self, values):
        bitmap = BitMap64()
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
            with pytest.raises(KeyError):
                bitmap.add_checked(value)
            expected_set.add(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[:size//2]:
            bitmap.remove(value)
            expected_set.remove(value)
            with pytest.raises(KeyError):
                bitmap.remove(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[size//2:]:
            bitmap.discard(value)
            # check that we can discard element not in the bitmap
            bitmap.discard(value)
            expected_set.discard(value)
        self.compare_with_set(bitmap, expected_set)

    @given(bitmap_cls, bitmap_cls, hyp_collection)
    def test_bitmap_equality(self, cls1, cls2, values):
        bitmap1 = cls1(values)
        bitmap2 = cls2(values)
        assert bitmap1 == bitmap2

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection)
    def test_bitmap_unequality(self, cls1, cls2, values1, values2):
        assume(set(values1) != set(values2))
        bitmap1 = cls1(values1)
        bitmap2 = cls2(values2)
        assert bitmap1 != bitmap2

    @given(bitmap_cls, hyp_range | hyp_set | hyp_array | hyp_collection)
    def test_constructor_values(self, cls, values):
        bitmap = cls(values)
        expected_set = set(values)
        self.compare_with_set(bitmap, expected_set)

    @given(bitmap_cls, bitmap_cls, hyp_collection)
    def test_constructor_copy(self, cls1, cls2, values):
        bitmap1 = cls1(values)
        bitmap2 = cls2(bitmap1)
        assert bitmap1 == bitmap2
        self.assert_is_not(bitmap1, bitmap2)

    @given(hyp_collection, hyp_collection)
    def test_update(self, initial_values, new_values):
        bm = BitMap64(initial_values)
        expected = BitMap64(bm)
        bm.update(new_values)
        expected |= BitMap64(new_values)
        assert bm == expected

    @given(hyp_collection, hyp_collection)
    def test_intersection_update(self, initial_values, new_values):
        bm = BitMap64(initial_values)
        expected = BitMap64(bm)
        bm.intersection_update(new_values)
        expected &= BitMap64(new_values)
        assert bm == expected

    def wrong_op(self, op):
        bitmap = BitMap64()
        with pytest.raises(OverflowError):
            op(bitmap, -3)
        with pytest.raises(OverflowError):
            op(bitmap, 2**65)
        with pytest.raises(TypeError):
            op(bitmap, 'bla')

    def test_wrong_add(self):
        self.wrong_op(lambda bitmap, value: bitmap.add(value))

    def test_wrong_contain(self):
        self.wrong_op(lambda bitmap, value: bitmap.__contains__(value))

    @given(bitmap_cls)
    def test_wrong_constructor_values(self, cls):
        with pytest.raises(TypeError):  # this should fire a type error!
            bitmap = cls([3, 'bla', 3, 42])

    @given(bitmap_cls, hyp_collection)
    def test_to_array(self, cls, values):
        bitmap = cls(values)
        result = bitmap.to_array()
        expected = array.array('Q', sorted(values))
        assert result == expected

    @given(bitmap_cls, st.integers(min_value=0, max_value=100))
    def test_constructor_generator(self, cls, size):
        def generator(n):
            for i in range(n):
                yield i
        bitmap = cls(generator(size))
        assert bitmap == cls(range(size))


class TestSelectRank(Util):

    @given(bitmap_cls, hyp_collection)
    def test_simple_select(self, cls, values):
        bitmap = cls(values)
        values = list(bitmap)  # enforce sorted order
        for i in range(-len(values), len(values)):
            assert bitmap[i] == values[i]

    @given(bitmap_cls, hyp_collection, uint32)
    def test_wrong_selection(self, cls, values, n):
        bitmap = cls(values)
        with pytest.raises(IndexError):
            bitmap[len(values)]
        with pytest.raises(IndexError):
            bitmap[n + len(values)]
        with pytest.raises(IndexError):
            bitmap[-len(values)-1]
        with pytest.raises(IndexError):
            bitmap[-n - len(values) - 1]

    def check_slice(self, cls, values, start, stop, step):
        bitmap = cls(values)
        values = list(bitmap)  # enforce sorted order
        expected = values[start:stop:step]
        expected.sort()
        observed = list(bitmap[start:stop:step])
        assert expected == observed

    def slice_arg(n):
        return st.integers(min_value=-n, max_value=n)

    @given(bitmap_cls, hyp_collection, slice_arg(2**12), slice_arg(2**12), slice_arg(2**5))
    def test_slice_select_non_empty(self, cls, values, start, stop, step):
        assume(step != 0)
        assume(len(range(start, stop, step)) > 0)
        self.check_slice(cls, values, start, stop, step)

    @given(bitmap_cls, hyp_collection, slice_arg(2**12), slice_arg(2**12), slice_arg(2**5))
    def test_slice_select_empty(self, cls, values, start, stop, step):
        assume(step != 0)
        assume(len(range(start, stop, step)) == 0)
        self.check_slice(cls, values, start, stop, step)

    @given(bitmap_cls, hyp_collection, slice_arg(2**12) | st.none(), slice_arg(2**12) | st.none(), slice_arg(2**5) | st.none())
    def test_slice_select_none(self, cls, values, start, stop, step):
        assume(step != 0)
        self.check_slice(cls, values, start, stop, step)

    @given(bitmap_cls, hyp_collection)
    def test_simple_rank(self, cls, values):
        bitmap = cls(values)
        for i, value in enumerate(sorted(values)):
            assert bitmap.rank(value) == i+1

    @given(bitmap_cls, hyp_collection, uint18)
    def test_general_rank(self, cls, values, element):
        bitmap = cls(values)
        observed_rank = bitmap.rank(element)
        expected_rank = len([n for n in set(values) if n <= element])
        assert expected_rank == observed_rank

    @given(bitmap_cls, hyp_collection)
    def test_min(self, cls, values):
        assume(len(values) > 0)
        bitmap = cls(values)
        assert bitmap.min() == min(values)

    @given(bitmap_cls)
    def test_wrong_min(self, cls):
        bitmap = cls()
        with pytest.raises(ValueError):
            m = bitmap.min()

    @given(bitmap_cls, hyp_collection)
    def test_max(self, cls, values):
        assume(len(values) > 0)
        bitmap = cls(values)
        assert bitmap.max() == max(values)

    @given(bitmap_cls)
    def test_wrong_max(self, cls):
        bitmap = cls()
        with pytest.raises(ValueError):
            m = bitmap.max()

    @given(bitmap_cls, hyp_collection, uint32)
    def test_next_set_bit(self, cls, values, other_value):
        assume(len(values) > 0)
        bitmap = cls(values)
        try:
            expected = next(i for i in sorted(values) if i >= other_value)
            assert bitmap.next_set_bit(other_value) == expected
        except StopIteration:
            with pytest.raises(ValueError):
                b = bitmap.next_set_bit(other_value)

    @given(bitmap_cls)
    def test_wrong_next_set_bit(self, cls):
        bitmap = cls()
        with pytest.raises(ValueError):
            b = bitmap.next_set_bit(0)


class TestBinaryOperations(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection)
    def test_binary_op(self, cls1, cls2,  values1, values2):
        for op in [operator.or_, operator.and_, operator.xor, operator.sub]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = cls1(values1)
            self.bitmap2 = cls2(values2)
            old_bitmap1 = cls1(self.bitmap1)
            old_bitmap2 = cls2(self.bitmap2)
            result_set = op(self.set1, self.set2)
            result_bitmap = op(self.bitmap1, self.bitmap2)
            assert self.bitmap1 == old_bitmap1
            assert self.bitmap2 == old_bitmap2
            self.compare_with_set(result_bitmap, result_set)
            assert type(self.bitmap1) == type(result_bitmap)

    @given(bitmap_cls, hyp_collection, hyp_collection)
    def test_binary_op_inplace(self, cls2, values1, values2):
        for op in [operator.ior, operator.iand, operator.ixor, operator.isub]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = BitMap64(values1)
            original = self.bitmap1
            self.bitmap2 = cls2(values2)
            old_bitmap2 = cls2(self.bitmap2)
            op(self.set1, self.set2)
            op(self.bitmap1, self.bitmap2)
            assert original is self.bitmap1
            assert self.bitmap2 == old_bitmap2
            self.compare_with_set(self.bitmap1, self.set1)

    @given(hyp_collection)
    def test_binary_op_inplace_self(self, values) -> None:
        for op in [operator.ior, operator.iand, operator.ixor, operator.isub]:
            self.set = set(values)
            self.bitmap = BitMap64(values)
            original = self.bitmap
            op(self.set, self.set)
            op(self.bitmap, self.bitmap)
            assert original is self.bitmap
            self.compare_with_set(self.bitmap, self.set)

    @given(bitmap_cls, hyp_collection, hyp_collection)
    def test_binary_op_inplace_frozen(self, cls2, values1, values2):
        for op in [operator.ior, operator.iand, operator.ixor, operator.isub]:
            self.set1 = frozenset(values1)
            self.set2 = frozenset(values2)

            self.bitmap1 = FrozenBitMap64(values1)
            old_bitmap1 = FrozenBitMap64(self.bitmap1)
            self.bitmap2 = cls2(values2)
            old_bitmap2 = cls2(self.bitmap2)

            new_set = op(self.set1, self.set2)
            new_bitmap = op(self.bitmap1, self.bitmap2)

            assert self.bitmap1 == old_bitmap1
            assert self.bitmap2 == old_bitmap2

            self.compare_with_set(new_bitmap, new_set)


class TestComparison(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection)
    def test_comparison(self, cls1, cls2, values1, values2):
        for op in [operator.le, operator.ge, operator.lt, operator.gt]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = cls1(values1)
            self.bitmap2 = cls2(values2)
            assert op(self.bitmap1, self.bitmap1) == \
                             op(self.set1, self.set1)
            assert op(self.bitmap1, self.bitmap2) == \
                             op(self.set1, self.set2)
            assert op(self.bitmap1 | self.bitmap2, self.bitmap2) == \
                             op(self.set1 | self.set2, self.set2)
            assert op(self.set1, self.set1 | self.set2) == \
                             op(self.set1, self.set1 | self.set2)

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection)
    def test_intersect(self, cls1, cls2, values1, values2):
        bm1 = cls1(values1)
        bm2 = cls2(values2)
        assert (bm1.intersect(bm2)) == (len(bm1 & bm2) > 0)


class TestRange(Util):
    @given(bitmap_cls, hyp_collection, uint32, uint32)
    def test_contains_range_arbitrary(self, cls, values, start, end):
        bm = cls(values)
        expected = (cls(range(start, end)) <= bm)
        assert expected == bm.contains_range(start, end)

    @given(bitmap_cls, uint32, uint32)
    def test_contains_range(self, cls, start, end):
        assume(start < end)
        assert cls(range(start, end)).contains_range(start, end)
        assert cls(range(start, end)).contains_range(start, end-1)
        assert not cls(range(start, end-1)).contains_range(start, end)
        assert cls(range(start, end)).contains_range(start+1, end)
        assert not cls(range(start+1, end)).contains_range(start, end)
        r = range(start, end)
        try:
            middle = r[len(r)//2]  # on 32bits systems, this call might fail when len(r) is too large
        except OverflowError:
            if sys.maxsize > 2**32:
                raise
            else:
                return
        bm = cls(range(start, end)) - cls([middle])
        assert not bm.contains_range(start, end)
        assert bm.contains_range(start, middle)
        assert bm.contains_range(middle+1, end)

    @given(hyp_collection, uint32, uint32)
    def test_add_remove_range(self, values, start, end):
        assume(start < end)
        bm = BitMap64(values)
        # Empty range
        original = BitMap64(bm)
        bm.add_range(end, start)
        assert bm == original
        bm.remove_range(end, start)
        assert bm == original
        # Adding the range
        bm.add_range(start, end)
        assert bm.contains_range(start, end)
        assert bm.intersection_cardinality(BitMap64(range(start, end))) == end-start
        # Empty range (again)
        original = BitMap64(bm)
        bm.remove_range(end, start)
        assert bm == original
        assert bm.intersection_cardinality(BitMap64(range(start, end))) == end-start
        # Removing the range
        bm.remove_range(start, end)
        assert not bm.contains_range(start, end)
        assert bm.intersection_cardinality(BitMap64(range(start, end))) == 0


class TestCardinality(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection)
    def test_cardinality(self, cls1, cls2, values1, values2):

        for real_op, estimated_op in [
            (operator.or_, cls1.union_cardinality),
            (operator.and_, cls1.intersection_cardinality),
            (operator.sub, cls1.difference_cardinality),
            (operator.xor, cls1.symmetric_difference_cardinality)
        ]:
            self.bitmap1 = cls1(values1)
            self.bitmap2 = cls2(values2)
            real_value = len(real_op(self.bitmap1, self.bitmap2))
            estimated_value = estimated_op(self.bitmap1, self.bitmap2)
            assert real_value == estimated_value

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection)
    def test_jaccard_index(self, cls1, cls2, values1, values2):
        assume(len(values1) > 0 or len(values2) > 0)
        self.bitmap1 = cls1(values1)
        self.bitmap2 = cls2(values2)
        real_value = float(len(self.bitmap1 & self.bitmap2)) / \
            float(max(1, len(self.bitmap1 | self.bitmap2)))
        estimated_value = self.bitmap1.jaccard_index(self.bitmap2)
        assert real_value == pytest.approx(estimated_value)

    @given(bitmap_cls, hyp_collection, uint32, uint32)
    def test_range_cardinality(self, cls, values, a, b):
        bm = cls(values)
        start, end = sorted([a, b])

        #make an intersection with the relevant range to test against
        test_bm = bm.intersection(BitMap64(range(start, end)))

        assert len(test_bm) == bm.range_cardinality(start, end)


class TestManyOperations(Util):

    @given(hyp_collection, hyp_many_collections)
    def test_update(self, initial_values, all_values):
        self.initial_bitmap = BitMap64(initial_values)
        self.all_bitmaps = [BitMap64(values)
                            for values in all_values]
        self.initial_bitmap.update(*all_values)
        expected_result = functools.reduce(
            lambda x, y: x | y, self.all_bitmaps+[self.initial_bitmap])
        assert expected_result == self.initial_bitmap
        assert type(expected_result) == type(self.initial_bitmap)

    @given(hyp_collection, hyp_many_collections)
    def test_intersection_update(self, initial_values, all_values):
        self.initial_bitmap = BitMap64(initial_values)
        self.all_bitmaps = [BitMap64(values)
                            for values in all_values]
        self.initial_bitmap.intersection_update(*all_values)
        expected_result = functools.reduce(
            lambda x, y: x & y, self.all_bitmaps+[self.initial_bitmap])
        assert expected_result == self.initial_bitmap
        assert type(expected_result) == type(self.initial_bitmap)

    @given(bitmap_cls, st.data(), hyp_many_collections)
    def test_union(self, cls, data, all_values):
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values)
                            for i, values in enumerate(all_values)]
        result = cls.union(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x | y, self.all_bitmaps)
        assert expected_result == result

    @given(bitmap_cls, st.data(), hyp_many_collections)
    def test_intersection(self, cls, data, all_values):
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values)
                            for i, values in enumerate(all_values)]
        result = cls.intersection(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x & y, self.all_bitmaps)
        assert expected_result == result

    @given(bitmap_cls, st.data(), hyp_many_collections)
    def test_difference(self, cls, data, all_values):
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values)
                            for i, values in enumerate(all_values)]
        result = cls.difference(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x - y, self.all_bitmaps)
        assert expected_result == result


class TestSerialization(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection)
    def test_serialization(self, cls1, cls2, values):
        old_bm = cls1(values)
        buff = old_bm.serialize()
        new_bm = cls2.deserialize(buff)
        assert old_bm == new_bm
        assert isinstance(new_bm, cls2)
        self.assert_is_not(old_bm, new_bm)

    @given(bitmap_cls, hyp_collection, st.integers(min_value=2, max_value=pickle.HIGHEST_PROTOCOL))
    def test_pickle_protocol(self, cls, values, protocol):
        old_bm = cls(values)
        pickled = pickle.dumps(old_bm, protocol=protocol)
        new_bm = pickle.loads(pickled)
        assert old_bm == new_bm
        self.assert_is_not(old_bm, new_bm)


class TestFlip(Util):

    def check_flip(self, bm_before, bm_after, start, end):
        size = 100
        iter_range = random.sample(
            range(start, end), min(size, len(range(start, end))))
        iter_before = self.bitmap_sample(bm_before, min(size, len(bm_before)))
        iter_after = self.bitmap_sample(bm_after, min(size, len(bm_after)))
        for elt in iter_range:
            if elt in bm_before:
                assert elt not in bm_after
            else:
                assert elt in bm_after
        for elt in iter_before:
            if not (start <= elt < end):
                assert elt in bm_after
        for elt in iter_after:
            if not (start <= elt < end):
                assert elt in bm_before

    @given(bitmap_cls, hyp_collection, integer, integer)
    def test_flip_empty(self, cls, values, start, end):
        assume(start >= end)
        bm_before = cls(values)
        bm_copy = cls(bm_before)
        bm_after = bm_before.flip(start, end)
        assert bm_before == bm_copy
        assert bm_before == bm_after

    @given(bitmap_cls, hyp_collection, integer, integer)
    def test_flip(self, cls, values, start, end):
        assume(start < end)
        bm_before = cls(values)
        bm_copy = cls(bm_before)
        bm_after = bm_before.flip(start, end)
        assert bm_before == bm_copy
        self.check_flip(bm_before, bm_after, start, end)

    @given(hyp_collection, integer, integer)
    def test_flip_inplace_empty(self, values, start, end):
        assume(start >= end)
        bm_before = BitMap64(values)
        bm_after = BitMap64(bm_before)
        bm_after.flip_inplace(start, end)
        assert bm_before == bm_after

    @given(hyp_collection, integer, integer)
    def test_flip_inplace(self, values, start, end):
        assume(start < end)
        bm_before = BitMap64(values)
        bm_after = BitMap64(bm_before)
        bm_after.flip_inplace(start, end)
        self.check_flip(bm_before, bm_after, start, end)

@pytest.mark.skip("not supported yet")
class TestShift(Util):
    @given(bitmap_cls, hyp_collection, int64)
    def test_shift(self, cls, values, offset):
        bm_before = cls(values)
        bm_copy = cls(bm_before)
        bm_after = bm_before.shift(offset)
        assert bm_before == bm_copy
        expected = cls([val+offset for val in values if val+offset in range(0, 2**32)])
        assert bm_after == expected


class TestBitMap:
    @given(hyp_collection, uint32)
    def test_iter_equal_or_larger(self, values, other_value):
        bm = BitMap64(values)
        bm_iter = bm.iter_equal_or_larger(other_value)
        expected = [i for i in values if i >= other_value]
        expected.sort()

        observed = list(bm_iter)
        assert expected == observed

    def test_unashability(self):
        bm = BitMap64()
        with pytest.raises(TypeError):
            hash(bm)


class TestFrozen:

    @given(hyp_collection, hyp_collection, integer)
    def test_immutability(self, values, other, number):
        frozen = FrozenBitMap64(values)
        copy = FrozenBitMap64(values)
        other = BitMap64(other)
        with pytest.raises(AttributeError):
            frozen.clear()
        with pytest.raises(AttributeError):
            frozen.pop()
        with pytest.raises(AttributeError):
            frozen.add(number)
        with pytest.raises(AttributeError):
            frozen.update(other)
        with pytest.raises(AttributeError):
            frozen.discard(number)
        with pytest.raises(AttributeError):
            frozen.remove(number)
        with pytest.raises(AttributeError):
            frozen.intersection_update(other)
        with pytest.raises(AttributeError):
            frozen.difference_update(other)
        with pytest.raises(AttributeError):
            frozen.symmetric_difference_update(other)
        with pytest.raises(AttributeError):
            frozen.update(number, number+10)
        with pytest.raises(AttributeError):
            frozen.overwrite(other)
        assert frozen == copy

    @given(hyp_collection, hyp_collection)
    def test_hash_uneq(self, values1, values2):
        """This test as a non null (but extremly low) probability to fail."""
        bitmap1 = FrozenBitMap64(values1)
        bitmap2 = FrozenBitMap64(values2)
        assume(bitmap1 != bitmap2)
        h1 = hash(bitmap1)
        h2 = hash(bitmap2)
        hd = hash(bitmap1 ^ bitmap2)
        hashes = [h1, h2, hd]
        nb_collisions = len(hashes) - len(set(hashes))
        assert 1 >= nb_collisions

    @given(hyp_collection)
    def test_hash_eq(self, values):
        bitmap1 = FrozenBitMap64(values)
        bitmap2 = FrozenBitMap64(values)
        bitmap3 = FrozenBitMap64(bitmap1)
        assert hash(bitmap1) == hash(bitmap2)
        assert hash(bitmap1) == hash(bitmap3)

    def test_hash_eq2(self):
        """It can happen that two bitmaps hold the same values but have a different data structure. They should still
        have a same hash.
        This test compares two bitmaps with the same values, one has a run container, the other has an array container."""
        n = 100
        bm1 = FrozenBitMap64(range(n))
        bm2 = BitMap64()
        for i in range(n):
            bm2.add(i)
        bm2 = FrozenBitMap64(bm2, optimize=False)
        assert bm1 == bm2
        assert hash(bm1) == hash(bm2)

small_integer = st.integers(min_value=0, max_value=200)
small_integer_list = st.lists(min_size=0, max_size=2000, elements=small_integer)
class TestPythonSetEquivalent:
    """
    The main goal of this class is to make sure the BitMap64 api is a superset of the python builtin set api.
    """

    @given(bitmap_cls, small_integer_list)
    def test_convert_to_set(self, BitMap64Class, list1):
        """
        Most of the tests depend on a working implementation for converting from BitMap64 to python set.
        This test sanity checks it.

        This test should be modified or removed if you want to run PythonSetEquivalentTest with integers drawn from
        a larger set than `small_integer`. It will become prohibitively time-consuming.
        """
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        b1 = BitMap64Class(list1)

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
            assert (i in s1) == (i in converted_set)


    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_difference(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.difference(s2) == set(b1.difference(b2))
        assert SetClass.difference(s1, s2) == set(BitMap64Class.difference(b1, b2))
        assert (s1 - s2) == set(b1 - b2)
        assert b1 - b2 == b1.difference(b2)

    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_symmetric_difference(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.symmetric_difference(s2) == set(b1.symmetric_difference(b2))
        assert SetClass.symmetric_difference(s1, s2) == set(BitMap64Class.symmetric_difference(b1, b2))

    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_union(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.union(s2) == set(b1.union(b2))
        assert SetClass.union(s1, s2) == set(BitMap64Class.union(b1, b2))
        assert (s1 | s2) == set(b1 | b2)
        assert b1 | b2 == b1.union(b2)

    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_issubset(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.issubset(s2) == b1.issubset(b2)
        assert SetClass.issubset(s1, s2) == BitMap64Class.issubset(b1, b2)


    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_le(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.__le__(s2) == b1.__le__(b2)
        assert SetClass.__le__(s1, s2) == BitMap64Class.__le__(b1, b2)
        assert (s1 <= s2) == (b1 <= b2)


    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_ge(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.__ge__(s2) == b1.__ge__(b2)
        assert SetClass.__ge__(s1, s2) == BitMap64Class.__ge__(b1, b2)
        assert (s1 >= s2) == (b1 >= b2)

    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_eq(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()
        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.__eq__(s2) == b1.__eq__(b2)
        assert SetClass.__eq__(s1, s2) == BitMap64Class.__eq__(b1, b2)
        assert (s1 == s2) == (b1 == b2)


    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_issuperset(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.issuperset(s2) == b1.issuperset(b2)
        assert SetClass.issuperset(s1, s2) == BitMap64Class.issuperset(b1, b2)

    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_isdisjoint(self, BitMap64Class, list1, list2):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.isdisjoint(s2) == b1.isdisjoint(b2)
        assert SetClass.isdisjoint(s1, s2) == BitMap64Class.isdisjoint(b1, b2)


    @given(small_integer_list)
    def test_clear(self, list1):
        b1 = BitMap64(list1)
        b1.clear()
        assert len(b1) == 0

    @given(small_integer_list)
    def test_pop(self, list1):
        b1 = BitMap64(list1)
        starting_length = len(b1)
        if starting_length >= 1:
            popped_element = b1.pop()
            assert len(b1) == starting_length - 1#length decreased by one
            assert not popped_element in b1#and element isn't in the BitMap64 anymore
        else:
            with pytest.raises(KeyError):
                b1.pop()

    @given(bitmap_cls, small_integer_list)
    def test_copy(self, BitMap64Class, list1):
        b1 = BitMap64Class(list1)
        b2 = b1.copy()
        assert b2 == b1


    @given(small_integer_list)
    def test_copy_writable(self, list1):
        b1 = BitMap64(list1)
        b2 = b1.copy()

        try:
            new_element = max(b1) + 1 #doesn't exist in the set
        except ValueError:
            new_element = 1

        b2.add(new_element)

        assert new_element in b2
        assert new_element not in b1

    @given(small_integer_list, small_integer_list)
    def test_difference_update(self, list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        s1.difference_update(s2)

        b1 = BitMap64(list1)
        b2 = BitMap64(list2)
        b1.difference_update(b2)

        assert s1 == set(b1)

    @given(small_integer_list, small_integer_list)
    def test_symmetric_difference_update(self, list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        s1.symmetric_difference_update(s2)

        b1 = BitMap64(list1)
        b2 = BitMap64(list2)
        b1.symmetric_difference_update(b2)

        assert s1 == set(b1)


    @given(bitmap_cls, small_integer_list, small_integer_list)
    def test_dunder(self, BitMap64Class, list1, list2):
        """
        Tests for &|^-
        """
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)

        assert s1.__and__(s2) == SetClass(b1.__and__(b2))
        assert s1.__or__(s2) == SetClass(b1.__or__(b2))
        assert s1.__xor__(s2) == SetClass(b1.__xor__(b2))
        assert s1.__sub__(s2) == SetClass(b1.__sub__(b2))

    @given(small_integer_list, small_integer)
    def test_add(self, list1, value):
        s1 = set(list1)
        b1 = BitMap64(list1)
        assert s1 == set(b1)

        s1.add(value)
        b1.add(value)
        assert s1 == set(b1)

    @given(small_integer_list, small_integer)
    def test_discard(self, list1, value):
        s1 = set(list1)
        b1 = BitMap64(list1)
        assert s1 == set(b1)

        s1.discard(value)
        b1.discard(value)
        assert s1 == set(b1)

    @given(small_integer_list, small_integer)
    def test_remove(self, list1, value):
        s1 = set(list1)
        b1 = BitMap64(list1)
        assert s1 == set(b1)

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

        assert s1 == set(b1)
        assert s1_raised == b1_raised#Either both raised exception or neither did

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list)
    def test_nary_union(self, BitMap64Class, list1, list2, list3):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)
        s3 = SetClass(list3)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)
        b3 = BitMap64Class(list3)

        assert SetClass.union(s1, s2, s3) == SetClass(BitMap64Class.union(b1, b2, b3))
        assert s1.union(s2, s3) == SetClass(b1.union(b2, b3))

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list)
    def test_nary_difference(self, BitMap64Class, list1, list2, list3):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)
        s3 = SetClass(list3)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)
        b3 = BitMap64Class(list3)

        assert SetClass.difference(s1, s2, s3) == SetClass(BitMap64Class.difference(b1, b2, b3))
        assert s1.difference(s2, s3) == SetClass(b1.difference(b2, b3))

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list)
    def test_nary_intersection(self, BitMap64Class, list1, list2, list3):
        if BitMap64Class == BitMap64:
            SetClass = set
        elif BitMap64Class == FrozenBitMap64:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)
        s3 = SetClass(list3)

        b1 = BitMap64Class(list1)
        b2 = BitMap64Class(list2)
        b3 = BitMap64Class(list3)

        assert SetClass.intersection(s1, s2, s3) == SetClass(BitMap64Class.intersection(b1, b2, b3))
        assert s1.intersection(s2, s3) == SetClass(b1.intersection(b2, b3))

    @given(small_integer_list, small_integer_list, small_integer_list)
    def test_nary_intersection_update(self, list1, list2, list3):
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap64(list1)
        b2 = BitMap64(list2)
        b3 = BitMap64(list3)

        set.intersection_update(s1, s2, s3)
        BitMap64.intersection_update(b1, b2, b3)
        assert s1 == set(b1)


        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap64(list1)
        b2 = BitMap64(list2)
        b3 = BitMap64(list3)

        s1.intersection_update(s2, s3)
        b1.intersection_update(b2, b3)

        assert s1 == set(b1)


    @given(small_integer_list, small_integer_list, small_integer_list)
    def test_nary_difference_update(self, list1, list2, list3):
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap64(list1)
        b2 = BitMap64(list2)
        b3 = BitMap64(list3)

        set.difference_update(s1, s2, s3)
        BitMap64.difference_update(b1, b2, b3)
        assert s1 == set(b1)


        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap64(list1)
        b2 = BitMap64(list2)
        b3 = BitMap64(list3)

        s1.difference_update(s2, s3)
        b1.difference_update(b2, b3)

        assert s1 == set(b1)

    @given(small_integer_list, small_integer_list, small_integer_list)
    def test_nary_update(self, list1, list2, list3):
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap64(list1)
        b2 = BitMap64(list2)
        b3 = BitMap64(list3)

        set.update(s1, s2, s3)
        BitMap64.update(b1, b2, b3)
        assert s1 == set(b1)


        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap64(list1)
        b2 = BitMap64(list2)
        b3 = BitMap64(list3)

        s1.update(s2, s3)
        b1.update(b2, b3)

        assert s1 == set(b1)

small_list_of_uin32 = st.lists(min_size=0, max_size=400, elements=uint32)
large_list_of_uin32 = st.lists(min_size=600, max_size=1000, elements=uint32, unique=True)
class TestString:

    @given(bitmap_cls, small_list_of_uin32)
    def test_small_list(self, cls, collection):
        #test that repr for a small bitmap is equal to the original bitmap
        bm = cls(collection)
        assert bm == eval(repr(bm))

    @settings(suppress_health_check=list(HealthCheck))
    @given(bitmap_cls, large_list_of_uin32)
    def test_large_list(self, cls, collection):
        # test that for a large bitmap the both the start and the end of the bitmap get printed
        bm = cls(collection)
        s = repr(bm).replace(cls.__name__, " ")
        nondigits = set(s) - set('0123456789\n.')
        for i in nondigits:
            s = s.replace(i, ' ')

        small, large = s.split('...')
        small_ints = [int(i) for i in small.split()]
        large_ints = [int(i) for i in large.split()]

        for i in small_ints:
            assert i in bm

        for i in large_ints:
            assert i in bm

        assert min(small_ints) == bm.min()
        assert max(large_ints) == bm.max()

if __name__ == "__main__":
    unittest.main()
