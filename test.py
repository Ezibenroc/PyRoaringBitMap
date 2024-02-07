#! /usr/bin/env python3

from __future__ import annotations

import pytest
import os
import re
import sys
import array
import pickle
import random
import operator
import unittest
import functools
from typing import TYPE_CHECKING
from collections.abc import Set, Callable, Iterable, Iterator

import hypothesis.strategies as st
from hypothesis import given, assume, errors, settings, Verbosity, HealthCheck

import pyroaring
from pyroaring import BitMap, FrozenBitMap, AbstractBitMap


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
hyp_set: st.SearchStrategy[set[int]] = st.builds(set, hyp_range)
hyp_array = st.builds(lambda x: array.array('I', x), hyp_range)
hyp_collection = hyp_range | hyp_set | hyp_array
hyp_many_collections = st.lists(hyp_collection, min_size=1, max_size=20)

bitmap_cls = st.sampled_from([BitMap, FrozenBitMap])

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    HypCollection: TypeAlias = range | set[int] | array.array[int] | list[int]
    EitherBitMap = BitMap | FrozenBitMap
    EitherSet = set | frozenset  # type: ignore[type-arg]


class Util:

    comparison_set = random.sample(
        range(2**8), 100) + random.sample(range(2**31 - 1), 50)

    def compare_with_set(self, bitmap: AbstractBitMap, expected_set: set[int]) -> None:
        assert len(bitmap) == len(expected_set)
        assert bool(bitmap) == bool(expected_set)
        assert set(bitmap) == expected_set
        assert sorted(list(bitmap)) == sorted(list(expected_set))
        assert BitMap(expected_set, copy_on_write=bitmap.copy_on_write) == bitmap
        for value in self.comparison_set:
            if value in expected_set:
                assert value in bitmap
            else:
                assert value not in bitmap

    @staticmethod
    def bitmap_sample(bitmap: AbstractBitMap, size: int) -> list[int]:
        indices = random.sample(range(len(bitmap)), size)
        return [bitmap[i] for i in indices]

    def assert_is_not(self, bitmap1: AbstractBitMap, bitmap2: AbstractBitMap) -> None:
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
            pytest.fail(
                'The two bitmaps are identical (modifying one also modifies the other).')


class BasicTest(Util):

    @given(hyp_collection, st.booleans())
    @settings(deadline=None)
    def test_basic(self, values: HypCollection, cow: bool) -> None:
        bitmap = BitMap(copy_on_write=cow)
        assert bitmap.copy_on_write == cow
        expected_set: set[int] = set()
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
            with pytest.raises(KeyError):
                bitmap.add_checked(value)
            expected_set.add(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[:size // 2]:
            bitmap.remove(value)
            expected_set.remove(value)
            with pytest.raises(KeyError):
                bitmap.remove(value)
        self.compare_with_set(bitmap, expected_set)
        for value in values[size // 2:]:
            bitmap.discard(value)
            # check that we can discard element not in the bitmap
            bitmap.discard(value)
            expected_set.discard(value)
        self.compare_with_set(bitmap, expected_set)

    @given(bitmap_cls, bitmap_cls, hyp_collection, st.booleans())
    def test_bitmap_equality(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
    ) -> None:
        bitmap1 = cls1(values, copy_on_write=cow)
        bitmap2 = cls2(values, copy_on_write=cow)
        assert bitmap1 == bitmap2

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_bitmap_unequality(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values1: HypCollection,
        values2: HypCollection,
        cow: bool,
    ) -> None:
        assume(set(values1) != set(values2))
        bitmap1 = cls1(values1, copy_on_write=cow)
        bitmap2 = cls2(values2, copy_on_write=cow)
        assert bitmap1 != bitmap2

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_constructor_values(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
    ) -> None:
        bitmap = cls(values, copy_on_write=cow)
        expected_set = set(values)
        self.compare_with_set(bitmap, expected_set)

    @given(bitmap_cls, bitmap_cls, hyp_collection, uint32, st.booleans(), st.booleans())
    def test_constructor_copy(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values: HypCollection,
        other_value: int,
        cow1: bool,
        cow2: bool,
    ) -> None:
        bitmap1 = cls1(values, copy_on_write=cow1)
        # should be robust even if cow2 != cow1
        bitmap2 = cls2(bitmap1, copy_on_write=cow2)
        assert bitmap1 == bitmap2
        self.assert_is_not(bitmap1, bitmap2)

    @given(hyp_collection, hyp_collection, st.booleans())
    def test_update(self, initial_values: HypCollection, new_values: HypCollection, cow: bool) -> None:
        bm = BitMap(initial_values, cow)
        expected = BitMap(bm)
        bm.update(new_values)
        expected |= BitMap(new_values, copy_on_write=cow)
        assert bm == expected

    @given(hyp_collection, hyp_collection, st.booleans())
    def test_intersection_update(self, initial_values: HypCollection, new_values: HypCollection, cow: bool) -> None:
        bm = BitMap(initial_values, cow)
        expected = BitMap(bm)
        bm.intersection_update(new_values)
        expected &= BitMap(new_values, copy_on_write=cow)
        assert bm == expected

    def wrong_op(self, op: Callable[[BitMap, int], object]) -> None:
        bitmap = BitMap()
        with pytest.raises(OverflowError):
            op(bitmap, -3)
        with pytest.raises(OverflowError):
            op(bitmap, 2**33)
        with pytest.raises(TypeError):
            op(bitmap, 'bla')  # type: ignore[arg-type]

    def test_wrong_add(self) -> None:
        self.wrong_op(lambda bitmap, value: bitmap.add(value))

    def test_wrong_contain(self) -> None:
        self.wrong_op(lambda bitmap, value: bitmap.__contains__(value))

    @given(bitmap_cls)
    def test_wrong_constructor_values(self, cls: type[EitherBitMap]) -> None:
        with pytest.raises(TypeError):  # this should fire a type error!
            cls([3, 'bla', 3, 42])  # type: ignore[list-item]
        bad_range = range(-3, 0)
        with pytest.raises(OverflowError):
            cls(bad_range)

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_to_array(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
    ) -> None:
        bitmap = cls(values, copy_on_write=cow)
        result = bitmap.to_array()
        expected = array.array('I', sorted(values))
        assert result == expected

    @given(bitmap_cls, st.booleans(), st.integers(min_value=0, max_value=100))
    def test_constructor_generator(self, cls: type[EitherBitMap], cow: bool, size: int) -> None:
        def generator(n: int) -> Iterator[int]:
            for i in range(n):
                yield i
        bitmap = cls(generator(size), copy_on_write=cow)
        assert bitmap == cls(range(size), copy_on_write=cow)


def slice_arg(n: int) -> st.SearchStrategy[int]:
    return st.integers(min_value=-n, max_value=n)


class SelectRankTest(Util):

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_simple_select(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
    ) -> None:
        bitmap = cls(values, copy_on_write=cow)
        values = list(bitmap)  # enforce sorted order
        for i in range(-len(values), len(values)):
            assert bitmap[i] == values[i]

    @given(bitmap_cls, hyp_collection, uint32, st.booleans())
    def test_wrong_selection(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        n: int,
        cow: bool,
    ) -> None:
        bitmap = cls(values, cow)
        with pytest.raises(IndexError):
            bitmap[len(values)]
        with pytest.raises(IndexError):
            bitmap[n + len(values)]
        with pytest.raises(IndexError):
            bitmap[-len(values) - 1]
        with pytest.raises(IndexError):
            bitmap[-n - len(values) - 1]

    def check_slice(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        start: int | None,
        stop: int | None,
        step: int | None,
        cow: bool,
    ) -> None:
        bitmap = cls(values, copy_on_write=cow)
        values = list(bitmap)  # enforce sorted order
        expected = values[start:stop:step]
        expected.sort()
        observed = list(bitmap[start:stop:step])
        assert expected == observed

    @given(bitmap_cls, hyp_collection, slice_arg(2**12), slice_arg(2**12), slice_arg(2**5), st.booleans())
    def test_slice_select_non_empty(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        start: int,
        stop: int,
        step: int,
        cow: bool,
    ) -> None:
        assume(step != 0)
        assume(len(range(start, stop, step)) > 0)
        self.check_slice(cls, values, start, stop, step, cow)

    @given(bitmap_cls, hyp_collection, slice_arg(2**12), slice_arg(2**12), slice_arg(2**5), st.booleans())
    def test_slice_select_empty(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        start: int,
        stop: int,
        step: int,
        cow: bool,
    ) -> None:
        assume(step != 0)
        assume(len(range(start, stop, step)) == 0)
        self.check_slice(cls, values, start, stop, step, cow)

    @given(bitmap_cls, hyp_collection, slice_arg(2**12) | st.none(), slice_arg(2**12) | st.none(), slice_arg(2**5) | st.none(), st.booleans())
    def test_slice_select_none(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        start: int | None,
        stop: int | None,
        step: int | None,
        cow: bool,
    ) -> None:
        assume(step != 0)
        self.check_slice(cls, values, start, stop, step, cow)

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_simple_rank(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
    ) -> None:
        bitmap = cls(values, copy_on_write=cow)
        for i, value in enumerate(sorted(values)):
            assert bitmap.rank(value) == i + 1

    @given(bitmap_cls, hyp_collection, uint18, st.booleans())
    def test_general_rank(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        element: int,
        cow: bool,
    ) -> None:
        bitmap = cls(values, copy_on_write=cow)
        observed_rank = bitmap.rank(element)
        expected_rank = len([n for n in set(values) if n <= element])
        assert expected_rank == observed_rank

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_min(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
    ) -> None:
        assume(len(values) > 0)
        bitmap = cls(values, copy_on_write=cow)
        assert bitmap.min() == min(values)

    @given(bitmap_cls)
    def test_wrong_min(self, cls: type[EitherBitMap]) -> None:
        bitmap = cls()
        with pytest.raises(ValueError):
            bitmap.min()

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_max(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
    ) -> None:
        assume(len(values) > 0)
        bitmap = cls(values, copy_on_write=cow)
        assert bitmap.max() == max(values)

    @given(bitmap_cls)
    def test_wrong_max(self, cls: type[EitherBitMap]) -> None:
        bitmap = cls()
        with pytest.raises(ValueError):
            bitmap.max()

    @given(bitmap_cls, hyp_collection, uint32, st.booleans())
    def test_next_set_bit(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        other_value: int,
        cow: bool,
    ) -> None:
        assume(len(values) > 0)
        bitmap = cls(values, copy_on_write=cow)
        try:
            expected = next(i for i in sorted(values) if i >= other_value)
            assert bitmap.next_set_bit(other_value) == expected
        except StopIteration:
            with pytest.raises(ValueError):
                bitmap.next_set_bit(other_value)

    @given(bitmap_cls)
    def test_wrong_next_set_bit(self, cls: type[EitherBitMap]) -> None:
        bitmap = cls()
        with pytest.raises(ValueError):
            bitmap.next_set_bit(0)


class BinaryOperationsTest(Util):
    set1: Set[int]
    set2: Set[int]

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_binary_op(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values1: HypCollection,
        values2: HypCollection,
        cow: bool,
    ) -> None:
        for op in [operator.or_, operator.and_, operator.xor, operator.sub]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = cls1(values1, cow)
            self.bitmap2 = cls2(values2, cow)
            old_bitmap1 = cls1(self.bitmap1)
            old_bitmap2 = cls2(self.bitmap2)
            result_set = op(self.set1, self.set2)
            result_bitmap = op(self.bitmap1, self.bitmap2)
            assert self.bitmap1 == old_bitmap1
            assert self.bitmap2 == old_bitmap2
            self.compare_with_set(result_bitmap, result_set)
            assert type(self.bitmap1) == type(result_bitmap)

    @given(bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_binary_op_inplace(
        self,
        cls2: type[EitherBitMap],
        values1: HypCollection,
        values2: HypCollection,
        cow: bool,
    ) -> None:
        for op in [operator.ior, operator.iand, operator.ixor, operator.isub]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = BitMap(values1, cow)
            original = self.bitmap1
            self.bitmap2 = cls2(values2, cow)
            old_bitmap2 = cls2(self.bitmap2)
            op(self.set1, self.set2)
            op(self.bitmap1, self.bitmap2)
            assert original is self.bitmap1
            assert self.bitmap2 == old_bitmap2
            self.compare_with_set(self.bitmap1, self.set1)

    @given(bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_binary_op_inplace_frozen(
        self,
        cls2: type[EitherBitMap],
        values1: HypCollection,
        values2: HypCollection,
        cow: bool,
    ) -> None:
        for op in [operator.ior, operator.iand, operator.ixor, operator.isub]:
            self.set1 = frozenset(values1)
            self.set2 = frozenset(values2)

            self.bitmap1 = FrozenBitMap(values1, cow)
            old_bitmap1 = FrozenBitMap(self.bitmap1)
            self.bitmap2 = cls2(values2, cow)
            old_bitmap2 = cls2(self.bitmap2)

            new_set = op(self.set1, self.set2)
            new_bitmap = op(self.bitmap1, self.bitmap2)

            assert self.bitmap1 == old_bitmap1
            assert self.bitmap2 == old_bitmap2

            self.compare_with_set(new_bitmap, new_set)


class ComparisonTest(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_comparison(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values1: HypCollection,
        values2: HypCollection,
        cow: bool,
    ) -> None:
        for op in [operator.le, operator.ge, operator.lt, operator.gt, operator.eq, operator.ne]:
            self.set1 = set(values1)
            self.set2 = set(values2)
            self.bitmap1 = cls1(values1, copy_on_write=cow)
            self.bitmap2 = cls2(values2, copy_on_write=cow)
            assert op(self.bitmap1, self.bitmap1) == \
                             op(self.set1, self.set1)
            assert op(self.bitmap1, self.bitmap2) == \
                             op(self.set1, self.set2)
            assert op(self.bitmap1 | self.bitmap2, self.bitmap2) == \
                             op(self.set1 | self.set2, self.set2)
            assert op(self.set1, self.set1 | self.set2) == \
                             op(self.set1, self.set1 | self.set2)

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_comparison_other_objects(self, cls: type[EitherBitMap], values: HypCollection, cow: bool) -> None:
        for op in [operator.le, operator.ge, operator.lt, operator.gt]:
            bm = cls(values, copy_on_write=cow)
            with pytest.raises(TypeError):
                op(bm, 42)
            with pytest.raises(TypeError):
                op(bm, None)

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_intersect(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values1: HypCollection,
        values2: HypCollection,
        cow: bool,
    ) -> None:
        bm1 = cls1(values1, copy_on_write=cow)
        bm2 = cls2(values2, copy_on_write=cow)
        assert bm1.intersect(bm2) == len(bm1 & bm2) > 0

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_eq_other_objects(self, cls: type[EitherBitMap], values: HypCollection, cow: bool) -> None:
        bm = cls(values, copy_on_write=cow)

        assert not bm == 42
        assert cls.__eq__(bm, 42) is NotImplemented
        assert not bm == None# noqa: E711
        assert cls.__eq__(bm, None) is NotImplemented

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_ne_other_objects(self, cls: type[EitherBitMap], values: HypCollection, cow: bool) -> None:
        bm = cls(values, copy_on_write=cow)

        assert bm != 42
        assert cls.__ne__(bm, 42) is NotImplemented
        assert bm != None# noqa: E711
        assert cls.__ne__(bm, None) is NotImplemented


class RangeTest(Util):
    @given(bitmap_cls, hyp_collection, st.booleans(), uint32, uint32)
    def test_contains_range_arbitrary(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
        start: int,
        end: int,
    ) -> None:
        bm = cls(values)
        expected = (cls(range(start, end)) <= bm)
        assert expected == bm.contains_range(start, end)

    @given(bitmap_cls, st.booleans(), uint32, uint32)
    def test_contains_range(self, cls: type[EitherBitMap], cow: bool, start: int, end: int) -> None:
        assume(start < end)
        assert cls(range(start, end)).contains_range(start, end)
        assert cls(range(start, end)).contains_range(start, end - 1)
        assert not cls(range(start, end - 1)).contains_range(start, end)
        assert cls(range(start, end)).contains_range(start + 1, end)
        assert not cls(range(start + 1, end)).contains_range(start, end)
        r = range(start, end)
        try:
            middle = r[len(r) // 2]  # on 32bits systems, this call might fail when len(r) is too large
        except OverflowError:
            if sys.maxsize > 2**32:
                raise
            else:
                return
        bm = cls(range(start, end)) - cls([middle])
        assert not bm.contains_range(start, end)
        assert bm.contains_range(start, middle)
        assert bm.contains_range(middle + 1, end)

    @given(hyp_collection, st.booleans(), uint32, uint32)
    def test_add_remove_range(self, values: HypCollection, cow: bool, start: int, end: int) -> None:
        assume(start < end)
        bm = BitMap(values, copy_on_write=cow)
        # Empty range
        original = BitMap(bm)
        bm.add_range(end, start)
        assert bm == original
        bm.remove_range(end, start)
        assert bm == original
        # Adding the range
        bm.add_range(start, end)
        assert bm.contains_range(start, end)
        assert bm.intersection_cardinality(BitMap(range(start, end), copy_on_write=cow)) == end - start
        # Empty range (again)
        original = BitMap(bm)
        bm.remove_range(end, start)
        assert bm == original
        assert bm.intersection_cardinality(BitMap(range(start, end), copy_on_write=cow)) == end - start
        # Removing the range
        bm.remove_range(start, end)
        assert not bm.contains_range(start, end)
        assert bm.intersection_cardinality(BitMap(range(start, end), copy_on_write=cow)) == 0

    @given(hyp_collection, st.booleans(), large_uint64, large_uint64)
    def test_large_values(self, values: HypCollection, cow: bool, start: int, end: int) -> None:
        bm = BitMap(values, copy_on_write=cow)
        original = BitMap(bm)
        bm.add_range(start, end)
        assert bm == original
        bm.remove_range(start, end)
        assert bm == original
        assert bm.contains_range(start, end)


class CardinalityTest(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_cardinality(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values1: HypCollection,
        values2: HypCollection,
        cow: bool,
    ) -> None:

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
            assert real_value == estimated_value

    @given(bitmap_cls, bitmap_cls, hyp_collection, hyp_collection, st.booleans())
    def test_jaccard_index(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values1: HypCollection,
        values2: HypCollection,
        cow: bool,
    ) -> None:
        assume(len(values1) > 0 or len(values2) > 0)
        self.bitmap1 = cls1(values1, copy_on_write=cow)
        self.bitmap2 = cls2(values2, copy_on_write=cow)
        real_value = float(len(self.bitmap1 & self.bitmap2)) / \
            float(max(1, len(self.bitmap1 | self.bitmap2)))
        estimated_value = self.bitmap1.jaccard_index(self.bitmap2)
        assert real_value == pytest.approx(estimated_value)

    @given(bitmap_cls, hyp_collection, uint32, uint32)
    def test_range_cardinality(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        a: int,
        b: int,
    ) -> None:
        bm = cls(values)
        start, end = sorted([a, b])

        # make an intersection with the relevant range to test against
        test_bm = bm.intersection(BitMap(range(start, end)))

        assert len(test_bm) == bm.range_cardinality(start, end)


class ManyOperationsTest(Util):
    all_bitmaps: Iterable[AbstractBitMap]

    @given(hyp_collection, hyp_many_collections, st.booleans())
    def test_update(
        self,
        initial_values: HypCollection,
        all_values: list[HypCollection],
        cow: bool,
    ) -> None:
        self.initial_bitmap = BitMap(initial_values, copy_on_write=cow)
        self.all_bitmaps = [BitMap(values, copy_on_write=cow)
                            for values in all_values]
        self.initial_bitmap.update(*all_values)
        expected_result = functools.reduce(
            lambda x, y: x | y, self.all_bitmaps + [self.initial_bitmap])
        assert expected_result == self.initial_bitmap
        assert type(expected_result) == type(self.initial_bitmap)

    @given(hyp_collection, hyp_many_collections, st.booleans())
    def test_intersection_update(
        self,
        initial_values: HypCollection,
        all_values: list[HypCollection],
        cow: bool,
    ) -> None:
        self.initial_bitmap = BitMap(initial_values, copy_on_write=cow)
        self.all_bitmaps = [BitMap(values, copy_on_write=cow)
                            for values in all_values]
        self.initial_bitmap.intersection_update(*all_values)
        expected_result = functools.reduce(
            lambda x, y: x & y, self.all_bitmaps + [self.initial_bitmap])
        assert expected_result == self.initial_bitmap
        assert type(expected_result) == type(self.initial_bitmap)

    @given(bitmap_cls, st.data(), hyp_many_collections, st.booleans())
    def test_union(
        self,
        cls: type[EitherBitMap],
        data: st.DataObject,
        all_values: list[HypCollection],
        cow: bool,
    ) -> None:
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values, copy_on_write=cow)
                            for i, values in enumerate(all_values)]
        result = cls.union(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x | y, self.all_bitmaps)
        assert expected_result == result

    @given(bitmap_cls, st.data(), hyp_many_collections, st.booleans())
    def test_intersection(
        self,
        cls: type[EitherBitMap],
        data: st.DataObject,
        all_values: list[HypCollection],
        cow: bool,
    ) -> None:
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values, copy_on_write=cow)
                            for i, values in enumerate(all_values)]
        result = cls.intersection(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x & y, self.all_bitmaps)
        assert expected_result == result

    @given(bitmap_cls, st.data(), hyp_many_collections, st.booleans())
    def test_difference(
        self,
        cls: type[EitherBitMap],
        data: st.DataObject,
        all_values: list[HypCollection],
        cow: bool,
    ) -> None:
        classes = [data.draw(bitmap_cls) for _ in range(len(all_values))]
        self.all_bitmaps = [classes[i](values, copy_on_write=cow)
                            for i, values in enumerate(all_values)]
        result = cls.difference(*self.all_bitmaps)
        expected_result = functools.reduce(
            lambda x, y: x - y, self.all_bitmaps)
        assert expected_result == result


class SerializationTest(Util):

    @given(bitmap_cls, bitmap_cls, hyp_collection)
    def test_serialization(
        self,
        cls1: type[EitherBitMap],
        cls2: type[EitherBitMap],
        values: HypCollection,
    ) -> None:
        old_bm = cls1(values)
        buff = old_bm.serialize()
        new_bm = cls2.deserialize(buff)
        assert old_bm == new_bm
        assert isinstance(new_bm, cls2)
        self.assert_is_not(old_bm, new_bm)

    @given(bitmap_cls, hyp_collection, st.integers(min_value=2, max_value=pickle.HIGHEST_PROTOCOL))
    def test_pickle_protocol(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        protocol: int,
    ) -> None:
        old_bm = cls(values)
        pickled = pickle.dumps(old_bm, protocol=protocol)
        new_bm = pickle.loads(pickled)
        assert old_bm == new_bm
        self.assert_is_not(old_bm, new_bm)


class StatisticsTest(Util):

    @given(bitmap_cls, hyp_collection, st.booleans())
    def test_basic_properties(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        cow: bool,
    ) -> None:
        bitmap = cls(values, copy_on_write=cow)
        stats = bitmap.get_statistics()
        assert stats['n_values_array_containers'] + stats['n_values_bitset_containers'] \
                         + stats['n_values_run_containers'] == len(bitmap)
        assert stats['n_bytes_array_containers'] == \
                         2 * stats['n_values_array_containers']
        assert stats['n_bytes_bitset_containers'] == \
                         2**13 * stats['n_bitset_containers']
        if len(values) > 0:
            assert stats['min_value'] == bitmap[0]
            assert stats['max_value'] == bitmap[len(bitmap) - 1]
        assert stats['cardinality'] == len(bitmap)
        assert stats['sum_value'] == sum(values)

    @given(bitmap_cls)
    def test_implementation_properties_array(self, cls: type[EitherBitMap]) -> None:
        values = range(2**16 - 10, 2**16 + 10, 2)
        stats = cls(values).get_statistics()
        assert stats['n_array_containers'] == 2
        assert stats['n_bitset_containers'] == 0
        assert stats['n_run_containers'] == 0
        assert stats['n_values_array_containers'] == len(values)
        assert stats['n_values_bitset_containers'] == 0
        assert stats['n_values_run_containers'] == 0

    @given(bitmap_cls)
    def test_implementation_properties_bitset(self, cls: type[EitherBitMap]) -> None:
        values = range(2**0, 2**17, 2)
        stats = cls(values).get_statistics()
        assert stats['n_array_containers'] == 0
        assert stats['n_bitset_containers'] == 2
        assert stats['n_run_containers'] == 0
        assert stats['n_values_array_containers'] == 0
        assert stats['n_values_bitset_containers'] == len(values)
        assert stats['n_values_run_containers'] == 0

    @given(bitmap_cls)
    def test_implementation_properties_run(self, cls: type[EitherBitMap]) -> None:
        values = range(2**0, 2**17, 1)
        stats = cls(values).get_statistics()
        assert stats['n_array_containers'] == 0
        assert stats['n_bitset_containers'] == 0
        assert stats['n_run_containers'] == 2
        assert stats['n_values_array_containers'] == 0
        assert stats['n_values_bitset_containers'] == 0
        assert stats['n_values_run_containers'] == len(values)
        assert stats['n_bytes_run_containers'] == 12


class FlipTest(Util):

    def check_flip(self, bm_before: AbstractBitMap, bm_after: AbstractBitMap, start: int, end: int) -> None:
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

    @given(bitmap_cls, hyp_collection, integer, integer, st.booleans())
    def test_flip_empty(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        start: int,
        end: int,
        cow: bool,
    ) -> None:
        assume(start >= end)
        bm_before = cls(values, copy_on_write=cow)
        bm_copy = cls(bm_before)
        bm_after = bm_before.flip(start, end)
        assert bm_before == bm_copy
        assert bm_before == bm_after

    @given(bitmap_cls, hyp_collection, integer, integer, st.booleans())
    def test_flip(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        start: int,
        end: int,
        cow: bool,
    ) -> None:
        assume(start < end)
        bm_before = cls(values, copy_on_write=cow)
        bm_copy = cls(bm_before)
        bm_after = bm_before.flip(start, end)
        assert bm_before == bm_copy
        self.check_flip(bm_before, bm_after, start, end)

    @given(hyp_collection, integer, integer, st.booleans())
    def test_flip_inplace_empty(
        self,
        values: HypCollection,
        start: int,
        end: int,
        cow: bool,
    ) -> None:
        assume(start >= end)
        bm_before = BitMap(values, copy_on_write=cow)
        bm_after = BitMap(bm_before)
        bm_after.flip_inplace(start, end)
        assert bm_before == bm_after

    @given(hyp_collection, integer, integer, st.booleans())
    def test_flip_inplace(
        self,
        values: HypCollection,
        start: int,
        end: int,
        cow: bool,
    ) -> None:
        assume(start < end)
        bm_before = BitMap(values, copy_on_write=cow)
        bm_after = BitMap(bm_before)
        bm_after.flip_inplace(start, end)
        self.check_flip(bm_before, bm_after, start, end)


class ShiftTest(Util):
    @given(bitmap_cls, hyp_collection, int64, st.booleans())
    def test_shift(
        self,
        cls: type[EitherBitMap],
        values: HypCollection,
        offset: int,
        cow: bool,
    ) -> None:
        bm_before = cls(values, copy_on_write=cow)
        bm_copy = cls(bm_before)
        bm_after = bm_before.shift(offset)
        assert bm_before == bm_copy
        expected = cls([val + offset for val in values if val + offset in range(0, 2**32)], copy_on_write=cow)
        assert bm_after == expected


class IncompatibleInteraction(Util):

    def incompatible_op(self, op: Callable[[BitMap, BitMap], object]) -> None:
        for cow1, cow2 in [(True, False), (False, True)]:
            bm1 = BitMap(copy_on_write=cow1)
            bm2 = BitMap(copy_on_write=cow2)
            with pytest.raises(ValueError):
                op(bm1, bm2)

    def test_incompatible_or(self) -> None:
        self.incompatible_op(lambda x, y: x | y)

    def test_incompatible_and(self) -> None:
        self.incompatible_op(lambda x, y: x & y)

    def test_incompatible_xor(self) -> None:
        self.incompatible_op(lambda x, y: x ^ y)

    def test_incompatible_sub(self) -> None:
        self.incompatible_op(lambda x, y: x - y)

    def test_incompatible_or_inplace(self) -> None:
        self.incompatible_op(lambda x, y: x.__ior__(y))

    def test_incompatible_and_inplace(self) -> None:
        self.incompatible_op(lambda x, y: x.__iand__(y))

    def test_incompatible_xor_inplace(self) -> None:
        self.incompatible_op(lambda x, y: x.__ixor__(y))

    def test_incompatible_sub_inplace(self) -> None:
        self.incompatible_op(lambda x, y: x.__isub__(y))

    def test_incompatible_eq(self) -> None:
        self.incompatible_op(lambda x, y: x == y)

    def test_incompatible_neq(self) -> None:
        self.incompatible_op(lambda x, y: x != y)

    def test_incompatible_le(self) -> None:
        self.incompatible_op(lambda x, y: x <= y)

    def test_incompatible_lt(self) -> None:
        self.incompatible_op(lambda x, y: x < y)

    def test_incompatible_ge(self) -> None:
        self.incompatible_op(lambda x, y: x >= y)

    def test_incompatible_gt(self) -> None:
        self.incompatible_op(lambda x, y: x > y)

    def test_incompatible_intersect(self) -> None:
        self.incompatible_op(lambda x, y: x.intersect(y))

    def test_incompatible_union(self) -> None:
        self.incompatible_op(lambda x, y: BitMap.union(x, y))
        self.incompatible_op(lambda x, y: BitMap.union(x, x, y, y, x, x, y, y))

    def test_incompatible_or_card(self) -> None:
        self.incompatible_op(lambda x, y: x.union_cardinality(y))

    def test_incompatible_and_card(self) -> None:
        self.incompatible_op(lambda x, y: x.intersection_cardinality(y))

    def test_incompatible_xor_card(self) -> None:
        self.incompatible_op(lambda x, y: x.symmetric_difference_cardinality(y))

    def test_incompatible_sub_card(self) -> None:
        self.incompatible_op(lambda x, y: x.difference_cardinality(y))

    def test_incompatible_jaccard(self) -> None:
        self.incompatible_op(lambda x, y: x.jaccard_index(y))


class TestBitMap:
    @given(hyp_collection, uint32)
    def test_iter_equal_or_larger(self, values: HypCollection, other_value: int) -> None:
        bm = BitMap(values)
        bm_iter = bm.iter_equal_or_larger(other_value)
        expected = [i for i in values if i >= other_value]
        expected.sort()

        observed = list(bm_iter)
        assert expected == observed

    def test_unashability(self) -> None:
        bm = BitMap()
        with pytest.raises(TypeError):
            hash(bm)


class TestFrozen:

    @given(hyp_collection, hyp_collection, integer)
    def test_immutability(self, values: HypCollection, raw_other: HypCollection, number: int) -> None:
        frozen = FrozenBitMap(values)
        copy = FrozenBitMap(values)
        other = BitMap(raw_other)
        with pytest.raises(AttributeError):
            frozen.clear()  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.pop()  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.add(number)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.update(other)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.discard(number)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.remove(number)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.intersection_update(other)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.difference_update(other)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.symmetric_difference_update(other)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.update(number, number + 10)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            frozen.overwrite(other)  # type: ignore[attr-defined]
        assert frozen == copy

    @given(hyp_collection, hyp_collection)
    def test_hash_uneq(self, values1: HypCollection, values2: HypCollection) -> None:
        """This test as a non null (but extremly low) probability to fail."""
        bitmap1 = FrozenBitMap(values1)
        bitmap2 = FrozenBitMap(values2)
        assume(bitmap1 != bitmap2)
        h1 = hash(bitmap1)
        h2 = hash(bitmap2)
        hd = hash(bitmap1 ^ bitmap2)
        hashes = [h1, h2, hd]
        nb_collisions = len(hashes) - len(set(hashes))
        assert 1 >= nb_collisions

    @given(hyp_collection)
    def test_hash_eq(self, values: HypCollection) -> None:
        bitmap1 = FrozenBitMap(values)
        bitmap2 = FrozenBitMap(values)
        bitmap3 = FrozenBitMap(bitmap1)
        assert hash(bitmap1) == hash(bitmap2)
        assert hash(bitmap1) == hash(bitmap3)

    def test_hash_eq2(self) -> None:
        """It can happen that two bitmaps hold the same values but have a different data structure. They should still
        have a same hash.
        This test compares two bitmaps with the same values, one has a run container, the other has an array container."""
        n = 100
        bm1 = FrozenBitMap(range(n))
        bm2 = BitMap()
        for i in range(n):
            bm2.add(i)
        bm2 = FrozenBitMap(bm2, optimize=False)  # type: ignore[assignment]
        assert bm1 == bm2
        assert bm1.get_statistics() != bm2.get_statistics()
        assert hash(bm1) == hash(bm2)


class TestOptimization:

    @given(bitmap_cls)
    def test_run_optimize(self, cls: type[EitherBitMap]) -> None:
        bm1 = BitMap()
        size = 1000
        for i in range(size):
            bm1.add(i)
        bm2 = cls(bm1, optimize=False)
        stats = bm2.get_statistics()
        assert bm1.get_statistics() == stats
        assert stats['n_containers'] == stats['n_array_containers']
        assert stats['n_values_array_containers'] == size
        assert bm2.run_optimize()
        stats = bm2.get_statistics()
        assert stats['n_containers'] == stats['n_run_containers']
        assert stats['n_values_run_containers'] == size
        bm3 = cls(bm1)  # optimize is True by default
        assert stats == bm3.get_statistics()

    @given(bitmap_cls)
    def test_shrink_to_fit(self, cls: type[EitherBitMap]) -> None:
        bm1 = BitMap()
        size = 1000
        for i in range(size):
            bm1.add(i)
        bm2 = cls(bm1, optimize=False)
        assert bm2.shrink_to_fit() > 0
        assert bm2.shrink_to_fit() == 0
        bm3 = cls(bm1, optimize=True)
        assert bm3.shrink_to_fit() == 0


small_integer = st.integers(min_value=0, max_value=200)
small_integer_list = st.lists(min_size=0, max_size=2000, elements=small_integer)


class TestPythonSetEquivalent:
    """
    The main goal of this class is to make sure the BitMap api is a superset of the python builtin set api.
    """

    @given(bitmap_cls, small_integer_list, st.booleans())
    def test_convert_to_set(self, BitMapClass: type[EitherBitMap], list1: list[int], cow: bool) -> None:
        """
        Most of the tests depend on a working implementation for converting from BitMap to python set.
        This test sanity checks it.

        This test should be modified or removed if you want to run PythonSetEquivalentTest with integers drawn from
        a larger set than `small_integer`. It will become prohibitively time-consuming.
        """
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
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
            assert (i in s1) == (i in converted_set)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_difference(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.difference(s2) == set(b1.difference(b2))
        assert SetClass.difference(s1, s2) == set(BitMapClass.difference(b1, b2))# type: ignore[arg-type]
        assert (s1 - s2) == set(b1 - b2)
        assert b1 - b2 == b1.difference(b2)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_symmetric_difference(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.symmetric_difference(s2) == set(b1.symmetric_difference(b2))
        assert SetClass.symmetric_difference(s1, s2) == set(BitMapClass.symmetric_difference(b1, b2))# type: ignore[arg-type]

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_union(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.union(s2) == set(b1.union(b2))
        assert SetClass.union(s1, s2) ==  set(BitMapClass.union(b1, b2))# type: ignore[arg-type]
        assert (s1 | s2) == set(b1 | b2)
        assert b1 | b2 == b1.union(b2)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_issubset(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.issubset(s2) == b1.issubset(b2)
        assert SetClass.issubset(s1, s2) == BitMapClass.issubset(b1, b2)# type: ignore[arg-type]

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_le(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.__le__(s2) == b1.__le__(b2)
        assert SetClass.__le__(s1, s2) == BitMapClass.__le__(b1, b2)# type: ignore[operator]
        assert (s1 <= s2) == (b1 <= b2)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_ge(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.__ge__(s2) == b1.__ge__(b2)
        assert SetClass.__ge__(s1, s2) == BitMapClass.__ge__(b1, b2)# type: ignore[operator]
        assert (s1 >= s2) == (b1 >= b2)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_eq(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()
        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.__eq__(s2) == b1.__eq__(b2)
        assert SetClass.__eq__(s1, s2) == BitMapClass.__eq__(b1, b2)# type: ignore[operator]
        assert (s1 == s2) == (b1 == b2)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_issuperset(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.issuperset(s2) == b1.issuperset(b2)
        assert SetClass.issuperset(s1, s2) == BitMapClass.issuperset(b1, b2)# type: ignore[arg-type]

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_isdisjoint(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.isdisjoint(s2) == b1.isdisjoint(b2)
        assert SetClass.isdisjoint(s1, s2) == BitMapClass.isdisjoint(b1, b2)# type: ignore[arg-type]

    @given(small_integer_list, st.booleans())
    def test_clear(self, list1: list[int], cow: bool) -> None:
        b1 = BitMap(list1, copy_on_write=cow)
        b1.clear()
        assert len(b1) == 0

    @given(small_integer_list, st.booleans())
    def test_pop(self, list1: list[int], cow: bool) -> None:
        b1 = BitMap(list1, copy_on_write=cow)
        starting_length = len(b1)
        if starting_length >= 1:
            popped_element = b1.pop()
            assert len(b1) == starting_length - 1# length decreased by one
            assert not popped_element in b1# and element isn't in the BitMap anymore
        else:
            with pytest.raises(KeyError):
                b1.pop()

    @given(bitmap_cls, small_integer_list, st.booleans())
    def test_copy(self, BitMapClass: type[EitherBitMap], list1: list[int], cow: bool) -> None:
        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = b1.copy()
        assert b2 == b1

    @given(small_integer_list, st.booleans())
    def test_copy_writable(self, list1: list[int], cow: bool) -> None:
        b1 = BitMap(list1, copy_on_write=cow)
        b2 = b1.copy()

        try:
            new_element = max(b1) + 1  # doesn't exist in the set
        except ValueError:
            new_element = 1

        b2.add(new_element)

        assert new_element in b2
        assert new_element not in b1

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_overwrite(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        assume(set(list1) != set(list2))
        b1 = BitMap(list1, copy_on_write=cow)
        orig1 = b1.copy()
        b2 = BitMapClass(list2, copy_on_write=cow)
        orig2 = b2.copy()
        b1.overwrite(b2)
        assert b1 == b2# the two bitmaps are now equal
        assert b1 != orig1# the first bitmap has been modified
        assert b2 == orig2# the second bitmap was left untouched
        with pytest.raises(ValueError):
            b1.overwrite(b1)

    @given(small_integer_list, small_integer_list, st.booleans())
    def test_difference_update(self, list1: list[int], list2: list[int], cow: bool) -> None:
        s1 = set(list1)
        s2 = set(list2)
        s1.difference_update(s2)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b1.difference_update(b2)

        assert s1 == set(b1)

    @given(small_integer_list, small_integer_list, st.booleans())
    def test_symmetric_difference_update(self, list1: list[int], list2: list[int], cow: bool) -> None:
        s1 = set(list1)
        s2 = set(list2)
        s1.symmetric_difference_update(s2)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b1.symmetric_difference_update(b2)

        assert s1 == set(b1)

    @given(bitmap_cls, small_integer_list, small_integer_list, st.booleans())
    def test_dunder(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], cow: bool) -> None:
        """
        Tests for &|^-
        """
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
        elif BitMapClass == FrozenBitMap:
            SetClass = frozenset
        else:
            raise AssertionError()

        s1 = SetClass(list1)
        s2 = SetClass(list2)

        b1 = BitMapClass(list1, copy_on_write=cow)
        b2 = BitMapClass(list2, copy_on_write=cow)

        assert s1.__and__(s2) == SetClass(b1.__and__(b2))
        assert s1.__or__(s2) == SetClass(b1.__or__(b2))
        assert s1.__xor__(s2) == SetClass(b1.__xor__(b2))
        assert s1.__sub__(s2) == SetClass(b1.__sub__(b2))

    @given(small_integer_list, small_integer, st.booleans())
    def test_add(self, list1: list[int], value: int, cow: bool) -> None:
        s1 = set(list1)
        b1 = BitMap(list1, copy_on_write=cow)
        assert s1 == set(b1)

        s1.add(value)
        b1.add(value)
        assert s1 == set(b1)

    @given(small_integer_list, small_integer, st.booleans())
    def test_discard(self, list1: list[int], value: int, cow: bool) -> None:
        s1 = set(list1)
        b1 = BitMap(list1, copy_on_write=cow)
        assert s1 == set(b1)

        s1.discard(value)
        b1.discard(value)
        assert s1 == set(b1)

    @given(small_integer_list, small_integer, st.booleans())
    def test_remove(self, list1: list[int], value: int, cow: bool) -> None:
        s1 = set(list1)
        b1 = BitMap(list1, copy_on_write=cow)
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
        assert s1_raised == b1_raised# Either both raised exception or neither did

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_union(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], list3: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
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

        assert SetClass.union(s1, s2, s3) == SetClass(BitMapClass.union(b1, b2, b3))# type: ignore[arg-type]
        assert s1.union(s2, s3) == SetClass(b1.union(b2, b3))

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_difference(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], list3: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
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

        assert SetClass.difference(s1, s2, s3) == SetClass(BitMapClass.difference(b1, b2, b3))# type: ignore[arg-type]
        assert s1.difference(s2, s3) == SetClass(b1.difference(b2, b3))

    @given(bitmap_cls, small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_intersection(self, BitMapClass: type[EitherBitMap], list1: list[int], list2: list[int], list3: list[int], cow: bool) -> None:
        if BitMapClass == BitMap:
            SetClass: type[EitherSet] = set
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

        assert SetClass.intersection(s1, s2, s3) == SetClass(BitMapClass.intersection(b1, b2, b3))# type: ignore[arg-type]
        assert s1.intersection(s2, s3) == SetClass(b1.intersection(b2, b3))

    @given(small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_intersection_update(self, list1: list[int], list2: list[int], list3: list[int], cow: bool) -> None:
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        set.intersection_update(s1, s2, s3)
        BitMap.intersection_update(b1, b2, b3)
        assert s1 == set(b1)

        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        s1.intersection_update(s2, s3)
        b1.intersection_update(b2, b3)

        assert s1 == set(b1)

    @given(small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_difference_update(self, list1: list[int], list2: list[int], list3: list[int], cow: bool) -> None:
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        set.difference_update(s1, s2, s3)
        BitMap.difference_update(b1, b2, b3)
        assert s1 == set(b1)

        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        s1.difference_update(s2, s3)
        b1.difference_update(b2, b3)

        assert s1 == set(b1)

    @given(small_integer_list, small_integer_list, small_integer_list, st.booleans())
    def test_nary_update(self, list1: list[int], list2: list[int], list3: list[int], cow: bool) -> None:
        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        set.update(s1, s2, s3)
        BitMap.update(b1, b2, b3)
        assert s1 == set(b1)

        s1 = set(list1)
        s2 = set(list2)
        s3 = set(list3)

        b1 = BitMap(list1, copy_on_write=cow)
        b2 = BitMap(list2, copy_on_write=cow)
        b3 = BitMap(list3, copy_on_write=cow)

        s1.update(s2, s3)
        b1.update(b2, b3)

        assert s1 == set(b1)


small_list_of_uin32 = st.lists(min_size=0, max_size=400, elements=uint32)
large_list_of_uin32 = st.lists(min_size=600, max_size=1000, elements=uint32, unique=True)


class TestString:

    @given(bitmap_cls, small_list_of_uin32)
    def test_small_list(self, cls: type[EitherBitMap], collection: list[int]) -> None:
        # test that repr for a small bitmap is equal to the original bitmap
        bm = cls(collection)
        assert bm == eval(repr(bm))

    @settings(suppress_health_check=HealthCheck)
    @given(bitmap_cls, large_list_of_uin32)
    def test_large_list(self, cls: type[EitherBitMap], collection: list[int]) -> None:
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
            assert i in bm

        for i in large_ints:
            assert i in bm

        assert min(small_ints) == min(bm)
        assert max(large_ints) == max(bm)


class TestVersion:
    def assert_regex(self, pattern: str, text: str) -> None:
        matches = re.findall(pattern, text)
        if len(matches) != 1 or matches[0] != text:
            pytest.fail('Regex "%s" does not match text "%s".' % (pattern, text))

    def test_version(self) -> None:
        self.assert_regex(r'\d+\.\d+\.\d+(?:\.dev\d+)?', pyroaring.__version__)
        self.assert_regex(r'v\d+\.\d+\.\d+', pyroaring.__croaring_version__)


if __name__ == "__main__":
    unittest.main()
