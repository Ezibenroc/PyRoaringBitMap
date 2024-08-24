from __future__ import annotations  # for using set[int] in Python 3.8

import hypothesis.strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule
from hypothesis import settings
from dataclasses import dataclass
from pyroaring import BitMap, BitMap64
from test import hyp_collection, uint32, uint64, is_32_bits

if is_32_bits:
    BitMapClass = BitMap
    int_class = uint32
    large_val = 2**30
else:
    BitMapClass = BitMap64
    int_class = uint64
    large_val = 2**40

@dataclass
class Collection:
    test: BitMapClass
    ref: set[int]

    def check(self):
        assert len(self.test) == len(self.ref)
        assert set(self.test) == self.ref

    def __post_init__(self):
        self.check()


class SetComparison(RuleBasedStateMachine):
    collections = Bundle("collections")

    @rule(target=collections, val=hyp_collection)
    def init_collection(self, val):
        return Collection(test=BitMapClass(val), ref=set(val))

    @rule(target=collections, col=collections)
    def copy(self, col):
        return Collection(test=BitMapClass(col.test), ref=set(col.ref))

    @rule(col=collections, val=int_class)
    def add_elt(self, col, val):
        col.test.add(val)
        col.ref.add(val)
        col.check()

    @rule(col=collections, val=int_class)
    def remove_elt(self, col, val):
        col.test.discard(val)
        col.ref.discard(val)
        col.check()

    @rule(target=collections, col1=collections, col2=collections)
    def union(self, col1, col2):
        return Collection(test=col1.test | col2.test, ref=col1.ref | col2.ref)

    @rule(col1=collections, col2=collections)
    def union_inplace(self, col1, col2):
        col1.test |= col2.test
        col1.ref |= col2.ref
        col1.check()

    @rule(target=collections, col1=collections, col2=collections)
    def intersection(self, col1, col2):
        return Collection(test=col1.test & col2.test, ref=col1.ref & col2.ref)

    @rule(col1=collections, col2=collections)
    def intersection_inplace(self, col1, col2):
        col1.test &= col2.test
        col1.ref &= col2.ref
        col1.check()

    @rule(target=collections, col1=collections, col2=collections)
    def difference(self, col1, col2):
        return Collection(test=col1.test - col2.test, ref=col1.ref - col2.ref)

    @rule(col1=collections, col2=collections)
    def difference_inplace(self, col1, col2):
        col1.test -= col2.test
        col1.ref -= col2.ref
        col1.check()

    @rule(target=collections, col1=collections, col2=collections)
    def symmetric_difference(self, col1, col2):
        return Collection(test=col1.test ^ col2.test, ref=col1.ref ^ col2.ref)

    @rule(col1=collections, col2=collections)
    def symmetric_difference_inplace(self, col1, col2):
        col1.test ^= col2.test
        col1.ref ^= col2.ref
        col1.check()

    @rule(
        target=collections,
        col=collections,
        start=st.integers(min_value=0, max_value=large_val),
        size=st.integers(min_value=0, max_value=2**18),
    )
    def flip(self, col, start, size):
        stop = start + size
        return Collection(
            test=col.test.flip(start, stop), ref=col.ref ^ set(range(start, stop))
        )

    @rule(
        col=collections,
        start=st.integers(min_value=0, max_value=large_val),
        size=st.integers(min_value=0, max_value=2**18),
    )
    def flip_inplace(self, col, start, size):
        stop = start + size
        col.test.flip_inplace(start, stop)
        col.ref ^= set(range(start, stop))
        col.check()


TestTrees = SetComparison.TestCase
TestTrees.settings = settings(max_examples=100, stateful_step_count=100)

if __name__ == "__main__":
    unittest.main()
