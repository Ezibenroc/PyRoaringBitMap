#! /usr/bin/env python3

import unittest
import random
from pyroaring import BitMap

class BaseTest(unittest.TestCase):

    def compare_with_set(self, bitmap, expected_set, universe):
        self.assertEqual(len(bitmap), len(expected_set))
        self.assertEqual(set(bitmap), expected_set)
        self.assertEqual(sorted(list(bitmap)), sorted(list(expected_set)))
        self.assertEqual(BitMap(expected_set), bitmap)
        for value in universe:
            if value in expected_set:
                self.assertIn(value, bitmap)
            else:
                self.assertNotIn(value, bitmap)

    def test_basic(self):
        bitmap = BitMap()
        expected_set = set()
        universe = range(100)
        self.compare_with_set(bitmap, expected_set, universe)
        values = list(universe)
        random.shuffle(values)
        for value in values:
            bitmap.add(value)
            expected_set.add(value)
            self.compare_with_set(bitmap, expected_set, universe)

    def test_bitmap_equality(self):
        bitmap1 = BitMap()
        bitmap2 = BitMap()
        self.assertEqual(bitmap1, bitmap2)
        bitmap1.add(42)
        self.assertNotEqual(bitmap1, bitmap2)
        bitmap2.add(27)
        self.assertNotEqual(bitmap1, bitmap2)
        bitmap1.add(27)
        self.assertNotEqual(bitmap1, bitmap2)
        bitmap2.add(42)
        self.assertEqual(bitmap1, bitmap2)

    def test_constructor_values(self):
        values = range(10,80,3)
        bitmap = BitMap(values)
        expected_set = set(values)
        universe = range(100)
        self.compare_with_set(bitmap, expected_set, universe)

    def test_constructor_copy(self):
        bitmap1 = BitMap(range(10))
        bitmap2 = BitMap(bitmap1)
        self.assertEqual(bitmap1, bitmap2)
        bitmap1.add(42)
        self.assertNotEqual(bitmap1, bitmap2)

class OperationsTest(BaseTest):

    def setUp(self):
        self.universe = range(100)
        self.set1 = self.get_random_set(self.universe)
        self.set2 = self.get_random_set(self.universe)
        self.bitmap1 = BitMap(self.set1)
        self.bitmap2 = BitMap(self.set2)

    @staticmethod
    def get_random_set(universe):
        size = random.randint(len(universe)/10, 4*len(universe)/10)
        return set(random.sample(universe, size))

    def do_test_binary_op(self, op):
        old_bitmap1 = BitMap(self.bitmap1)
        old_bitmap2 = BitMap(self.bitmap2)
        result_set = op(self.set1, self.set2)
        result_bitmap = op(self.bitmap1, self.bitmap2)
        self.assertEqual(self.bitmap1, old_bitmap1)
        self.assertEqual(self.bitmap2, old_bitmap2)
        self.compare_with_set(result_bitmap, result_set, self.universe)

    def test_or(self):
        for _ in range(10):
            self.do_test_binary_op(lambda x,y : x|y)

    def test_and(self):
        for _ in range(10):
            self.do_test_binary_op(lambda x,y : x&y)

    def do_test_binary_op_inplace(self, op):
        old_bitmap2 = BitMap(self.bitmap2)
        op(self.set1, self.set2)
        op(self.bitmap1, self.bitmap2)
        self.assertEqual(self.bitmap2, old_bitmap2)
        self.compare_with_set(self.bitmap1, self.set1, self.universe)

    def test_or_inplace(self):
        for _ in range(10):
            self.do_test_binary_op_inplace(lambda x,y : x.__ior__(y))

    def test_and_inplace(self):
        for _ in range(10):
            self.do_test_binary_op_inplace(lambda x,y : x.__iand__(y))

if __name__ == "__main__":
    unittest.main()
