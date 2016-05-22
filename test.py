#! /usr/bin/env python3

import unittest
import random
import functools
import os
from pyroaring import BitMap, load, dump

class Util(unittest.TestCase):

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

    @staticmethod
    def get_random_set(universe, set_proportion=None):
        if set_proportion is None:
            set_proportion = 0.1
        else:
            assert set_proportion > 0 and set_proportion < 1
        min_number = int(len(universe)*set_proportion)
        max_number = min_number*2
        size = random.randint(min_number, max_number)
        return set(random.sample(universe, size))

class BasicTest(Util):

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

class BinaryOperationsTest(Util):

    def setUp(self):
        self.universe = range(100)
        self.set1 = self.get_random_set(self.universe)
        self.set2 = self.get_random_set(self.universe)
        self.bitmap1 = BitMap(self.set1)
        self.bitmap2 = BitMap(self.set2)

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
            self.setUp()
            self.do_test_binary_op(lambda x,y : x|y)

    def test_and(self):
        for _ in range(10):
            self.setUp()
            self.do_test_binary_op(lambda x,y : x&y)

    def do_test_binary_op_inplace(self, op):
        old_bitmap2 = BitMap(self.bitmap2)
        op(self.set1, self.set2)
        op(self.bitmap1, self.bitmap2)
        self.assertEqual(self.bitmap2, old_bitmap2)
        self.compare_with_set(self.bitmap1, self.set1, self.universe)

    def test_or_inplace(self):
        for _ in range(10):
            self.setUp()
            self.do_test_binary_op_inplace(lambda x,y : x.__ior__(y))

    def test_and_inplace(self):
        for _ in range(10):
            self.setUp()
            self.do_test_binary_op_inplace(lambda x,y : x.__iand__(y))

class ManyOperationsTest(Util):

    def setUp(self):
        self.universe = range(1000)
        self.bitmaps = []
        self.nb_bitmaps = random.randint(5, 30)
        for _ in range(self.nb_bitmaps):
            self.bitmaps.append(BitMap(self.get_random_set(self.universe, set_proportion=0.01)))

    def do_test_or_many(self):
        copy = [BitMap(bm) for bm in self.bitmaps]
        result = BitMap.or_many(self.bitmaps)
        self.assertEqual(copy, self.bitmaps)
        expected_result = functools.reduce(lambda x, y: x|y, self.bitmaps)
        self.assertEqual(expected_result, result)

    def test_or_many(self):
        for _ in range(10):
            self.setUp()
            self.do_test_or_many()

class SerializationTest(Util):

    def do_test_serialization(self):
        old_bm = BitMap(self.get_random_set(range(1000)))
        buff = old_bm.serialize()
        new_bm = BitMap.deserialize(buff)
        self.assertEqual(old_bm, new_bm)
        old_bm.add(1001)
        self.assertNotEqual(old_bm, new_bm)

    def test_serialization(self):
        for _ in range(100):
            self.do_test_serialization()

    def test_load_dump(self):
        filepath = 'testfile'
        old_bm = BitMap(self.get_random_set(range(1000)))
        with open(filepath, 'wb') as f:
            dump(f, old_bm)
        with open(filepath, 'rb') as f:
            new_bm = load(f)
        self.assertEqual(old_bm, new_bm)
        os.remove(filepath)

if __name__ == "__main__":
    unittest.main()
