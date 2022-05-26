cimport croaring
from libc.stdint cimport uint32_t, uint64_t, int64_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdlib cimport free, malloc

from cpython cimport array
import array

try:
    range = xrange
except NameError: # python 3
    pass

def print_platform():
    croaring.print_platform_information()

cdef croaring.roaring_bitmap_t *deserialize_ptr(char *buff):
    cdef croaring.roaring_bitmap_t *ptr
    ptr = croaring.roaring_bitmap_portable_deserialize(buff)
    return ptr

cdef class AbstractBitMap:
    """
    An efficient and light-weight ordered set of 32 bits integers.
    """
    cdef croaring.roaring_bitmap_t* _c_bitmap
    cdef int64_t _h_val

    def __cinit__(self, values=None, copy_on_write=False, optimize=True, no_init=False):
        if no_init:
            assert values is None and not copy_on_write
            return
        cdef vector[uint32_t] buff_vect
        cdef unsigned[:] buff
        if values is None:
            self._c_bitmap = croaring.roaring_bitmap_create()
        elif isinstance(values, AbstractBitMap):
            self._c_bitmap = croaring.roaring_bitmap_copy((<AbstractBitMap?>values)._c_bitmap)
            self._h_val = (<AbstractBitMap?>values)._h_val
        elif isinstance(values, range):
            _, (start, stop, step) = values.__reduce__()
            if step < 0:
                values = range(min(values), max(values)+1, -step)
                _, (start, stop, step) = values.__reduce__()
            if start >= stop:
                self._c_bitmap = croaring.roaring_bitmap_create()
            else:
                self._c_bitmap = croaring.roaring_bitmap_from_range(start, stop, step)
        elif isinstance(values, array.array):
            size = len(values)
            if size == 0:
                self._c_bitmap = croaring.roaring_bitmap_create()
            else:
                buff = <array.array> values
                self._c_bitmap = croaring.roaring_bitmap_of_ptr(size, &buff[0])
        else:
            try:
                size = len(values)
            except TypeError:  # object has no length, creating a list
                values = list(values)
                size = len(values)
            self._c_bitmap = croaring.roaring_bitmap_create()
            if size > 0:
                buff_vect = values
                croaring.roaring_bitmap_add_many(self._c_bitmap, size, &buff_vect[0])
        if not isinstance(values, AbstractBitMap):
            croaring.roaring_bitmap_set_copy_on_write(self._c_bitmap, copy_on_write)
            self._h_val = 0
        if optimize:
            self.run_optimize()
            self.shrink_to_fit()

    def __init__(self, values=None, copy_on_write=False, optimize=True):
        """
        Construct a AbstractBitMap object, either empry or from an iterable.

        Copy on write can be enabled with the field copy_on_write.

        >>> BitMap()
        BitMap([])
        >>> BitMap([1, 123456789, 27])
        BitMap([1, 27, 123456789])
        >>> BitMap([1, 123456789, 27], copy_on_write=True)
        BitMap([1, 27, 123456789])
        """

    cdef from_ptr(self, croaring.roaring_bitmap_t *ptr):
        """
        Return an instance of AbstractBitMap (or one of its subclasses) initialized with the given pointer.

        FIXME: this should be a classmethod, but this is (currently) impossible for cdef methods.
        See https://groups.google.com/forum/#!topic/cython-users/FLHiLzzKqj4
        """
        bm = self.__class__.__new__(self.__class__, no_init=True)
        (<AbstractBitMap>bm)._c_bitmap = ptr
        return bm

    @property
    def copy_on_write(self):
        """
        True if and only if the bitmap has "copy on write" optimization enabled.

        >>> BitMap(copy_on_write=False).copy_on_write
        False
        >>> BitMap(copy_on_write=True).copy_on_write
        True
        """
        return croaring.roaring_bitmap_get_copy_on_write(self._c_bitmap)

    def run_optimize(self):
        return croaring.roaring_bitmap_run_optimize(self._c_bitmap)

    def shrink_to_fit(self):
        return croaring.roaring_bitmap_shrink_to_fit(self._c_bitmap)

    def __dealloc__(self):
        if self._c_bitmap is not NULL:
            croaring.roaring_bitmap_free(self._c_bitmap)

    def __check_compatibility(self, AbstractBitMap other):
        if self.copy_on_write != other.copy_on_write:
            raise ValueError('Cannot have interactions between bitmaps with and without copy_on_write.\n')

    def __contains__(self, uint32_t value):
        return croaring.roaring_bitmap_contains(self._c_bitmap, value)

    def __bool__(self):
        return not croaring.roaring_bitmap_is_empty(self._c_bitmap)

    def __len__(self):
        return croaring.roaring_bitmap_get_cardinality(self._c_bitmap)

    def __richcmp__(self, other, int op):
        self.__check_compatibility(other)
        if op == 0: # <
            return croaring.roaring_bitmap_is_strict_subset((<AbstractBitMap?>self)._c_bitmap, (<AbstractBitMap?>other)._c_bitmap)
        elif op == 1: # <=
            return croaring.roaring_bitmap_is_subset((<AbstractBitMap?>self)._c_bitmap, (<AbstractBitMap?>other)._c_bitmap)
        elif op == 2: # ==
            return croaring.roaring_bitmap_equals((<AbstractBitMap?>self)._c_bitmap, (<AbstractBitMap?>other)._c_bitmap)
        elif op == 3: # !=
            return not (self == other)
        elif op == 4: # >
            return croaring.roaring_bitmap_is_strict_subset((<AbstractBitMap?>other)._c_bitmap, (<AbstractBitMap?>self)._c_bitmap)
        else:         # >=
            assert op == 5
            return croaring.roaring_bitmap_is_subset((<AbstractBitMap?>other)._c_bitmap, (<AbstractBitMap?>self)._c_bitmap)

    def contains_range(self, uint64_t range_start, uint64_t range_end):
        """
        Check whether a range of values from range_start (included) to range_end (excluded) is present.

        >>> bm = BitMap([5, 6, 7, 8, 9, 10])
        >>> bm.contains_range(6, 9)
        True
        >>> bm.contains_range(8, 12)
        False
        """
        if range_end <= range_start or range_end == 0 or range_start >= 2**32:
            return True  # empty range
        if range_end >= 2**32:
            range_end = 2**32
        return croaring.roaring_bitmap_contains_range(self._c_bitmap, range_start, range_end)

    def range_cardinality(self, uint64_t range_start, uint64_t range_end):
        """
        Return cardinality from range_start (included) to range_end (excluded).

        >>> bm = BitMap(range(10))
        >>> bm.range_cardinality(0, 10)
        10
        >>> bm.range_cardinality(10, 100)
        0
        """
        if range_end < range_start:
            raise AssertionError('range_end must not be lower than range_start')
        return croaring.roaring_bitmap_range_cardinality(self._c_bitmap, range_start, range_end)

    cdef compute_hash(self):
        cdef int64_t h_val = 0
        cdef uint32_t i, count, max_count=256
        cdef croaring.roaring_uint32_iterator_t *iterator = croaring.roaring_create_iterator(self._c_bitmap)
        cdef uint32_t *buff = <uint32_t*>malloc(max_count*4)
        while True:
            count = croaring.roaring_read_uint32_iterator(iterator, buff, max_count)
            i = 0
            while i < count:
                h_val = (h_val << 2) + buff[i] + 1
                # TODO find a good hash formula
                # This one should be better, but is too long:
                # h_val = ((h_val<<16) + buff[i]) % 1748104473534059
                i += 1
            if count != max_count:
                break
        croaring.roaring_free_uint32_iterator(iterator)
        free(buff)
        if not self:
            return -1
        return h_val

    def __hash__(self):
        if self._h_val == 0: 
            self._h_val = self.compute_hash()
        return self._h_val

    def iter_equal_or_larger(self, uint32_t val):
        """
        Iterate over items in the bitmap equal or larger than a given value.

        >>> bm = BitMap([1, 2, 4])
        >>> list(bm.iter_equal_or_larger(2))
        [2, 4]
        """
        cdef croaring.roaring_uint32_iterator_t *iterator = croaring.roaring_create_iterator(self._c_bitmap)
        valid = croaring.roaring_move_uint32_iterator_equalorlarger(iterator, val)
        if not valid:
            return
        try:
            while iterator.has_value:
                yield iterator.current_value
                croaring.roaring_advance_uint32_iterator(iterator)
        finally:
            croaring.roaring_free_uint32_iterator(iterator)

    def __iter__(self):
        cdef croaring.roaring_uint32_iterator_t *iterator = croaring.roaring_create_iterator(self._c_bitmap)
        try:
            while iterator.has_value:
                yield iterator.current_value
                croaring.roaring_advance_uint32_iterator(iterator)
        finally:
            croaring.roaring_free_uint32_iterator(iterator)

    def __repr__(self):
        return str(self)

    def __str__(self):
        skip_rows = len(self) > 500 #this is the cutoff number for the truncating to kick in.
        table_max_width = 80  # this isn't the length of the entire output, it's only for the numeric part
        num_lines_if_skipping = 5  # the number of lines to show in the beginning and the end when output is being truncated

        head = self.__class__.__name__ + '(['
        row_start_buffer = ' ' * len(head)
        tail = '])'

        try:
            maxval = self.max()
        except ValueError:
            # empty bitmap
            return head + tail

        element_max_length = len(str(maxval))
        column_width = element_max_length + 2

        num_columns = table_max_width // column_width

        num_rows = len(self) / float(num_columns)
        if not num_rows.is_integer():
            num_rows += 1
        num_rows = int(num_rows)
        rows = []
        row_idx = 0
        skipped = False
        while row_idx < num_rows:
            row_ints = self[row_idx * num_columns:(row_idx + 1) * num_columns]

            line = []
            for i in row_ints:
                s = str(i)
                if num_rows == 1:
                    # no padding if all numbers fit on a single line
                    line.append(s)
                else:
                    line.append(' ' * (element_max_length - len(s)) + s)

            if row_idx == 0:
                prefix = head
            else:
                prefix = row_start_buffer
            rows.append(prefix + ', '.join(line) + ',')
            row_idx += 1
            if skip_rows and not skipped and row_idx >= num_lines_if_skipping:
                rows.append((' ' * ((table_max_width + len(head)) // 2)) + '...')
                skipped = True
                row_idx = num_rows - num_lines_if_skipping

        rows[-1] = rows[-1].rstrip(',')  # remove trailing comma from the last line
        return '\n'.join(rows) + tail

    def flip(self, uint64_t start, uint64_t end):
        """
        Compute the negation of the bitmap within the specified interval.

        Areas outside the range are passed unchanged.

        >>> bm = BitMap([3, 12])
        >>> bm.flip(10, 15)
        BitMap([3, 10, 11, 13, 14])
        """
        return self.from_ptr(croaring.roaring_bitmap_flip(self._c_bitmap, start, end))

    def shift(self, int64_t offset):
        """
        Add the value 'offset' to each and every value of the bitmap.

        If offset + element is outside of the range [0,2^32), that the element will be dropped.

        >>> bm = BitMap([3, 12])
        >>> bm.shift(21)
        BitMap([24, 33])
        """
        return self.from_ptr(croaring.roaring_bitmap_add_offset(self._c_bitmap, offset))

    def copy(self):
        """
        Return a copy of a set.

        >>> bm = BitMap([3, 12])
        >>> bm2 = bm.copy()
        >>> bm == bm2
        True
        >>> bm.add(1)
        >>> bm == bm2
        False

        """
        return self.__class__(self)

    def isdisjoint(self, other):
        """
        Return True if two sets have a null intersection.

        >>> BitMap([1, 2]).isdisjoint(BitMap([3, 4]))
        True

        >>> BitMap([1, 2, 3]).isdisjoint(BitMap([3, 4]))
        False

        """
        return self.intersection_cardinality(other) == 0

    def issubset(self, other):
        """
        Report whether another set contains this set.

        >>> BitMap([1, 2]).issubset(BitMap([1, 2, 3, 4]))
        True

        >>> BitMap([1, 2]).issubset(BitMap([3, 4]))
        False

        """
        return self <= other

    def issuperset(self, other):
        """
        Report whether this set contains another set.

        >>> BitMap([1, 2, 3, 4]).issuperset(BitMap([1, 2]))
        True

        >>> BitMap([1, 2]).issuperset(BitMap([3, 4]))
        False

        """
        return self >= other

    def difference(*bitmaps):
        """
        Return the difference of two or more sets as a new set.

        (i.e. all elements that are in this set but not the others.)

        >>> BitMap.difference(BitMap([1, 2, 3]), BitMap([2, 20]), BitMap([3, 30]))
        BitMap([1])

        """
        size = len(bitmaps)
        cdef AbstractBitMap result, bm
        if size <= 1:
            return bitmaps[0].copy()
        elif size == 2:
            return bitmaps[0] - bitmaps[1]
        else:
            result = BitMap(bitmaps[0])
            for bm in bitmaps[1:]:
                result -= bm
            return bitmaps[0].__class__(result)


    def symmetric_difference(self, other):
        """
        Return the symmetric difference of two sets as a new set.

        (i.e. all elements that are in exactly one of the sets.)

        >>> BitMap([1, 2, 3]).symmetric_difference(BitMap([2, 3, 4]))
        BitMap([1, 4])
        """
        return self.__xor__(other)

    def union(*bitmaps):
        """
        Return the union of the bitmaps.

        >>> BitMap.union(BitMap([3, 12]), BitMap([5]), BitMap([0, 10, 12]))
        BitMap([0, 3, 5, 10, 12])
        """
        size = len(bitmaps)
        cdef croaring.roaring_bitmap_t *result
        cdef AbstractBitMap bm
        cdef vector[const croaring.roaring_bitmap_t*] buff
        if size <= 1:
            return bitmaps[0].copy()
        elif size == 2:
            return bitmaps[0] | bitmaps[1]
        else:
            for bm in bitmaps:
                bitmaps[0].__check_compatibility(bm)
                buff.push_back(bm._c_bitmap)
            result = croaring.roaring_bitmap_or_many(size, &buff[0])
            return (<AbstractBitMap>bitmaps[0].__class__()).from_ptr(result) # FIXME to change when from_ptr is a classmethod

    def intersection(*bitmaps): # FIXME could be more efficient
        """
        Return the intersection of the bitmaps.

        >>> BitMap.intersection(BitMap(range(0, 15)), BitMap(range(5, 20)), BitMap(range(10, 25)))
        BitMap([10, 11, 12, 13, 14])
        """
        size = len(bitmaps)
        cdef AbstractBitMap result, bm
        if size <= 1:
            return bitmaps[0].copy()
        elif size == 2:
            return bitmaps[0] & bitmaps[1]
        else:
            result = BitMap(bitmaps[0])
            for bm in bitmaps[1:]:
                result &= bm
            return bitmaps[0].__class__(result)

    cdef binary_op(self, AbstractBitMap other, (croaring.roaring_bitmap_t*)func(const croaring.roaring_bitmap_t*, const croaring.roaring_bitmap_t*)):
        self.__check_compatibility(other)
        cdef croaring.roaring_bitmap_t *r = func(self._c_bitmap, other._c_bitmap)
        return self.from_ptr(r)

    cdef binary_iop(self, AbstractBitMap other, (void)func(croaring.roaring_bitmap_t*, const croaring.roaring_bitmap_t*)):
        self.__check_compatibility(other)
        func(self._c_bitmap, other._c_bitmap)
        return self

    def __or__(self, other):
        return (<AbstractBitMap>self).binary_op(<AbstractBitMap?>other, croaring.roaring_bitmap_or)

    def __ior__(self, other):
        return (<AbstractBitMap>self).binary_iop(<AbstractBitMap?>other, croaring.roaring_bitmap_or_inplace)

    def __and__(self, other):
        return (<AbstractBitMap>self).binary_op(<AbstractBitMap?>other, croaring.roaring_bitmap_and)

    def __iand__(self, other):
        return (<AbstractBitMap>self).binary_iop(<AbstractBitMap?>other, croaring.roaring_bitmap_and_inplace)

    def __xor__(self, other):
        return (<AbstractBitMap>self).binary_op(<AbstractBitMap?>other, croaring.roaring_bitmap_xor)

    def __ixor__(self, other):
        return (<AbstractBitMap>self).binary_iop(<AbstractBitMap?>other, croaring.roaring_bitmap_xor_inplace)

    def __sub__(self, other):
        return (<AbstractBitMap>self).binary_op(<AbstractBitMap?>other, croaring.roaring_bitmap_andnot)

    def __isub__(self, other):
        return (<AbstractBitMap>self).binary_iop(<AbstractBitMap?>other, croaring.roaring_bitmap_andnot_inplace)

    def union_cardinality(self, AbstractBitMap other):
        """
        Return the number of elements in the union of the two bitmaps.

        It is equivalent to len(self | other), but faster.

        >>> BitMap([3, 12]).union_cardinality(AbstractBitMap([3, 5, 8]))
        4
        """
        self.__check_compatibility(other)
        return croaring.roaring_bitmap_or_cardinality(self._c_bitmap, other._c_bitmap)

    def intersection_cardinality(self, AbstractBitMap other):
        """
        Return the number of elements in the intersection of the two bitmaps.

        It is equivalent to len(self & other), but faster.

        >>> BitMap([3, 12]).intersection_cardinality(BitMap([3, 5, 8]))
        1
        """
        self.__check_compatibility(other)
        return croaring.roaring_bitmap_and_cardinality(self._c_bitmap, other._c_bitmap)

    def difference_cardinality(self, AbstractBitMap other):
        """
        Return the number of elements in the difference of the two bitmaps.

        It is equivalent to len(self - other), but faster.

        >>> BitMap([3, 12]).difference_cardinality(BitMap([3, 5, 8]))
        1
        """
        self.__check_compatibility(other)
        return croaring.roaring_bitmap_andnot_cardinality(self._c_bitmap, other._c_bitmap)

    def symmetric_difference_cardinality(self, AbstractBitMap other):
        """
        Return the number of elements in the symmetric difference of the two bitmaps.

        It is equivalent to len(self ^ other), but faster.

        >>> BitMap([3, 12]).symmetric_difference_cardinality(BitMap([3, 5, 8]))
        3
        """
        self.__check_compatibility(other)
        return croaring.roaring_bitmap_xor_cardinality(self._c_bitmap, other._c_bitmap)

    def intersect(self, AbstractBitMap other):
        """
        Return True if and only if the two bitmaps have elements in common.

        It is equivalent to len(self & other) > 0, but faster.

        >>> BitMap([3, 12]).intersect(BitMap([3, 18]))
        True
        >>> BitMap([3, 12]).intersect(BitMap([5, 18]))
        False
        """
        self.__check_compatibility(other)
        return croaring.roaring_bitmap_intersect(self._c_bitmap, other._c_bitmap)

    def jaccard_index(self, AbstractBitMap other):
        """
        Compute the Jaccard index of the two bitmaps.

        It is equivalent to len(self&other)/len(self|other), but faster.
        See https://en.wikipedia.org/wiki/Jaccard_index
        
        >>> BitMap([3, 10, 12]).jaccard_index(BitMap([3, 18]))
        0.25
        """
        self.__check_compatibility(other)
        return croaring.roaring_bitmap_jaccard_index(self._c_bitmap, other._c_bitmap)

    def get_statistics(self):
        """
        Return relevant metrics about the bitmap.

        >>> stats = BitMap(range(18, 66000, 2)).get_statistics()
        >>> stats['cardinality']
        32991
        >>> stats['max_value']
        65998
        >>> stats['min_value']
        18
        >>> stats['n_array_containers']
        1
        >>> stats['n_bitset_containers']
        1
        >>> stats['n_bytes_array_containers']
        464
        >>> stats['n_bytes_bitset_containers']
        8192
        >>> stats['n_bytes_run_containers']
        0
        >>> stats['n_containers']
        2
        >>> stats['n_run_containers']
        0
        >>> stats['n_values_array_containers']
        232
        >>> stats['n_values_bitset_containers']
        32759
        >>> stats['n_values_run_containers']
        0
        >>> stats['sum_value'] 
        1088966928
        """
        cdef croaring.roaring_statistics_t stat
        croaring.roaring_bitmap_statistics(self._c_bitmap, &stat)
        return stat

    def min(self):
        """
        Return the minimum element of the bitmap.

        It is equivalent to min(self), but faster.

        >>> BitMap([3, 12]).min()
        3
        """
        if len(self) == 0:
            raise ValueError('Empty roaring bitmap, there is no minimum.')
        else:
            return croaring.roaring_bitmap_minimum(self._c_bitmap)

    def max(self):
        """
        Return the maximum element of the bitmap.

        It is equivalent to max(self), but faster.

        >>> BitMap([3, 12]).max()
        12
        """
        if len(self) == 0:
            raise ValueError('Empty roaring bitmap, there is no maximum.')
        else:
            return croaring.roaring_bitmap_maximum(self._c_bitmap)

    def rank(self, uint32_t value):
        """
        Return the rank of the element in the bitmap.

        >>> BitMap([3, 12]).rank(12)
        2
        """
        return croaring.roaring_bitmap_rank(self._c_bitmap, value)

    def next_set_bit(self, uint32_t value):
        """
        Return the next set bit larger or equal to the given value.

        >>> BitMap([1, 2, 4]).next_set_bit(1)
        1

        >>> BitMap([1, 2, 4]).next_set_bit(3)
        4

        >>> BitMap([1, 2, 4]).next_set_bit(5)
        Traceback (most recent call last):
        ValueError: No value larger or equal to specified value.
        """
        try:
            return next(self.iter_equal_or_larger(value))
        except StopIteration:
            raise ValueError('No value larger or equal to specified value.')

    cdef int64_t _shift_index(self, int64_t index) except -1:
        cdef int64_t size = len(self)
        if index >= size or index < -size:
            raise IndexError('Index out of bound')
        if index < 0:
            return (index + size)
        else:
            return index

    cdef uint32_t _get_elt(self, int64_t index) except? 0:
        cdef uint64_t s_index = self._shift_index(index)
        cdef uint32_t elt
        cdef bool valid = croaring.roaring_bitmap_select(self._c_bitmap, s_index, &elt)
        if not valid:
            raise ValueError('Invalid rank')
        return elt

    cdef _get_slice(self, sl):
        """For a faster computation, four different methods, depending on the slice."""
        start, stop, step = sl.indices(len(self))
        sign = 1 if step > 0 else -1
        if (sign > 0 and start >= stop) or (sign < 0 and start <= stop):
            return self.__class__()
        r = range(start, stop, step)
        assert len(r) > 0
        if abs(step) == 1:
            first_elt = self._get_elt(start)
            last_elt  = self._get_elt(stop-sign)
            values = range(first_elt, last_elt+sign, step)
            result = self & self.__class__(values, copy_on_write=self.copy_on_write)
            return result
        elif len(r) < len(self) / 100: # TODO find a good threshold for performance?
            if step < 0:
                start = r[-1]
                stop = r[0] + 1
                step = -step
            else:
                start = r[0]
                stop = r[-1] + 1
            return self._generic_get_slice(start, stop, step)
        else:
            return self.__class__(self.to_array()[sl]) # could be more efficient...

    cdef _generic_get_slice(self, uint32_t start, uint32_t stop, uint32_t step):
        """Assume that start, stop and step > 0 and that the result will not be empty."""
        cdef croaring.roaring_bitmap_t *result = croaring.roaring_bitmap_create()
        cdef croaring.roaring_uint32_iterator_t *iterator = croaring.roaring_create_iterator(self._c_bitmap)
        cdef uint32_t  count, max_count=256
        cdef uint32_t *buff = <uint32_t*>malloc(max_count*4)
        cdef uint32_t i_loc=0, i_glob=start, i_buff=0
        croaring.roaring_bitmap_set_copy_on_write(result, self.copy_on_write)
        first_elt = self._get_elt(start)
        valid = croaring.roaring_move_uint32_iterator_equalorlarger(iterator, first_elt)
        assert valid
        while True:
            count = croaring.roaring_read_uint32_iterator(iterator, buff, max_count)
            while i_buff < max_count and i_glob < stop:
                buff[i_loc] = buff[i_buff]
                i_loc += 1
                i_buff += step
                i_glob += step
            croaring.roaring_bitmap_add_many(result, i_loc, buff) 
            if count != max_count or i_glob >= stop:
                break
            i_loc = 0
            i_buff = i_buff % max_count
        croaring.roaring_free_uint32_iterator(iterator)
        free(buff)
        return self.from_ptr(result)

    def __getitem__(self, value):
        if isinstance(value, int):
            return self._get_elt(value)
        elif isinstance(value, slice):
            return self._get_slice(value)
        else:
            return TypeError('Indices must be integers or slices, not %s' % type(value))

    def serialize(self):
        """
        Return the serialization of the bitmap. See AbstractBitMap.deserialize for the reverse operation.

        >>> BitMap.deserialize(BitMap([3, 12]).serialize())
        BitMap([3, 12])
        """
        cdef size_t size = croaring.roaring_bitmap_portable_size_in_bytes(self._c_bitmap)
        cdef char *buff = <char*>malloc(size)
        cdef real_size = croaring.roaring_bitmap_portable_serialize(self._c_bitmap, buff)
        result = buff[:size]
        free(buff)
        return result


    @classmethod
    def deserialize(cls, char *buff):
        """
        Generate a bitmap from the given serialization. See AbstractBitMap.serialize for the reverse operation.

        >>> BitMap.deserialize(BitMap([3, 12]).serialize())
        BitMap([3, 12])
        """
        return (<AbstractBitMap>cls()).from_ptr(deserialize_ptr(buff)) # FIXME to change when from_ptr is a classmethod

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, state):
        try:                                            # compatibility between Python2 and Python3 (see #27)
            self._c_bitmap = deserialize_ptr(state)
        except TypeError:
            self._c_bitmap = deserialize_ptr(state.encode())


    def __sizeof__(self):
        cdef size_t size = croaring.roaring_bitmap_portable_size_in_bytes(self._c_bitmap)
        return size


    def to_array(self):
        """
        Return an array.array containing the elements of the bitmap, in increasing order.

        It is equivalent to array.array('I', self), but more efficient.

        >>> BitMap([3, 12]).to_array()
        array('I', [3, 12])
        """
        cdef int64_t size = len(self)
        if size == 0:
            return array.array('I', [])
        cdef array.array result = array.array('I')
        array.resize(result, size)
        cdef unsigned[:] buff = result
        croaring.roaring_bitmap_to_uint32_array(self._c_bitmap, &buff[0])
        return result
