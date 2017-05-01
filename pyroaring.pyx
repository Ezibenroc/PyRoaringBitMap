cimport croaring
from libc.stdint cimport uint32_t, uint64_t, int64_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdlib cimport free, malloc

try:
    range = xrange
except NameError: # python 3
    pass

cdef BitMap create_from_ptr(croaring.roaring_bitmap_t *r):
    bm = <BitMap>BitMap.__new__(BitMap)
    bm._c_bitmap = r
    return bm

cdef BitMap binary_operation(BitMap left, BitMap right, (croaring.roaring_bitmap_t*)func(const croaring.roaring_bitmap_t*, const croaring.roaring_bitmap_t*)):
    cdef croaring.roaring_bitmap_t *r = func(left._c_bitmap, right._c_bitmap)
    return create_from_ptr(r)

cdef binary_or(BitMap left, BitMap right):
    return binary_operation(left, right, croaring.roaring_bitmap_or)

cdef binary_and(BitMap left, BitMap right):
    return binary_operation(left, right, croaring.roaring_bitmap_and)

cdef binary_xor(BitMap left, BitMap right):
    return binary_operation(left, right, croaring.roaring_bitmap_xor)

cdef binary_sub(BitMap left, BitMap right):
    return binary_operation(left, right, croaring.roaring_bitmap_andnot)

cdef binary_ior(BitMap left, BitMap right):
    croaring.roaring_bitmap_or_inplace(left._c_bitmap, right._c_bitmap)
    return left

cdef binary_iand(BitMap left, BitMap right):
    croaring.roaring_bitmap_and_inplace(left._c_bitmap, right._c_bitmap)
    return left

cdef binary_ixor(BitMap left, BitMap right):
    croaring.roaring_bitmap_xor_inplace(left._c_bitmap, right._c_bitmap)
    return left

cdef binary_isub(BitMap left, BitMap right):
    croaring.roaring_bitmap_andnot_inplace(left._c_bitmap, right._c_bitmap)
    return left

cdef croaring.roaring_bitmap_t *deserialize_ptr(char *buff):
    cdef croaring.roaring_bitmap_t *ptr
    ptr = croaring.roaring_bitmap_portable_deserialize(buff)
    return ptr

cdef class BitMap:
    cdef croaring.roaring_bitmap_t* _c_bitmap

    def __cinit__(self, values=None, copy_on_write=False):
        """ Construct a BitMap object. If a list of integers is provided, the integers are truncated down to the least significant 32 bits"""
        if values is None:
            self._c_bitmap = croaring.roaring_bitmap_create()
        elif isinstance(values, BitMap):
            self._c_bitmap = croaring.roaring_bitmap_copy((<BitMap?>values)._c_bitmap)
        elif isinstance(values, range):
            _, (start, stop, step) = values.__reduce__()
            if step < 0:
                values = range(min(values), max(values)+1, -step)
                _, (start, stop, step) = values.__reduce__()
            if start >= stop:
                self._c_bitmap = croaring.roaring_bitmap_create()
            else:
                self._c_bitmap = croaring.roaring_bitmap_from_range(start, stop, step)
        else:
            self._c_bitmap = croaring.roaring_bitmap_create()
            self.update(values)
        if not isinstance(values, BitMap):
            self._c_bitmap.copy_on_write = copy_on_write

    @property
    def copy_on_write(self):
        return self._c_bitmap.copy_on_write

    def __dealloc__(self):
        if self._c_bitmap is not NULL:
            croaring.roaring_bitmap_free(self._c_bitmap)

    def check_compatibility(self, BitMap other):
        if self._c_bitmap.copy_on_write != other._c_bitmap.copy_on_write:
            raise ValueError('Cannot have interactions between bitmaps with and without copy_on_write.\n')
        pass

    def add(self, uint32_t value):
        croaring.roaring_bitmap_add(self._c_bitmap, value)

    def update(self, values): # FIXME: could be more efficient, using roaring_bitmap_add_many
        cdef vector[uint32_t] buff = values
        croaring.roaring_bitmap_add_many(self._c_bitmap, len(values), &buff[0])

    def remove(self, uint32_t value):
        croaring.roaring_bitmap_remove(self._c_bitmap, value)

    def __contains__(self, uint32_t value):
        return croaring.roaring_bitmap_contains(self._c_bitmap, value)

    def __bool__(self):
        return not croaring.roaring_bitmap_is_empty(self._c_bitmap)

    def __len__(self):
        return croaring.roaring_bitmap_get_cardinality(self._c_bitmap)

    def __richcmp__(self, other, int op):
        self.check_compatibility(other)
        if op == 0: # <
            return croaring.roaring_bitmap_is_strict_subset((<BitMap?>self)._c_bitmap, (<BitMap?>other)._c_bitmap)
        elif op == 1: # <=
            return croaring.roaring_bitmap_is_subset((<BitMap?>self)._c_bitmap, (<BitMap?>other)._c_bitmap)
        elif op == 2: # ==
            return croaring.roaring_bitmap_equals((<BitMap?>self)._c_bitmap, (<BitMap?>other)._c_bitmap)
        elif op == 3: # !=
            return not (self == other)
        elif op == 4: # >
            return croaring.roaring_bitmap_is_strict_subset((<BitMap?>other)._c_bitmap, (<BitMap?>self)._c_bitmap)
        else:         # >=
            assert op == 5
            return croaring.roaring_bitmap_is_subset((<BitMap?>other)._c_bitmap, (<BitMap?>self)._c_bitmap)

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
        values = ', '.join([str(n) for n in self])
        return 'BitMap([%s])' % values

    def flip(self, uint64_t start, uint64_t end):
        return create_from_ptr(croaring.roaring_bitmap_flip(self._c_bitmap, start, end))

    def flip_inplace(self, uint64_t start, uint64_t end):
        croaring.roaring_bitmap_flip_inplace(self._c_bitmap, start, end)

    @classmethod
    def union(cls, *bitmaps):
        size = len(bitmaps)
        cdef croaring.roaring_bitmap_t *result
        cdef BitMap bm
        cdef vector[const croaring.roaring_bitmap_t*] buff
        if size <= 1:
            return cls(*bitmaps)
        elif size == 2:
            return bitmaps[0] | bitmaps[1]
        else:
            for bm in bitmaps:
                bitmaps[0].check_compatibility(bm)
                buff.push_back(bm._c_bitmap)
            result = croaring.roaring_bitmap_or_many(size, &buff[0])
            return create_from_ptr(result)

    def __or__(self, other):
        self.check_compatibility(other)
        return binary_or(self, <BitMap?>other)

    def __ior__(self, other):
        self.check_compatibility(other)
        return binary_ior(self, <BitMap?>other)

    def __and__(self, other):
        self.check_compatibility(other)
        return binary_and(self, <BitMap?>other)

    def __iand__(self, other):
        self.check_compatibility(other)
        return binary_iand(self, <BitMap?>other)

    def __xor__(self, other):
        self.check_compatibility(other)
        return binary_xor(self, <BitMap?>other)

    def __ixor__(self, other):
        self.check_compatibility(other)
        return binary_ixor(self, <BitMap?>other)

    def __sub__(self, other):
        self.check_compatibility(other)
        return binary_sub(self, <BitMap?>other)

    def __isub__(self, other):
        self.check_compatibility(other)
        return binary_isub(self, <BitMap?>other)

    def union_cardinality(self, BitMap other):
        self.check_compatibility(other)
        return croaring.roaring_bitmap_or_cardinality(self._c_bitmap, other._c_bitmap)

    def intersection_cardinality(self, BitMap other):
        self.check_compatibility(other)
        return croaring.roaring_bitmap_and_cardinality(self._c_bitmap, other._c_bitmap)

    def difference_cardinality(self, BitMap other):
        self.check_compatibility(other)
        return croaring.roaring_bitmap_andnot_cardinality(self._c_bitmap, other._c_bitmap)

    def symmetric_difference_cardinality(self, BitMap other):
        self.check_compatibility(other)
        return croaring.roaring_bitmap_xor_cardinality(self._c_bitmap, other._c_bitmap)

    def intersect(self, BitMap other):
        self.check_compatibility(other)
        return croaring.roaring_bitmap_intersect(self._c_bitmap, other._c_bitmap)

    def jaccard_index(self, BitMap other):
        self.check_compatibility(other)
        return croaring.roaring_bitmap_jaccard_index(self._c_bitmap, other._c_bitmap)

    def get_statistics(self):
        cdef croaring.roaring_statistics_t stat
        croaring.roaring_bitmap_statistics(self._c_bitmap, &stat)
        return stat

    def min(self):
        if len(self) == 0:
            raise ValueError('Empty roaring bitmap, there is no minimum.')
        else:
            return croaring.roaring_bitmap_minimum(self._c_bitmap)

    def max(self):
        if len(self) == 0:
            raise ValueError('Empty roaring bitmap, there is no maximum.')
        else:
            return croaring.roaring_bitmap_maximum(self._c_bitmap)

    def rank(self, uint32_t value):
        return croaring.roaring_bitmap_rank(self._c_bitmap, value)

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
        start, stop, step = sl.indices(len(self))
        sign = 1 if step > 0 else -1
        if (sign > 0 and start >= stop) or (sign < 0 and start <= stop):
            return self.__class__()
        if abs(step) == 1:
            first_elt = self._get_elt(start)
            last_elt  = self._get_elt(stop-sign)
            values = range(first_elt, last_elt+sign, step)
            result = self.__class__(values, copy_on_write=self.copy_on_write)
            result &= self
            return result
        else:
            return self.__class__(list(self)[sl]) # very inefficient...

    def __getitem__(self, value):
        if isinstance(value, int):
            return self._get_elt(value)
        elif isinstance(value, slice):
            return self._get_slice(value)
        else:
            return TypeError('Indices must be integers or slices, not %s' % type(value))

    def serialize(self):
        cdef size_t size = croaring.roaring_bitmap_portable_size_in_bytes(self._c_bitmap)
        cdef char *buff = <char*>malloc(size)
        cdef real_size = croaring.roaring_bitmap_portable_serialize(self._c_bitmap, buff)
        result = buff[:size]
        free(buff)
        return result


    @classmethod
    def deserialize(cls, char *buff):
        return create_from_ptr(deserialize_ptr(buff))

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, state):
        self._c_bitmap = deserialize_ptr(state)
