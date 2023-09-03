cdef class BitMap(AbstractBitMap):

    cdef compute_hash(self):
        '''Unsupported method.'''
        # For some reason, if we directly override __hash__ (either in BitMap or in FrozenBitMap), the __richcmp__
        # method disappears.
        raise TypeError('Cannot compute the hash of a %s.' % self.__class__.__name__)

    def add(self, uint32_t value):
        """
        Add an element to the bitmap. This has no effect if the element is already present.

        >>> bm = BitMap()
        >>> bm.add(42)
        >>> bm
        BitMap([42])
        >>> bm.add(42)
        >>> bm
        BitMap([42])
        """
        croaring.roaring_bitmap_add(self._c_bitmap, value)

    def add_checked(self, uint32_t value):
        """
        Add an element to the bitmap. This raises a KeyError exception if the element is already present.

        >>> bm = BitMap()
        >>> bm.add_checked(42)
        >>> bm
        BitMap([42])
        >>> bm.add_checked(42)
        Traceback (most recent call last):
        ...
        KeyError: 42
        """
        cdef bool test = croaring.roaring_bitmap_add_checked(self._c_bitmap, value)
        if not test:
            raise KeyError(value)

    def update(self, *all_values): # FIXME could be more efficient
        """
        Add all the given values to the bitmap.

        >>> bm = BitMap([3, 12])
        >>> bm.update([8, 12, 55, 18])
        >>> bm
        BitMap([3, 8, 12, 18, 55])
        """
        cdef vector[uint32_t] buff_vect
        cdef unsigned[:] buff
        for values in all_values:
            if isinstance(values, AbstractBitMap):
                self |= values
            elif isinstance(values, range):
                if len(values) == 0:
                    continue
                _, (start, stop, step) = values.__reduce__()
                if step == -1:
                    step = 1
                    start, stop = stop+1, start+1
                if step == 1:
                    self.add_range(start, stop)
                else:
                    self |= AbstractBitMap(values, copy_on_write=self.copy_on_write)
            elif isinstance(values, array.array) and len(values) > 0:
                buff = <array.array> values
                croaring.roaring_bitmap_add_many(self._c_bitmap, len(values), &buff[0])
            else:
                try:
                    size = len(values)
                except TypeError:  # object has no length, creating a list
                    values = list(values)
                    size = len(values)
                if size > 0:
                    buff_vect = values
                    croaring.roaring_bitmap_add_many(self._c_bitmap, size, &buff_vect[0])

    def discard(self, uint32_t value):
        """
        Remove an element from the bitmap. This has no effect if the element is not present.

        >>> bm = BitMap([3, 12])
        >>> bm.discard(3)
        >>> bm
        BitMap([12])
        >>> bm.discard(3)
        >>> bm
        BitMap([12])
        """
        croaring.roaring_bitmap_remove(self._c_bitmap, value)

    def remove(self, uint32_t value):
        """
        Remove an element from the bitmap. This raises a KeyError exception if the element does not exist in the bitmap.

        >>> bm = BitMap([3, 12])
        >>> bm.remove(3)
        >>> bm
        BitMap([12])
        >>> bm.remove(3)
        Traceback (most recent call last):
        ...
        KeyError: 3
        """
        cdef bool test = croaring.roaring_bitmap_remove_checked(self._c_bitmap, value)
        if not test:
            raise KeyError(value)

    cdef binary_iop(self, AbstractBitMap other, (void)func(croaring.roaring_bitmap_t*, const croaring.roaring_bitmap_t*) noexcept) noexcept:
        func(self._c_bitmap, other._c_bitmap)
        return self

    def __ior__(self, other):
        self._check_compatibility(other)
        return (<BitMap>self).binary_iop(<AbstractBitMap?>other, croaring.roaring_bitmap_or_inplace)

    def __iand__(self, other):
        self._check_compatibility(other)
        return (<BitMap>self).binary_iop(<AbstractBitMap?>other, croaring.roaring_bitmap_and_inplace)

    def __ixor__(self, other):
        self._check_compatibility(other)
        return (<BitMap>self).binary_iop(<AbstractBitMap?>other, croaring.roaring_bitmap_xor_inplace)

    def __isub__(self, other):
        self._check_compatibility(other)
        return (<BitMap>self).binary_iop(<AbstractBitMap?>other, croaring.roaring_bitmap_andnot_inplace)

    def intersection_update(self, *all_values): # FIXME could be more efficient
        """
        Update the bitmap by taking its intersection with the given values.

        >>> bm = BitMap([3, 12])
        >>> bm.intersection_update([8, 12, 55, 18])
        >>> bm
        BitMap([12])
        """
        cdef uint32_t elt
        for values in all_values:
            if isinstance(values, AbstractBitMap):
                self &= values
            else:
                self &= AbstractBitMap(values, copy_on_write=self.copy_on_write)

    def difference_update(self, *others):
        """
        Remove all elements of another set from this set.

        >>> bm = BitMap([1, 2, 3, 4, 5])
        >>> bm.difference_update(BitMap([1, 2, 10]), BitMap([3, 4, 20]))
        >>> bm
        BitMap([5])
        """
        self.__isub__(AbstractBitMap.union(*others))

    def symmetric_difference_update(self, other):
        """
        Update a set with the symmetric difference of itself and another.

        >>> bm = BitMap([1, 2, 3, 4])
        >>> bm.symmetric_difference_update(BitMap([1, 2, 10]))
        >>> bm
        BitMap([3, 4, 10])

        """
        self.__ixor__(other)
        
    def overwrite(self, AbstractBitMap other):
        """
        Clear the bitmap and overwrite it with another.

        >>> bm = BitMap([3, 12])
        >>> other = BitMap([4, 14])
        >>> bm.overwrite(other)
        >>> other.remove(4)
        >>> bm
        BitMap([4, 14])
        >>> other
        BitMap([14])
        """
        if self._c_bitmap == other._c_bitmap:
            raise ValueError('Cannot overwrite itself')
        croaring.roaring_bitmap_overwrite(self._c_bitmap, other._c_bitmap)

    def clear(self):
        """
        Remove all elements from this set.

        >>> bm = BitMap([1, 2, 3])
        >>> bm.clear()
        >>> bm
        BitMap([])
        """
        croaring.roaring_bitmap_clear(self._c_bitmap)

    def pop(self):
        """
        Remove and return an arbitrary set element.
        Raises KeyError if the set is empty.

        >>> bm = BitMap([1, 2])
        >>> a = bm.pop()
        >>> b = bm.pop()
        >>> bm
        BitMap([])
        >>> bm.pop()
        Traceback (most recent call last):
        ...
        KeyError: 'pop from an empty BitMap'

        """
        try:
            value = self.min()
        except ValueError:
            raise KeyError('pop from an empty BitMap')
        self.remove(value)
        return value


    def flip_inplace(self, uint64_t start, uint64_t end):
        """
        Compute (in place) the negation of the bitmap within the specified interval.

        Areas outside the range are passed unchanged.

        >>> bm = BitMap([3, 12])
        >>> bm.flip_inplace(10, 15)
        >>> bm
        BitMap([3, 10, 11, 13, 14])
        """
        croaring.roaring_bitmap_flip_inplace(self._c_bitmap, start, end)

    def add_range(self, uint64_t range_start, uint64_t range_end):
        """
        Add a range of values from range_start (included) to range_end (excluded).

        >>> bm = BitMap([5, 7])
        >>> bm.add_range(6, 9)
        >>> bm
        BitMap([5, 6, 7, 8])
        """
        if range_end <= range_start or range_end == 0 or range_start >= 2**32:
            return
        if range_end >= 2**32:
            range_end = 2**32
        croaring.roaring_bitmap_add_range(self._c_bitmap, range_start, range_end)

    def remove_range(self, uint64_t range_start, uint64_t range_end):
        """
        Remove a range of values from range_start (included) to range_end (excluded).

        >>> bm = BitMap([5, 6, 7, 8, 9, 10])
        >>> bm.remove_range(6, 9)
        >>> bm
        BitMap([5, 9, 10])
        """
        if range_end <= range_start or range_end == 0 or range_start >= 2**32:
            return
        if range_end >= 2**32:
            range_end = 2**32
        croaring.roaring_bitmap_remove_range(self._c_bitmap, range_start, range_end)
