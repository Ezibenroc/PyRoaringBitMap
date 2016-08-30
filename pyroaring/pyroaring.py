import sys
import array
from .types_declarations import *

is_python2 = sys.version_info < (3, 0)
if is_python2:
    sys.stderr.write('Warning: %s is faster with Python 3.\n' % __name__)

def load(fp):
    buff = fp.read()
    return BitMap.deserialize(buff)

def dump(fp, bitmap):
    buff = bitmap.serialize()
    fp.write(buff)

class BitMap:

    def __init__(self, values=None, obj=None):
        """ Construct a BitMap object. If a list of integers is provided, the integers are truncated down to the least significant 32 bits"""
        if obj is not None and values is None:
            self.__obj__ = obj
            return
        if values is None:
            self.__obj__ = libroaring.roaring_bitmap_create()
        elif isinstance(values, BitMap):
            self.__obj__ = libroaring.roaring_bitmap_copy(values.__obj__)
        elif not is_python2 and isinstance(values, range):
            if values.step < 0:
                values = range(values.stop+1, values.start+1, -values.step)
            if values.start >= values.stop:
                raise ValueError('Invalid range: max value must be greater than min value.')
            self.check_value(values.start)
            self.check_value(values.stop)
            self.__obj__ = libroaring.roaring_bitmap_from_range(values.start, values.stop, values.step)
        elif is_python2 and isinstance(values, xrange): # cannot access the fields start, stop and step in a xrange object
            if len(values) == 0:
                raise ValueError('Invalid xrange: max value must be greater than min value.')
            elif len(values) == 1 or values[1]-values[0] != 1:
                self.__init__(list(values))
            else:
                self.check_value(values[0])
                self.check_value(values[-1])
                self.__obj__ = libroaring.roaring_bitmap_from_range(values[0], values[-1]+1, 1)
        elif isinstance(values, array.array) and values.typecode == 'I':
            assert values.itemsize == 4 # on some hardware an "unsigned int" could have sizeof diff. from 4, but uncommon
            addr, count = values.buffer_info()
            p = ctypes.cast(addr,ctypes.POINTER(ctypes.c_uint32))
            self.__obj__ = libroaring.roaring_bitmap_of_ptr(count, p)
            libroaring.roaring_bitmap_run_optimize(self.__obj__)
        else:
            v = array.array('I',values)
            assert v.itemsize == 4 # on some hardware an "unsigned int" could have sizeof diff. from 4, but uncommon
            addr, count = v.buffer_info()
            p = ctypes.cast(addr,ctypes.POINTER(ctypes.c_uint32))
            self.__obj__ = libroaring.roaring_bitmap_of_ptr(count, p)
            libroaring.roaring_bitmap_run_optimize(self.__obj__)

    def __del__(self):
        try:
            libroaring.roaring_bitmap_free(self.__obj__)
        except AttributeError: # happens if there is an excepion in __init__ before the creation of __obj__
            pass

    @staticmethod
    def check_value(value):
        if not isinstance(value, int) or value < 0 or value >= 2**32:
            raise ValueError('Value %r is not an uint32.' % value)

    def add(self, value):
        self.check_value(value)
        libroaring.roaring_bitmap_add(self.__obj__, value)

    def remove(self, value):
        self.check_value(value)
        libroaring.roaring_bitmap_remove(self.__obj__, value)

    def __contains__(self, value):
        self.check_value(value)
        return libroaring.roaring_bitmap_contains(self.__obj__, value)

    def __len__(self):
        return int(libroaring.roaring_bitmap_get_cardinality(self.__obj__))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return bool(libroaring.roaring_bitmap_equals(self.__obj__, other.__obj__))

    def __ne__(self, other):
        return not(self == other)

    def __iter__(self):
        size = len(self)
        Array = val_type*size
        buff = Array()
        libroaring.roaring_bitmap_to_uint32_array(self.__obj__, buff)
        for i in range(size):
            yield int(buff[i])

    def __repr__(self):
        values = ', '.join([str(n) for n in self])
        return 'BitMap([%s])' % values

    def __binary_op__(self, other, function):
        try:
            return BitMap(obj=function(self.__obj__, other.__obj__))
        except AttributeError:
            raise TypeError('Not a BitMap.')

    def __or__(self, other):
        return self.__binary_op__(other, libroaring.roaring_bitmap_or)

    def __and__(self, other):
        return self.__binary_op__(other, libroaring.roaring_bitmap_and)

    def __xor__(self, other):
        return self.__binary_op__(other, libroaring.roaring_bitmap_xor)

    def __sub__(self, other):
        return self.__binary_op__(other, libroaring.roaring_bitmap_andnot)

    def __binary_op_inplace__(self, other, function):
        try:
            function(self.__obj__, other.__obj__)
            return self
        except AttributeError:
            raise TypeError('Not a BitMap.')

    def __ior__(self, other):
        return self.__binary_op_inplace__(other, libroaring.roaring_bitmap_or_inplace)

    def __iand__(self, other):
        return self.__binary_op_inplace__(other, libroaring.roaring_bitmap_and_inplace)

    def __ixor__(self, other):
        return self.__binary_op_inplace__(other, libroaring.roaring_bitmap_xor_inplace)

    def __isub__(self, other):
        return self.__binary_op_inplace__(other, libroaring.roaring_bitmap_andnot_inplace)

    def __getitem__(self, value):
        self.check_value(value)
        elt = ctypes.pointer(val_type(-1))
        valid = libroaring.roaring_bitmap_select(self.__obj__, value, elt)
        if not valid:
            raise ValueError('Invalid rank.')
        return int(elt.contents.value)

    def __le__(self, other):
        return self&other == self

    def __ge__(self, other):
        return self&other == other

    def __lt__(self, other):
        return self&other == self and self != other

    def __gt__(self, other):
        return self&other == other and self != other

    @classmethod
    def union(cls, *bitmaps):
        size = len(bitmaps)
        if size <= 1:
            return cls(*bitmaps)
        elif size == 2:
            return bitmaps[0] | bitmaps[1]
        else:
            Array = ctypes.c_void_p*size
            bitmaps = Array(*(ctypes.c_void_p(bm.__obj__) for bm in bitmaps))
            obj = libroaring.roaring_bitmap_or_many(size, bitmaps)
            return cls(obj=obj)

    def serialize(self):
        size = libroaring.roaring_bitmap_portable_size_in_bytes(self.__obj__)
        Array = ctypes.c_char*size
        buff = Array()
        size = libroaring.roaring_bitmap_portable_serialize(self.__obj__, buff)
        return buff[:size]

    @classmethod
    def deserialize(cls, buff):
        size = len(buff)
        Array = ctypes.c_char*size
        buff = Array(*buff)
        obj = libroaring.roaring_bitmap_portable_deserialize(buff)
        return cls(obj=obj)

    def get_statistics(self):
        stats = ctypes.pointer(BitMapStats())
        libroaring.roaring_bitmap_statistics(self.__obj__, stats)
        return stats.contents

    def flip(self, start, end):
        self.check_value(start)
        self.check_value(end)
        return BitMap(obj=libroaring.roaring_bitmap_flip(self.__obj__, start, end))

    def flip_inplace(self, start, end):
        self.check_value(start)
        self.check_value(end)
        libroaring.roaring_bitmap_flip_inplace(self.__obj__, start, end)
