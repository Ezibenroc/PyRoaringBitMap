import ctypes
import time
libroaring = ctypes.CDLL('libroaring.so')

def load(fp):
    buff = fp.read()
    return BitMap.deserialize(buff)

def dump(fp, bitmap):
    buff = bitmap.serialize()
    fp.write(buff)

class BitMap:
    __BASE_TYPE__ = ctypes.c_uint32
    def __init__(self, values=None, *, obj=None):
        if obj is not None:
            self.__obj__ = obj
            return
        if values is None:
            self.__obj__ = libroaring.roaring_bitmap_create()
        elif isinstance(values, BitMap):
            self.__obj__ = libroaring.roaring_bitmap_copy(values.__obj__)
        elif isinstance(values, range):
            if values.step < 0:
                values = range(values.stop+1, values.start+1, -values.step)
            if values.start >= values.stop:
                raise ValueError('Invalid range: max value must be greater than min value.')
            self.check_value(values.start)
            self.check_value(values.stop)
            self.__obj__ = libroaring.roaring_bitmap_from_range(values.start, values.stop, values.step)
        else:
            self.check_values(values)
            size = len(values)
            Array = self.__BASE_TYPE__*size
            values = Array(*values)
            self.__obj__ = libroaring.roaring_bitmap_of_ptr(size, values)

    def __del__(self):
        try:
            libroaring.roaring_bitmap_free(self.__obj__)
        except AttributeError: # happens if there is an excepion in __init__ before the creation of __obj__
            pass

    @staticmethod
    def check_value(value):
        if not isinstance(value, int) or value < 0 or value > 4294967295:
            raise ValueError('Value %s is not an uint32.' % value)

    @staticmethod
    def check_values(values):
        for value in values:
            BitMap.check_value(value)

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
        return libroaring.roaring_bitmap_get_cardinality(self.__obj__)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return bool(libroaring.roaring_bitmap_equals(self.__obj__, other.__obj__))

    def __iter__(self):
        size = len(self)
        Array = self.__BASE_TYPE__*size
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

    def __getitem__(self, value):
        self.check_value(value)
        elt = ctypes.pointer(self.__BASE_TYPE__(-1))
        valid = libroaring.roaring_bitmap_select(self.__obj__, value, elt)
        if not valid:
            raise ValueError('Invalid rank.')
        return elt.contents.value

    @classmethod
    def or_many(cls, bitmaps):
        size = len(bitmaps)
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
