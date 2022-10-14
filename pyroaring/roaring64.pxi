cimport croaring
from libc.stdlib cimport free, malloc
from cython cimport view


cdef class BitMap64:
    cdef croaring.Roaring64Map _roaring64

    @staticmethod
    def deserialize(char [::1] buf):
        cdef BitMap64 m = BitMap64()
        m._roaring64 = croaring.Roaring64Map.readSafe(&buf[0], len(buf))
        return m

    def size_in_bytes(self, portable=True):
        return self._roaring64.getSizeInBytes(portable)

    def serialize(self, portable=True):
        size = self._roaring64.getSizeInBytes(portable)
        buf = bytearray(size)
        cdef real_size = self._roaring64.write(buf, portable)
        return buf[:real_size]

    def add(self, n):
        self._roaring64.add(n)

    def contains(self, n):
        return self._roaring64.contains(n)

    def maximum(self):
        return self._roaring64.maximum()

    def minimum(self):
        return self._roaring64.minimum()

    def __len__(self):
        return self._roaring64.cardinality()
