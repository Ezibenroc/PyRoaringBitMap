cdef class FrozenBitMap(AbstractBitMap):

    def __ior__(self, other):
        return self.__or__(other)

    def __iand__(self, other):
        return self.__and__(other)

    def __ixor__(self, other):
        return self.__xor__(other)

    def __isub__(self, other):
        return self.__sub__(other)
