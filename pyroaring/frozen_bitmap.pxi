cdef class FrozenBitMap(AbstractBitMap):

    def __ior__(self, other):
        '''Unsupported method.'''
        raise TypeError('Cannot modify a %s.' % self.__class__.__name__)

    def __iand__(self, other):
        '''Unsupported method.'''
        raise TypeError('Cannot modify a %s.' % self.__class__.__name__)

    def __ixor__(self, other):
        '''Unsupported method.'''
        raise TypeError('Cannot modify a %s.' % self.__class__.__name__)

    def __isub__(self, other):
        '''Unsupported method.'''
        raise TypeError('Cannot modify a %s.' % self.__class__.__name__)
