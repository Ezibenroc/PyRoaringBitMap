from libc.stdint cimport uint8_t, int32_t, uint32_t, uint64_t, int64_t
from libcpp cimport bool

cdef extern from "roaring.h":
    ctypedef struct roaring_array_t:
        pass
    ctypedef struct roaring_bitmap_t:
        roaring_array_t high_low_container
    ctypedef struct roaring_uint32_iterator_t:
        const roaring_bitmap_t *parent
        int32_t container_index
        int32_t in_container_index
        int32_t run_index
        uint32_t in_run_index
        uint32_t current_value
        bool has_value
        const void *container
        uint8_t typecode
        uint32_t highbits
    ctypedef struct roaring_statistics_t:
        uint32_t n_containers
        uint32_t n_array_containers
        uint32_t n_run_containers
        uint32_t n_bitset_containers
        uint32_t n_values_array_containers
        uint32_t n_values_run_containers
        uint32_t n_values_bitset_containers
        uint32_t n_bytes_array_containers
        uint32_t n_bytes_run_containers
        uint32_t n_bytes_bitset_containers
        uint32_t max_value
        uint32_t min_value
        uint64_t sum_value
        uint64_t cardinality

    roaring_bitmap_t *roaring_bitmap_create()
    bool roaring_bitmap_get_copy_on_write(const roaring_bitmap_t* r)
    void roaring_bitmap_set_copy_on_write(roaring_bitmap_t* r, bool cow)
    void roaring_bitmap_add(roaring_bitmap_t *r, uint32_t x)
    bool roaring_bitmap_add_checked(roaring_bitmap_t *r, uint32_t x)
    void roaring_bitmap_add_many(roaring_bitmap_t *r, size_t n_args, const uint32_t *vals)
    void roaring_bitmap_add_range(roaring_bitmap_t *ra, uint64_t min, uint64_t max);
    void roaring_bitmap_remove(roaring_bitmap_t *r, uint32_t x)
    inline void roaring_bitmap_remove_range(roaring_bitmap_t *ra, uint64_t min, uint64_t max)
    bool roaring_bitmap_remove_checked(roaring_bitmap_t *r, uint32_t x)
    void roaring_bitmap_clear(roaring_bitmap_t *r)
    bool roaring_bitmap_contains(const roaring_bitmap_t *r, uint32_t val)
    bool roaring_bitmap_contains_range(const roaring_bitmap_t *r, uint64_t range_start, uint64_t range_end)
    roaring_bitmap_t *roaring_bitmap_copy(const roaring_bitmap_t *r)
    bool roaring_bitmap_overwrite(roaring_bitmap_t *dest, const roaring_bitmap_t *src)
    roaring_bitmap_t *roaring_bitmap_from_range(uint64_t min, uint64_t max, uint32_t step)
    bool roaring_bitmap_run_optimize(roaring_bitmap_t *r)
    size_t roaring_bitmap_shrink_to_fit(roaring_bitmap_t *r)
    void roaring_bitmap_free(roaring_bitmap_t *r)
    roaring_bitmap_t *roaring_bitmap_of_ptr(size_t n_args, const uint32_t *vals)
    uint64_t roaring_bitmap_get_cardinality(const roaring_bitmap_t *r)
    uint64_t roaring_bitmap_range_cardinality(const roaring_bitmap_t *r, uint64_t range_start, uint64_t range_end)
    bool roaring_bitmap_is_empty(const roaring_bitmap_t *ra)
    bool roaring_bitmap_equals(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2)
    bool roaring_bitmap_is_strict_subset(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2)
    bool roaring_bitmap_is_subset(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2)
    void roaring_bitmap_to_uint32_array(const roaring_bitmap_t *r, uint32_t *ans)
    roaring_bitmap_t *roaring_bitmap_or_many(size_t number, const roaring_bitmap_t **x)
    roaring_bitmap_t *roaring_bitmap_or(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    void roaring_bitmap_or_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    roaring_bitmap_t *roaring_bitmap_and(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    void roaring_bitmap_and_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    roaring_bitmap_t *roaring_bitmap_xor(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    void roaring_bitmap_xor_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    roaring_bitmap_t *roaring_bitmap_andnot(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    void roaring_bitmap_andnot_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    uint64_t roaring_bitmap_or_cardinality(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    uint64_t roaring_bitmap_and_cardinality(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    uint64_t roaring_bitmap_andnot_cardinality(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    uint64_t roaring_bitmap_xor_cardinality(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    bool roaring_bitmap_intersect(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    double roaring_bitmap_jaccard_index(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2)
    uint32_t roaring_bitmap_minimum(const roaring_bitmap_t *r)
    uint32_t roaring_bitmap_maximum(const roaring_bitmap_t *r)
    uint64_t roaring_bitmap_rank(const roaring_bitmap_t *r, uint32_t x)
    roaring_bitmap_t *roaring_bitmap_flip(const roaring_bitmap_t *x1, uint64_t range_start, uint64_t range_end)
    void roaring_bitmap_flip_inplace(roaring_bitmap_t *x1, uint64_t range_start, uint64_t range_end)
    roaring_bitmap_t *roaring_bitmap_add_offset(const roaring_bitmap_t *bm, int64_t offset)
    bool roaring_bitmap_select(const roaring_bitmap_t *r, uint32_t rank, uint32_t *element)
    void roaring_bitmap_statistics(const roaring_bitmap_t *r, roaring_statistics_t *stat)
    size_t roaring_bitmap_portable_size_in_bytes(const roaring_bitmap_t *ra)
    size_t roaring_bitmap_portable_serialize(const roaring_bitmap_t *ra, char *buf)
    roaring_bitmap_t *roaring_bitmap_portable_deserialize(const char *buf)
    roaring_uint32_iterator_t *roaring_create_iterator(const roaring_bitmap_t *ra)
    bool roaring_advance_uint32_iterator(roaring_uint32_iterator_t *it)
    uint32_t roaring_read_uint32_iterator(roaring_uint32_iterator_t *it, uint32_t* buf, uint32_t count)
    bool roaring_move_uint32_iterator_equalorlarger(roaring_uint32_iterator_t *it, uint32_t val)
    void roaring_free_uint32_iterator(roaring_uint32_iterator_t *it)
    void print_platform_information()
