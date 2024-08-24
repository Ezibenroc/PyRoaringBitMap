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
    ctypedef struct roaring64_statistics_t:
        uint64_t n_containers
        uint64_t n_array_containers
        uint64_t n_run_containers
        uint64_t n_bitset_containers
        uint64_t n_values_array_containers
        uint64_t n_values_run_containers
        uint64_t n_values_bitset_containers
        uint64_t n_bytes_array_containers
        uint64_t n_bytes_run_containers
        uint64_t n_bytes_bitset_containers
        uint64_t max_value
        uint64_t min_value
        uint64_t cardinality

    roaring_bitmap_t *roaring_bitmap_create()
    bool roaring_bitmap_get_copy_on_write(const roaring_bitmap_t* r)
    void roaring_bitmap_set_copy_on_write(roaring_bitmap_t* r, bool cow)
    void roaring_bitmap_add(roaring_bitmap_t *r, uint32_t x)
    bool roaring_bitmap_add_checked(roaring_bitmap_t *r, uint32_t x)
    void roaring_bitmap_add_many(roaring_bitmap_t *r, size_t n_args, const uint32_t *vals)
    void roaring_bitmap_add_range(roaring_bitmap_t *ra, uint64_t min, uint64_t max);
    void roaring_bitmap_remove(roaring_bitmap_t *r, uint32_t x)
    void roaring_bitmap_remove_range(roaring_bitmap_t *ra, uint64_t min, uint64_t max)
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
    roaring_uint32_iterator_t *roaring_iterator_create(const roaring_bitmap_t *ra)
    bool roaring_uint32_iterator_advance(roaring_uint32_iterator_t *it)
    uint32_t roaring_uint32_iterator_read(roaring_uint32_iterator_t *it, uint32_t* buf, uint32_t count)
    bool roaring_uint32_iterator_move_equalorlarger(roaring_uint32_iterator_t *it, uint32_t val)
    void roaring_uint32_iterator_free(roaring_uint32_iterator_t *it)

    # 64-bit roaring bitmaps
    ctypedef struct roaring64_bitmap_t:
        pass

    ctypedef struct roaring64_iterator_t:
        pass

    roaring64_bitmap_t *roaring64_bitmap_create()
    void roaring64_bitmap_free(roaring64_bitmap_t *r)
    roaring64_bitmap_t *roaring64_bitmap_copy(const roaring64_bitmap_t *r)
    roaring64_bitmap_t *roaring64_bitmap_of_ptr(size_t n_args, const uint64_t *vals)
    roaring64_bitmap_t *roaring64_bitmap_from_range(uint64_t min, uint64_t max, uint64_t step)
    void roaring64_bitmap_add(roaring64_bitmap_t *r, uint64_t val)
    bool roaring64_bitmap_add_checked(roaring64_bitmap_t *r, uint64_t val)
    void roaring64_bitmap_add_many(roaring64_bitmap_t *r, size_t n_args, const uint64_t *vals)
    void roaring64_bitmap_add_range(roaring64_bitmap_t *r, uint64_t min, uint64_t max)
    void roaring64_bitmap_remove(roaring64_bitmap_t *r, uint64_t val)
    bool roaring64_bitmap_remove_checked(roaring64_bitmap_t *r, uint64_t val)
    void roaring64_bitmap_remove_many(roaring64_bitmap_t *r, size_t n_args, const uint64_t *vals)
    void roaring64_bitmap_remove_range(roaring64_bitmap_t *r, uint64_t min, uint64_t max)
    bool roaring64_bitmap_contains(const roaring64_bitmap_t *r, uint64_t val)
    bool roaring64_bitmap_contains_range(const roaring64_bitmap_t *r, uint64_t min, uint64_t max)
    bool roaring64_bitmap_select(const roaring64_bitmap_t *r, uint64_t rank, uint64_t *element)
    void roaring64_bitmap_statistics(const roaring64_bitmap_t *r, roaring64_statistics_t *stat)
    uint64_t roaring64_bitmap_rank(const roaring64_bitmap_t *r, uint64_t val)
    roaring64_bitmap_t *roaring64_bitmap_flip(const roaring64_bitmap_t *r, uint64_t min, uint64_t max)
    void roaring64_bitmap_flip_inplace(roaring64_bitmap_t *r, uint64_t min, uint64_t max)
    bool roaring64_bitmap_get_index(const roaring64_bitmap_t *r, uint64_t val, uint64_t *out_index)
    uint64_t roaring64_bitmap_get_cardinality(const roaring64_bitmap_t *r)
    uint64_t roaring64_bitmap_range_cardinality(const roaring64_bitmap_t *r, uint64_t min, uint64_t max)
    bool roaring64_bitmap_is_empty(const roaring64_bitmap_t *r)
    uint64_t roaring64_bitmap_minimum(const roaring64_bitmap_t *r)
    uint64_t roaring64_bitmap_maximum(const roaring64_bitmap_t *r)
    bool roaring64_bitmap_run_optimize(roaring64_bitmap_t *r)
    size_t roaring64_bitmap_size_in_bytes(const roaring64_bitmap_t *r)
    bool roaring64_bitmap_equals(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    bool roaring64_bitmap_is_subset(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    bool roaring64_bitmap_is_strict_subset(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    void roaring64_bitmap_to_uint64_array(const roaring64_bitmap_t *r, uint64_t *out)
    roaring64_bitmap_t *roaring64_bitmap_and(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    uint64_t roaring64_bitmap_and_cardinality(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    void roaring64_bitmap_and_inplace(roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    bool roaring64_bitmap_intersect(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    double roaring64_bitmap_jaccard_index(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    roaring64_bitmap_t *roaring64_bitmap_or(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    uint64_t roaring64_bitmap_or_cardinality(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    void roaring64_bitmap_or_inplace(roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    roaring64_bitmap_t *roaring64_bitmap_xor(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    uint64_t roaring64_bitmap_xor_cardinality(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    void roaring64_bitmap_xor_inplace(roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    roaring64_bitmap_t *roaring64_bitmap_andnot(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    uint64_t roaring64_bitmap_andnot_cardinality(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    void roaring64_bitmap_andnot_inplace(roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2)
    size_t roaring64_bitmap_portable_size_in_bytes(const roaring64_bitmap_t *r)
    size_t roaring64_bitmap_portable_serialize(const roaring64_bitmap_t *r, char *buf)
    size_t roaring64_bitmap_portable_deserialize_size(const char *buf, size_t maxbytes)
    roaring64_bitmap_t *roaring64_bitmap_portable_deserialize_safe(const char *buf, size_t maxbytes)
    roaring64_iterator_t *roaring64_iterator_create(const roaring64_bitmap_t *r)
    void roaring64_iterator_free(roaring64_iterator_t *it)
    bool roaring64_iterator_has_value(const roaring64_iterator_t *it)
    bool roaring64_iterator_advance(roaring64_iterator_t *it)
    uint64_t roaring64_iterator_value(const roaring64_iterator_t *it)
    bool roaring64_iterator_move_equalorlarger(roaring64_iterator_t *it, uint64_t val)
    uint64_t roaring64_iterator_read(roaring64_iterator_t *it, uint64_t *buf, uint64_t count)