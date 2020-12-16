/*
 * Copyright (c) 2020, 2021 The University of Rochester
 *
 * Author: Ruohuang Zheng <rzheng3@ur.rochester.edu>
 *
 * Intended to be licensed under GPL3
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#include <sys/time.h>
#include <sys/resource.h>


#define CACHELINE_SIZE  64


/*
 * Random number
 */
static unsigned int next_rand = 1;

static unsigned int gen_rand()
{
    next_rand = next_rand * 1103515245u + 12345u;
}

static void set_rand_seed(unsigned int seed)
{
    next_rand = seed;
}


/*
 * Memory access
 */
struct chain_idx {
    long next;
    long prev;
    long self;
    char pad[CACHELINE_SIZE - sizeof(long) * 3];
} __attribute__ ((packed, aligned(sizeof(long))));;

static struct chain_idx *pool = NULL;

static inline void swap(struct chain_idx *a, struct chain_idx *b)
{
    if (a->next == b->self || b->next == a->self) {
        
    } else {
        struct chain_idx *a_prev = &pool[a->prev];
        struct chain_idx *a_next = &pool[a->next];
        struct chain_idx *b_prev = &pool[b->prev];
        struct chain_idx *b_next = &pool[b->next];
        
        a_prev->next = b->self;
        a_next->prev = b->self;
        b->prev = a_prev->self;
        b->next = a_next->self;
        
        b_prev->next = a->self;
        b_next->prev = a->self;
        a->prev = b_prev->self;
        a->next = b_next->self;
        
        assert(a->next != a->self);
        assert(b->next != b->self);
    }
}

static void init(long access_count, long pool_count, long step)
{
    long pool_size = pool_count * sizeof(struct chain_idx);
    pool = (void *)malloc(pool_size);
    memset(pool, 0, pool_size);
    
    long prev = 0, cur = 0, next = 2;
    long unique_count = 0;
    while (unique_count < access_count) {
        if (pool[cur].self) {
            //printf("WARN: #%ld override pool[%ld], old: %ld\n", unique_count, cur, pool[cur].self);
        } else {
            unique_count++;
        }
        
        pool[cur].self = cur;
        pool[cur].next = next;
        pool[cur].prev = prev;
        
        prev = cur;
        cur = next;
        next = (next + step + 1) % pool_count;
    }
    
    pool[0].prev = prev;
    pool[prev].next = 0;
}

static double get_process_time()
{
    struct rusage usage;
    if (!getrusage(RUSAGE_SELF, &usage)) {
        return (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / 1.0e6;
    }
    return 0.0;
}

#ifdef __AVX2__

static int probe_avx2_pure(int idx_init, int idx_step, long count, long repeat)
{
    register __m256i idx asm ("%ymm0") =
        _mm256_set_epi32(
            idx_init + idx_step * 0,
            idx_init + idx_step * 1,
            idx_init + idx_step * 2,
            idx_init + idx_step * 3,
            idx_init + idx_step * 4,
            idx_init + idx_step * 5,
            idx_init + idx_step * 6,
            idx_init + idx_step * 7
        );
    register void *base asm ("%rbx") = pool;
    register long exp_count asm ("%rcx") = count * repeat;
    
    __asm__ __volatile (
        "   vpcmpeqd    %%ymm8, %%ymm8, %%ymm8;"
        "1:;"
        "   vmovdqa     %%ymm8, %%ymm2;"
        "   vpslld      $0x3, %%ymm0, %%ymm4;"
        "   vpgatherdd  %%ymm2, (%%rbx, %%ymm4, 8), %%ymm6;"
        "   vmovdqa     %%ymm6, %%ymm0;"
        "   loop        1b;"
        "   vzeroupper;"
        :
        : "x" (idx), "r" (base), "r" (exp_count)
    );
    
    return 0;
}

static int probe_avx2_intrinsic(int idx_init, int idx_step, long count, long repeat)
{
    __m256i idx = _mm256_set_epi32(
            idx_init + idx_step * 0,
            idx_init + idx_step * 1,
            idx_init + idx_step * 2,
            idx_init + idx_step * 3,
            idx_init + idx_step * 4,
            idx_init + idx_step * 5,
            idx_init + idx_step * 6,
            idx_init + idx_step * 7
        );
    const __m256i scale = _mm256_set1_epi32(8);
    
    long exp_count = count * repeat;
    for (long i = 0; i < exp_count; i++) {
        idx = _mm256_mullo_epi32(idx, scale);
        idx = _mm256_i32gather_epi32((void *)pool, idx, 8);
    }
    
    return _mm256_movemask_ps(_mm256_cvtepi32_ps(idx));
}

#endif

#ifdef __AVX512F__

static int probe_avx512_pure(int idx_init, int idx_step, long count, long repeat)
{
    volatile register __m512i idx asm ("%zmm2") =
        _mm512_set_epi32(idx_init + idx_step * 0,
            idx_init + idx_step * 1,
            idx_init + idx_step * 2,
            idx_init + idx_step * 3,
            idx_init + idx_step * 4,
            idx_init + idx_step * 5,
            idx_init + idx_step * 6,
            idx_init + idx_step * 7,
            idx_init + idx_step * 8,
            idx_init + idx_step * 9,
            idx_init + idx_step * 10,
            idx_init + idx_step * 11,
            idx_init + idx_step * 12,
            idx_init + idx_step * 13,
            idx_init + idx_step * 14,
            idx_init + idx_step * 15
        );
    register unsigned int mask asm ("%eax") = 0xffffffff;
    register void *base asm ("%rbx") = pool;
    register long exp_count asm ("%rcx") = count * repeat;
    
    __asm__ __volatile (
        "   kmovw       %%eax, %%k2;"
        "1:;"
        "   kmovw       %%k2, %%k1;"
        "   vpslld      $0x3, %%zmm2, %%zmm0;"
        //"   vpgatherdd  (%%rbx, %%zmm0, 8), %%zmm1{{%%k1}};"
        "   .byte 0x62, 0xf2, 0x7d, 0x49, 0x90, 0x0c, 0xc3;"
        "   vmovdqa64   %%zmm1, %%zmm2;"
        "   loop        1b;"
        "   vzeroupper;"
        :
        : "x" (idx), "r" (mask), "r" (base), "r" (exp_count)
        :
    );
    
    return 0;
}

static int probe_avx512_intrinsic(int idx_init, int idx_step, long count, long repeat)
{
    __m512i idx = _mm512_set_epi32(
            idx_init + idx_step * 0,
            idx_init + idx_step * 1,
            idx_init + idx_step * 2,
            idx_init + idx_step * 3,
            idx_init + idx_step * 4,
            idx_init + idx_step * 5,
            idx_init + idx_step * 6,
            idx_init + idx_step * 7,
            idx_init + idx_step * 8,
            idx_init + idx_step * 9,
            idx_init + idx_step * 10,
            idx_init + idx_step * 11,
            idx_init + idx_step * 12,
            idx_init + idx_step * 13,
            idx_init + idx_step * 14,
            idx_init + idx_step * 15
        );
    const __m512i scale = _mm512_set1_epi32(8);
    
    long exp_count = count * repeat;
    for (long i = 0; i < exp_count; i++) {
        idx = _mm512_mullo_epi32(idx, scale);
        idx = _mm512_i32gather_epi32(idx, (void *)pool, 8);
    }
    
    return _mm512_reduce_add_epi32(idx);
}

#endif

static int probe_scalar1(int idx_init, int idx_step, long count, long repeat)
{
    long idx0 = idx_init + idx_step * 0;
    
    volatile struct chain_idx *base = pool;
    long exp_count = count * repeat;
    for (long i = 0; i < exp_count; i++) {
        idx0 = base[idx0].next;
    }
    
    return 0;
}

static int probe_scalar2(int idx_init, int idx_step, long count, long repeat)
{
    long idx0 = idx_init + idx_step * 0;
    long idx1 = idx_init + idx_step * 1;
    
    volatile struct chain_idx *base = pool;
    long exp_count = count * repeat;
    for (long i = 0; i < exp_count; i++) {
        idx0 = base[idx0].next;
        idx1 = base[idx1].next;
    }
    
    return 0;
}

static int probe_scalar4(int idx_init, int idx_step, long count, long repeat)
{
    long idx0 = idx_init + idx_step * 0;
    long idx1 = idx_init + idx_step * 1;
    long idx2 = idx_init + idx_step * 2;
    long idx3 = idx_init + idx_step * 3;
    
    volatile struct chain_idx *base = pool;
    long exp_count = count * repeat;
    for (long i = 0; i < exp_count; i++) {
        idx0 = base[idx0].next;
        idx1 = base[idx1].next;
        idx2 = base[idx2].next;
        idx3 = base[idx3].next;
    }
    
    return 0;
}

static int probe_scalar8(int idx_init, int idx_step, long count, long repeat)
{
    long idx0 = idx_init + idx_step * 0;
    long idx1 = idx_init + idx_step * 1;
    long idx2 = idx_init + idx_step * 2;
    long idx3 = idx_init + idx_step * 3;
    long idx4 = idx_init + idx_step * 4;
    long idx5 = idx_init + idx_step * 5;
    long idx6 = idx_init + idx_step * 6;
    long idx7 = idx_init + idx_step * 7;
    
    volatile struct chain_idx *base = pool;
    long exp_count = count * repeat;
    for (long i = 0; i < exp_count; i++) {
        idx0 = base[idx0].next;
        idx1 = base[idx1].next;
        idx2 = base[idx2].next;
        idx3 = base[idx3].next;
        idx4 = base[idx4].next;
        idx5 = base[idx5].next;
        idx6 = base[idx6].next;
        idx7 = base[idx7].next;
    }
    
    return 0;
}

static int probe_scalar16(int idx_init, int idx_step, long count, long repeat)
{
    long idx0 = idx_init + idx_step * 0;
    long idx1 = idx_init + idx_step * 1;
    long idx2 = idx_init + idx_step * 2;
    long idx3 = idx_init + idx_step * 3;
    long idx4 = idx_init + idx_step * 4;
    long idx5 = idx_init + idx_step * 5;
    long idx6 = idx_init + idx_step * 6;
    long idx7 = idx_init + idx_step * 7;
    long idx8 = idx_init + idx_step * 8;
    long idx9 = idx_init + idx_step * 9;
    long idx10 = idx_init + idx_step * 10;
    long idx11 = idx_init + idx_step * 11;
    long idx12 = idx_init + idx_step * 12;
    long idx13 = idx_init + idx_step * 13;
    long idx14 = idx_init + idx_step * 14;
    long idx15 = idx_init + idx_step * 15;
    
    volatile struct chain_idx *base = pool;
    long exp_count = count * repeat;
    for (long i = 0; i < exp_count; i++) {
        idx0 = base[idx0].next;
        idx1 = base[idx1].next;
        idx2 = base[idx2].next;
        idx3 = base[idx3].next;
        idx4 = base[idx4].next;
        idx5 = base[idx5].next;
        idx6 = base[idx6].next;
        idx7 = base[idx7].next;
        idx8 = base[idx8].next;
        idx9 = base[idx9].next;
        idx10 = base[idx10].next;
        idx11 = base[idx11].next;
        idx12 = base[idx12].next;
        idx13 = base[idx13].next;
        idx14 = base[idx14].next;
        idx15 = base[idx15].next;
    }
    
    return 0;
}

static int probe_scalar32(int idx_init, int idx_step, long count, long repeat)
{
    long idx0 = idx_init + idx_step * 0;
    long idx1 = idx_init + idx_step * 1;
    long idx2 = idx_init + idx_step * 2;
    long idx3 = idx_init + idx_step * 3;
    long idx4 = idx_init + idx_step * 4;
    long idx5 = idx_init + idx_step * 5;
    long idx6 = idx_init + idx_step * 6;
    long idx7 = idx_init + idx_step * 7;
    long idx8 = idx_init + idx_step * 8;
    long idx9 = idx_init + idx_step * 9;
    long idx10 = idx_init + idx_step * 10;
    long idx11 = idx_init + idx_step * 11;
    long idx12 = idx_init + idx_step * 12;
    long idx13 = idx_init + idx_step * 13;
    long idx14 = idx_init + idx_step * 14;
    long idx15 = idx_init + idx_step * 15;
    long idx16 = idx_init + idx_step * 16;
    long idx17 = idx_init + idx_step * 17;
    long idx18 = idx_init + idx_step * 18;
    long idx19 = idx_init + idx_step * 19;
    long idx20 = idx_init + idx_step * 20;
    long idx21 = idx_init + idx_step * 21;
    long idx22 = idx_init + idx_step * 22;
    long idx23 = idx_init + idx_step * 23;
    long idx24 = idx_init + idx_step * 24;
    long idx25 = idx_init + idx_step * 25;
    long idx26 = idx_init + idx_step * 26;
    long idx27 = idx_init + idx_step * 27;
    long idx28 = idx_init + idx_step * 28;
    long idx29 = idx_init + idx_step * 29;
    long idx30 = idx_init + idx_step * 30;
    long idx31 = idx_init + idx_step * 31;
    
    volatile struct chain_idx *base = pool;
    long exp_count = count * repeat;
    for (long i = 0; i < exp_count; i++) {
        idx0 = base[idx0].next;
        idx1 = base[idx1].next;
        idx2 = base[idx2].next;
        idx3 = base[idx3].next;
        idx4 = base[idx4].next;
        idx5 = base[idx5].next;
        idx6 = base[idx6].next;
        idx7 = base[idx7].next;
        idx8 = base[idx8].next;
        idx9 = base[idx9].next;
        idx10 = base[idx10].next;
        idx11 = base[idx11].next;
        idx12 = base[idx12].next;
        idx13 = base[idx13].next;
        idx14 = base[idx14].next;
        idx15 = base[idx15].next;
        idx16 = base[idx16].next;
        idx17 = base[idx17].next;
        idx18 = base[idx18].next;
        idx19 = base[idx19].next;
        idx20 = base[idx20].next;
        idx21 = base[idx21].next;
        idx22 = base[idx22].next;
        idx23 = base[idx23].next;
        idx24 = base[idx24].next;
        idx25 = base[idx25].next;
        idx26 = base[idx26].next;
        idx27 = base[idx27].next;
        idx28 = base[idx28].next;
        idx29 = base[idx29].next;
        idx30 = base[idx30].next;
        idx31 = base[idx31].next;
    }
    
    return 0;
}

typedef int (*exp_t)(int idx_init, int idx_step, long count, long repeat);

static void experiment(const char *name, exp_t func, int div,
                       long idx_init, int idx_step, long count, long repeat)
{
    func(idx_init, idx_step, count, 100);
    
    double t_begin = get_process_time();
    func(idx_init, idx_step, count, repeat);
    double t_end = get_process_time();
    
    double time = t_end - t_begin;
    double sweep_time = time / (double)count;
    
    printf("Experiment: %18.18s | total time: %8.2lf ms | sweep time: %8.2lf us"
           " | trip time: %8.4lf ns | ** load time: %7.4lf ns\n",
           name, time * 1.0e3, sweep_time * 1.0e6,
           sweep_time / (double)repeat * 1.0e9,
           sweep_time / (double)repeat / (double)div * 1.0e9);
    fflush(stdout);
}

int main(int argc, char *argv[])
{
    // argv[1] -- Access size, in KB
    // argv[2] -- Pool size, in MB
    // argv[3] -- Step size, in KB
    // argv[3] -- Repeat
    long access_size = 1024 * atol(argv[1]);
    long access_count = access_size / sizeof(struct chain_idx);
    
    long pool_size = 1024l * 1024l * atol(argv[2]);
    long pool_count = pool_size / sizeof(struct chain_idx);
    
    long step_size = 1024 * atol(argv[3]);
    long step_count = step_size / sizeof(struct chain_idx);
    
    init(access_count, pool_count, step_count);
    
    long repeat = atol(argv[4]);
    long idx_init = 0;
    long idx_step = 1;
    
    printf("Access: %ld KB, pool: %ld MB, step: %ld KB, repeat: %ld\n",
           access_size / 1024l, pool_size / 1024l / 1024l, step_size / 1024l, repeat);
    
#ifdef __AVX2__
    experiment("AVX2-Pure", probe_avx2_pure, 8, idx_init, idx_step, access_count, repeat);
    experiment("AVX2-Intrinsic", probe_avx2_intrinsic, 8, idx_init, idx_step, access_count, repeat);
#endif
#ifdef __AVX512F__
    experiment("AVX512-Pure", probe_avx512_pure, 16, idx_init, idx_step, access_count, repeat);
    experiment("AVX512-Intrinsic", probe_avx512_intrinsic, 16, idx_init, idx_step, access_count, repeat);
#endif
    experiment("Scaler1", probe_scalar1, 1, idx_init, idx_step, access_count, repeat);
    experiment("Scaler2", probe_scalar2, 2, idx_init, idx_step, access_count, repeat);
    experiment("Scaler4", probe_scalar4, 4, idx_init, idx_step, access_count, repeat);
    experiment("Scaler8", probe_scalar8, 8, idx_init, idx_step, access_count, repeat);
    experiment("Scaler16", probe_scalar16, 16, idx_init, idx_step, access_count, repeat);
    experiment("Scaler32", probe_scalar32, 32, idx_init, idx_step, access_count, repeat);
    
    return 0;
}

