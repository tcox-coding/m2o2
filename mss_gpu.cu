// mss_gpu.cu - mss -s on an NVIDIA GPU (mss -g): the large-prime phase, and
// most leaves of the small-prime search.
//
// Centres m*q with a prime q > B = isqrt(N) are tested exactly as run_block()
// and leaf() in mss.c do it, with the same masks and the same candidate pairs,
// so every count agrees with the CPU search.  The work is cut into units: a
// sieve segment of q (multiples of 3..13 from a pattern, larger primes marked
// in a shared-memory bitmap), its lattice points X^2 + Y^2 = q, and a range of
// cofactor indices.  Each thread takes one q and a run of up to CH cofactors;
// the threads of a warp take different q with the same cofactors, so they
// step through the cofactor table in lockstep and its loads are broadcasts.
// Candidate pairs are queued and tested by a second kernel.  The display may
// run on this GPU (kernel time limit), so blocks take units for LAUNCH_MS and
// the host relaunches until all are done.
//
// Then the leaves n*p of the CPU's search (dfs() in mss.c), p <= B: the CPU
// threads hand over a node n with its run of leaf primes (the same leaf, with
// n for m and p for q), batched in pinned buffers; a warp takes 32 primes of
// one node.  When the buffers are full the CPU threads do the leaves
// themselves, so neither side waits for the other.  In both phases a leaf
// with many offsets that have half the primes (pass 2 of leaf()) is left to
// a third kernel, which gives it a whole warp.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mss_gpu.h"

namespace cg = cooperative_groups;
typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned __int128 u128;

#ifndef WB
#define WB 17              // a sieve segment holds lo + 4i for i < W, about 2^WB
#endif
#define W ((1u << WB) - 128)  // (a block's bitmap and a few words fit in 16 KB)
#define NT 256             // threads per block
#ifndef MINB
#define MINB 3             // blocks per SM to fit (register budget)
#endif
#define PCAP (W / 4)       // primes per segment (at most about 0.12 W)
#define CH 32              // cofactors per work item
#ifndef NS
#define NS 8               // big offsets of a leaf kept in registers (more go to heavy())
#endif
#define GMAXK 1024         // largest leaf handled, 3 cnt + 1
#define PPER 15015         // 3*5*7*11*13
#ifndef HITCAP
#define HITCAP (1u << 22)  // candidates with n^2 + a + b square, per launch (more: it is redone in parts)
#endif
#define QCAP (1u << 26)    // candidate queue, cut into one region per block
#define HQCAP (1u << 22)   // leaves for heavy(), per launch (more: pass2() in place)
#ifndef LAUNCH_MS
#define LAUNCH_MS 40.0     // aimed-for kernel time (the display shares the GPU)
#endif
#ifndef TU
#define TU 5e5             // largest unit, in cost units (one per h step)
#endif
#define NBUF 4             // buffers of leaf runs from the CPU
#ifndef BCOST
#define BCOST (1ull << 27) // cost units per buffer (about 20 ms of GPU time)
#endif
#define BENT (1u << 19)    // node entries per buffer
#define BRUN (1u << 16)    // runs per buffer
#define BITEM (1u << 19)   // warp items per buffer

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                                 \
        }                                                                                            \
    } while (0)

typedef struct {
    u64 lo;
    uint32_t len, i0, i1, pad;  // q in [lo, lo + len), cofactors M[i0..i1)
} unit_t;

struct cparams {
    u64 N;
    const unit_t *U;
    unsigned *ctr;             // next unit
    const uint4 *M;            // m, off, cnt | full << 16 | t1 << 24 | t2 << 28, mask of H(m)[0]
    const uint2 *R;            // projective residues, all bytes below 0x80
    const uint8_t *MK;
    const longlong2 *H;
    const u64 *CSK, *CSP;      // prefix sums over M of k = 3 cnt + 1 and of k (k - 1) / 2
    const uint32_t *AB;
    const uint32_t *SP;        // sieving primes from 17 on
    const u64 *SPM;            // floor(2^64 / p)
    unsigned nsp, nspw;        // how many, and how many are below W / 64 (one warp each)
    const uint32_t *PATB;      // bit j: 1 + 4j has a factor 3..13
    uint2 *PL;                 // per block: lattice points X, Y of the segment's primes
    uint4 *Q;                  // candidates: X, Y, cofactor index, offset indices u | v << 16
    unsigned qb;               // region per block; it takes units while its region is at most half full
    unsigned *nq;              // per block: candidates queued in this launch
    u64 *hits;                 // 5 words per hit: n, x, y
    unsigned *nhits, *err;
    unsigned long long *st;    // nodes, sumk, pairs, fhits, qpass
    // The leaves of the CPU's search: per prime p <= B, X, Y and its qv (two
    // uint4); a batch of runs, as M, R, MK, H for the nodes (BRA: end of the
    // run of primes, off, cnt | ..., mask of H(n)[0]; BRN: n), and its warp
    // items (run, first prime).
    const uint2 *PXY;
    const uint4 *PQ;
    const uint4 *BRA;
    const u64 *BRN;
    const uint2 *BR;
    const uint8_t *BMK;
    const longlong2 *BH;
    const uint2 *BI;
    unsigned *ctr1;            // next warp item
    uint4 *HQ;                 // leaves with more than NS big offsets: X, Y, index as for Q
    unsigned *nh, *ctrh;       // how many, and the next for heavy()
};
__constant__ cparams C;

// -DPROF: block cycles in the sieve, lattice and item stages, and event counts.
#ifdef PROF
__device__ unsigned long long PRF[16];
#define PCNT(i, v) atomicAdd(PRF + (i), (unsigned long long)(v))
#else
#define PCNT(i, v) ((void)0)
#endif

// ---------------------------------------------------------------- masks
//
// A Gaussian integer h is one byte per usable prime (rho, as in mss.c, with
// 0x7f for h = 0 mod p), a prime factor f its alpha and beta bytes.  Bit i of
// the offset mask of h f is rho_i in {alpha_i, beta_i, 0x7f}.  With every
// byte below 0x80, (r ^ a) + 0x7f has bit 7 of a byte set iff that byte
// differs, and no carries cross bytes.

#define PAIR_U 0x211000u  // the six pairs u < v of four things, a nibble each
#define PAIR_V 0x332321u
#define K7 0x7f7f7f7fu
#define H8 0x80808080u

__device__ __forceinline__ unsigned ne(unsigned r, unsigned a) { return (r ^ a) + K7; }

// 0x80 in each byte where r is a, b or 0x7f (tz = ne(r, K7)).
__device__ __forceinline__ unsigned eq3(unsigned r, unsigned a, unsigned b, unsigned tz) {
    return ~(ne(r, a) & ne(r, b) & tz) & H8;
}

// Bits 7, 15, 23, 31 of lo and hi to bits 0..7 (the products do not overlap).
__device__ __forceinline__ unsigned gather(unsigned lo, unsigned hi) {
    return (lo * 0x00204081u) >> 28 | (hi * 0x00204081u) >> 28 << 4;
}

// Bits 0..3 to 0x80 in bytes 0..3.
__device__ __forceinline__ unsigned spread(unsigned x) { return ((x * 0x00204081u) & 0x01010101u) << 7; }

__device__ __forceinline__ void top2(unsigned r, unsigned &t1, unsigned &t2) {
    unsigned lo = min(r, t1);
    t2 = max(lo, t2);
    t1 = max(r, t1);
}

struct qv {
    unsigned a0l, a0h, b0l, b0h, a2l, a2h, b2l, b2h;  // alpha, beta of conj(pi)^2 and pi^2, lanes 0-3 and 4-7
};

// alpha, beta bytes of q = X^2 + Y^2, from the table of each prime p.
__device__ __forceinline__ qv prime_ab(unsigned X, unsigned Y) {
    unsigned t[8];
#define ABL(i, p, o) t[i] = __ldg(C.AB + (o) + (X % (p)) * (p) + Y % (p))
    ABL(0, 7, 0), ABL(1, 11, 49), ABL(2, 13, 170), ABL(3, 17, 339);
    ABL(4, 19, 628), ABL(5, 29, 989), ABL(6, 31, 1830), ABL(7, 37, 2791);
#undef ABL
    qv v;
    unsigned x = __byte_perm(t[0], t[1], 0x5140), y = __byte_perm(t[2], t[3], 0x5140);
    v.a0l = __byte_perm(x, y, 0x5410), v.b0l = __byte_perm(x, y, 0x7632);
    x = __byte_perm(t[0], t[1], 0x7362), y = __byte_perm(t[2], t[3], 0x7362);
    v.a2l = __byte_perm(x, y, 0x5410), v.b2l = __byte_perm(x, y, 0x7632);
    x = __byte_perm(t[4], t[5], 0x5140), y = __byte_perm(t[6], t[7], 0x5140);
    v.a0h = __byte_perm(x, y, 0x5410), v.b0h = __byte_perm(x, y, 0x7632);
    x = __byte_perm(t[4], t[5], 0x7362), y = __byte_perm(t[6], t[7], 0x7362);
    v.a2h = __byte_perm(x, y, 0x5410), v.b2h = __byte_perm(x, y, 0x7632);
    return v;
}

// ---------------------------------------------------------------- candidates
//
// Candidate pairs are queued (q's lattice point, the cofactor's index, the two
// offsets' indices) and tested by check(), one per thread.  Tested in place,
// in divergent code, they took most of the time; so did one global counter,
// and inlined copies of push() (the kernel outgrew the instruction cache).

__shared__ unsigned s_nq;  // candidates in this block's region

__device__ __forceinline__ u128 offs(i64 re, i64 im) {
    return (u128)(2 * (u64)(re < 0 ? -re : re)) * (u64)(im < 0 ? -im : im);
}

// The i-th offset of the leaf m*q (as leaf_offset in mss.c): blocks H f0, H q,
// H f2, then m f0, with f0 = (fr, -fi), f2 = (fr, fi).
__device__ __forceinline__ u128 leaf_offset(const longlong2 *H, u64 q, i64 fr, i64 fi, u64 m, unsigned cnt,
                                            unsigned off, unsigned i) {
    unsigned t = (i >= cnt) + (i >= 2 * cnt) + (i >= 3 * cnt);
    if (t == 3) return offs(fr * (i64)m, fi * (i64)m);
    longlong2 h = __ldg(H + off + i - t * cnt);
    i64 a = t == 1 ? (i64)q : fr, b = t == 1 ? 0 : t == 0 ? -fi : fi;
    return offs(h.x * a - h.y * b, h.x * b + h.y * a);
}

// As is_sq_fast in mss.c, bit for bit.
__device__ __forceinline__ bool is_sq_fast(u128 v) {
    double d = (double)(u64)(v >> 64) * 18446744073709551616.0 + (double)(u64)v;
    u64 r = (u64)sqrt(d);
    return (u128)r * r == v || (u128)(r + 1) * (r + 1) == v;
}

// A queued candidate (sol_pair() in mss.c): if n^2 + a + b is a square it
// goes to the host for the rest of the check.  c.z is a cofactor's index, or
// with bit 31 set a run's in the batch of leaves.
__device__ __forceinline__ void test(uint4 c) {
    u64 X = c.x, Y = c.y, q = X * X + Y * Y, m;
    i64 fr = (i64)(X * X) - (i64)(Y * Y), fi = (i64)(2 * X * Y);
    const longlong2 *H = C.H;
    uint4 r;
    if (c.z >> 31) {
        r = __ldg(C.BRA + (c.z & 0x7fffffff)), m = __ldg(C.BRN + (c.z & 0x7fffffff)), H = C.BH;
    } else {
        r = __ldg(C.M + c.z), m = r.x;
    }
    unsigned cnt = r.z & 0xffff;
    u64 n = m * q;
    u128 du = leaf_offset(H, q, fr, fi, m, cnt, r.y, c.w & 0xffff), dv = leaf_offset(H, q, fr, fi, m, cnt, r.y, c.w >> 16);
    u128 x = du < dv ? du : dv, y = du < dv ? dv : du, sq = (u128)n * n, a = x + y;
    if (a >= sq || !is_sq_fast(sq + a)) return;
    atomicAdd(C.st + 4, 1ull);
    unsigned h = atomicAdd(C.nhits, 1u);
    if (h < HITCAP) {
        u64 *o = C.hits + 5 * (size_t)h;
        o[0] = n, o[1] = (u64)x, o[2] = (u64)(x >> 64), o[3] = (u64)y, o[4] = (u64)(y >> 64);
    } else {
        atomicOr(C.err, 2u);
    }
}

// The candidates queued by phase2() block b, tested by block b.
__global__ void check(void) {
    unsigned nq = min(C.nq[blockIdx.x], C.qb);
    if (threadIdx.x == 0) atomicAdd(C.st + 3, (unsigned long long)nq);
    for (unsigned k = threadIdx.x; k < nq; k += blockDim.x) test(C.Q[(size_t)blockIdx.x * C.qb + k]);
}

__device__ __noinline__ void test_here(uint4 c) {  // a block's region is full
    atomicAdd(C.st + 3, 1ull);
    test(c);
}

__device__ __forceinline__ void put(uint4 c, unsigned k) {  // the k-th candidate of the block
    if (k < C.qb) C.Q[(size_t)blockIdx.x * C.qb + k] = c;
    else test_here(c);
}

__device__ __noinline__ void push(unsigned X, unsigned Y, unsigned i, unsigned u, unsigned v) {
    cg::coalesced_group g = cg::coalesced_threads();
    unsigned k = 0;
    if (g.thread_rank() == 0) k = atomicAdd(&s_nq, g.size());
    put(make_uint4(X, Y, i, u | v << 16), g.shfl(k, 0) + g.thread_rank());
}

// Slots for n candidates of each lane of a whole warp, with one atomic: the
// lane's first.
__device__ __forceinline__ unsigned warp_slots(unsigned n, unsigned lane) {
    unsigned pre = n, k = 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        unsigned y = __shfl_up_sync(~0u, pre, o);
        if (lane >= o) pre += y;
    }
    if (lane == 31) k = atomicAdd(&s_nq, pre);
    return __shfl_sync(~0u, k, 31) + pre - n;
}

// Pass 2 of leaf() for more than NS big offsets: the masks, and every pair
// covering full (cover_pairs() in mss.c, scalar), if heavy() has no room.
__device__ __noinline__ void pass2(unsigned X, unsigned Y, unsigned i, unsigned cnt, unsigned off, unsigned full,
                                   qv v, unsigned wn, const uint2 *R, const uint8_t *MK) {
    uint8_t K[GMAXK];
    uint16_t big[GMAXK];
    unsigned B = __popc(full), half = (B + 1) / 2, k = 3 * cnt + 1, nb = 0;
    for (unsigned h = 0; h < cnt; h++) {
        uint2 x = __ldg(R + off + h);
        unsigned zl = ne(x.x, K7), zh = ne(x.y, K7);
        unsigned m0 = gather(eq3(x.x, v.a0l, v.b0l, zl), eq3(x.y, v.a0h, v.b0h, zh)) & full;
        unsigned m2 = gather(eq3(x.x, v.a2l, v.b2l, zl), eq3(x.y, v.a2h, v.b2h, zh)) & full;
        unsigned m1 = __ldg(MK + off + h) & full;
        K[h] = (uint8_t)m0, K[cnt + h] = (uint8_t)m1, K[2 * cnt + h] = (uint8_t)m2;
        big[nb] = (uint16_t)h, nb += (unsigned)__popc(m0) >= half;
        big[nb] = (uint16_t)(cnt + h), nb += (unsigned)__popc(m1) >= half;
        big[nb] = (uint16_t)(2 * cnt + h), nb += (unsigned)__popc(m2) >= half;
    }
    K[3 * cnt] = (uint8_t)wn;
    big[nb] = (uint16_t)(3 * cnt), nb += (unsigned)__popc(wn) >= half;
    for (unsigned b = 0; b < nb; b++) {
        unsigned u = big[b], need = full & ~(unsigned)K[u];
        if (B - __popc(K[u]) >= half) {  // the partner is big too: take the pair from the larger index
            for (unsigned w = 0; w < nb; w++)
                if (big[w] < u && !(need & ~(unsigned)K[big[w]])) push(X, Y, i, u, big[w]);
        } else {
            for (unsigned x = 0; x < k; x++)
                if (x != u && !(need & ~(unsigned)K[x]) && (x < u || (unsigned)__popc(K[x]) < half))
                    push(X, Y, i, u, x);
        }
    }
}

// A leaf for heavy(): its records as for a candidate (i is m's index).
__device__ __noinline__ void defer(unsigned X, unsigned Y, unsigned i, unsigned cnt, unsigned off, unsigned full,
                                   qv v, unsigned wn, const uint2 *R, const uint8_t *MK) {
    unsigned k = atomicAdd(C.nh, 1u);
    if (k < HQCAP) C.HQ[k] = make_uint4(X, Y, i, 0);
    else pass2(X, Y, i, cnt, off, full, v, wn, R, MK);
}

// sl[a] for an a not known at compile time, so that sl stays in registers.
__device__ __forceinline__ unsigned slot(const unsigned *sl, unsigned a) {
    unsigned r = sl[0];
#pragma unroll
    for (unsigned k = 1; k < NS; k++) r = a == k ? sl[k] : r;
    return r;
}

// The leaf m*q for the prime q = X^2 + Y^2 (leaf() in mss.c): r is m's record
// (off, cnt | full << 16 | t1 << 24 | t2 << 28 and the mask of H(m)[0] in y, z,
// w), i its index for the candidates, R and MK its table.
__device__ __forceinline__ void leafm(unsigned X, unsigned Y, const qv &v, uint4 r, unsigned i,
                                      const uint2 *__restrict__ R, const uint8_t *__restrict__ MK) {
    unsigned cnt = r.z & 0xffff, full = r.z >> 16 & 0xff, off = r.y;
    if (cnt == 1) {
        // Prime m: offsets h f0, h q, h f2, m f0, and the six pairs directly.
        uint2 h = __ldg(R + off), w = __ldg(R + off + 1);
        unsigned zl = ne(h.x, K7), zh = ne(h.y, K7);
        unsigned m0 = gather(eq3(h.x, v.a0l, v.b0l, zl), eq3(h.y, v.a0h, v.b0h, zh)) & full;
        unsigned m2 = gather(eq3(h.x, v.a2l, v.b2l, zl), eq3(h.y, v.a2h, v.b2h, zh)) & full;
        unsigned m3 = gather(eq3(w.x, v.a0l, v.b0l, ne(w.x, K7)), eq3(w.y, v.a0h, v.b0h, ne(w.y, K7))) & full;
        unsigned m1 = r.w & full;
        unsigned pm = ((m0 | m1) == full) | ((m0 | m2) == full) << 1 | ((m0 | m3) == full) << 2 |
                      ((m1 | m2) == full) << 3 | ((m1 | m3) == full) << 4 | ((m2 | m3) == full) << 5;
        for (; __builtin_expect(pm != 0, 0); pm &= pm - 1) {
            unsigned b = 4 * (__ffs(pm) - 1);
            push(X, Y, i, PAIR_U >> b & 15, PAIR_V >> b & 15);
        }
        return;
    }
    // Pass 1: the two largest popcounts (those of the h q block are known),
    // and the first NS offsets with at least half the primes ("big"; one
    // is in every covering pair) as index | mask << 16.
    unsigned B = __popc(full), half = (B + 1) / 2, t1 = r.z >> 24 & 15, t2 = r.z >> 28;
    unsigned fl = spread(full & 15), fhm = spread(full >> 4), nb = 0, sl[NS];
#pragma unroll
    for (unsigned a = 0; a < NS; a++) sl[a] = 0;
#define PUSHB(i, mk)                                                                \
    do {                                                                            \
        _Pragma("unroll") for (unsigned a_ = NS - 1; a_; a_--) sl[a_] = sl[a_ - 1]; \
        sl[0] = (i) | (mk) << 16, nb++;                                             \
    } while (0)
    for (unsigned h = 0; h < cnt; h++) {
        uint2 x = __ldg(R + off + h);
        unsigned zl = ne(x.x, K7), zh = ne(x.y, K7);
        unsigned e0l = eq3(x.x, v.a0l, v.b0l, zl) & fl, e0h = eq3(x.y, v.a0h, v.b0h, zh) & fhm;
        unsigned e2l = eq3(x.x, v.a2l, v.b2l, zl) & fl, e2h = eq3(x.y, v.a2h, v.b2h, zh) & fhm;
        unsigned p0 = __popc(e0l | e0h >> 1), p2 = __popc(e2l | e2h >> 1), m1 = __ldg(MK + off + h) & full;
        top2(p0, t1, t2), top2(p2, t1, t2);
        if (__builtin_expect(max(max(p0, p2), (unsigned)__popc(m1)) >= half, 0)) {
            if (p0 >= half) PUSHB(h, gather(e0l, e0h));
            if ((unsigned)__popc(m1) >= half) PUSHB(cnt + h, m1);
            if (p2 >= half) PUSHB(2 * cnt + h, gather(e2l, e2h));
        }
    }
    uint2 x = __ldg(R + off + cnt);
    unsigned wn = gather(eq3(x.x, v.a0l, v.b0l, ne(x.x, K7)), eq3(x.y, v.a0h, v.b0h, ne(x.y, K7))) & full;
    top2(__popc(wn), t1, t2);
    if ((unsigned)__popc(wn) >= half) PUSHB(3 * cnt, wn);
#undef PUSHB
    if (t1 + t2 < B) return;
    PCNT(3, 1), PCNT(4, nb > NS);
    if (nb > NS) {
        defer(X, Y, i, cnt, off, full, v, wn, R, MK);
        return;
    }
    // Pairs of two big offsets; then, for the big ones with more than
    // B - half primes, their partners with fewer than half.  Hits are
    // collected as bit masks and pushed from one place.
    unsigned sw = 0, pm = 0, pb = 0;  // pm: the big pairs a < b in order
#pragma unroll
    for (unsigned a = 0; a < NS; a++) {
#pragma unroll
        for (unsigned b = a + 1; b < NS; b++) pm |= (unsigned)(b < nb && ((sl[a] | sl[b]) >> 16) == full) << pb++;
        if (a < nb && (unsigned)__popc(sl[a] >> 16) > B - half) sw |= 1u << a;
    }
    for (; __builtin_expect(pm != 0, 0); pm &= pm - 1) {
        unsigned p = __ffs(pm) - 1, a = 0;
        for (; p >= NS - 1 - a; a++) p -= NS - 1 - a;
        push(X, Y, i, slot(sl, a) & 0xffff, slot(sl, a + 1 + p) & 0xffff);
    }
    PCNT(5, sw != 0);
    if (sw) {
        // One more sweep over the offsets, recomputing their masks.
        unsigned nd[NS];
#pragma unroll
        for (unsigned a = 0; a < NS; a++) nd[a] = sw >> a & 1 ? full & ~(sl[a] >> 16) : 0x100;
        for (unsigned h = 0; h <= cnt; h++) {
            unsigned mk[3];
            if (h < cnt) {
                uint2 x = __ldg(R + off + h);
                unsigned zl = ne(x.x, K7), zh = ne(x.y, K7);
                mk[0] = gather(eq3(x.x, v.a0l, v.b0l, zl), eq3(x.y, v.a0h, v.b0h, zh)) & full;
                mk[1] = __ldg(MK + off + h) & full;
                mk[2] = gather(eq3(x.x, v.a2l, v.b2l, zl), eq3(x.y, v.a2h, v.b2h, zh)) & full;
            } else {
                mk[0] = wn, mk[1] = mk[2] = full;  // the n term; full: never a partner
            }
            unsigned hm = 0;  // bit NS e + a: entry e partners big a
#pragma unroll
            for (unsigned e = 0; e < 3; e++)
#pragma unroll
                for (unsigned a = 0; a < NS; a++)
                    hm |= (unsigned)((unsigned)__popc(mk[e]) < half && !(nd[a] & ~mk[e])) << (NS * e + a);
            for (; __builtin_expect(hm != 0, 0); hm &= hm - 1) {
                unsigned b = __ffs(hm) - 1, e = b / NS;
                push(X, Y, i, slot(sl, b % NS) & 0xffff, h == cnt ? 3 * cnt : e * cnt + h);
            }
        }
    }
}

// The leaves m*q for the prime q = X^2 + Y^2 and the cofactors M[ia..ib) up to
// N/q (run_block() in mss.c).
__device__ __forceinline__ void item(unsigned X, unsigned Y, unsigned ia, unsigned ib, unsigned &nodes, u64 &sumk,
                                     u64 &pairs) {
    u64 q = (u64)X * X + (u64)Y * Y;
    qv v = prime_ab(X, Y);
    unsigned i = ia;
    for (; i < ib; i++) {
        uint4 r = __ldg(C.M + i);
        if ((u64)r.x * q > C.N) break;
        leafm(X, Y, v, r, i, C.R, C.MK);
    }
    nodes += i - ia;
    sumk += __ldg(C.CSK + i) - __ldg(C.CSK + ia);
    pairs += __ldg(C.CSP + i) - __ldg(C.CSP + ia);
}

// ---------------------------------------------------------------- sieve and lattice

__device__ __forceinline__ u64 isqrt_d(u64 x) {  // x < 2^53
    u64 r = (u64)sqrt((double)x);
    while (r * r > x) r--;
    while ((r + 1) * (r + 1) <= x) r++;
    return r;
}

// Index (s - lo) / 4 of the first multiple s >= max(lo, p^2) of p with s = 1 (mod 4).
__device__ __forceinline__ u64 first_idx(u64 lo, u64 p, u64 mg) {
    u64 r = lo - __umul64hi(lo, mg) * p;  // lo mod p, or that plus p
    if (r >= p) r -= p;
    u64 s = lo + (r ? p - r : 0);
    if (s < p * p) s = p * p;
    while ((s & 3) != 1) s += p;
    return (s - lo) >> 2;
}

__device__ __forceinline__ u64 gtimer(void) {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Units from *C.ctr on, below nu, taken until ns nanoseconds have passed.
__global__ void __launch_bounds__(NT, MINB) phase2(unsigned nu, u64 ns) {
    __shared__ uint32_t bm[W / 32];  // bit i: lo + 4i is composite
    __shared__ unsigned s_u, s_np, s_ws[NT / 32];
    unsigned nodes = 0;
    u64 sumk = 0, pairs = 0;
    const unsigned tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    uint2 *PL = C.PL + (size_t)blockIdx.x * PCAP;
    const u64 t0 = gtimer();
    if (tid == 0) s_nq = 0;
#ifdef PROF
    long long T[3] = {0, 0, 0}, ta = clock64();
#define PSTAGE(i) (T[i] += clock64() - ta, ta = clock64())
#else
#define PSTAGE(i) ((void)0)
#endif
    for (;;) {
        if (tid == 0)
            s_u = gtimer() - t0 > ns || s_nq > C.qb / 2 ? ~0u : atomicAdd(C.ctr, 1u);
        __syncthreads();
        const unsigned u = s_u;
        if (u >= nu) break;
        const unit_t U = C.U[u];
        const u64 lo = U.lo, hi = lo + U.len;
        const unsigned nidx = (U.len + 3) / 4;

        unsigned j0 = (unsigned)((lo >> 2) % PPER);
        for (unsigned w = tid; w < (nidx + 31) / 32; w += NT) {
            unsigned b = j0 + 32 * w;
            bm[w] = __funnelshift_r(__ldg(C.PATB + (b >> 5)), __ldg(C.PATB + (b >> 5) + 1), b & 31);
        }
        __syncthreads();
        for (unsigned i = wid; i < C.nspw; i += NT / 32) {
            u64 p = __ldg(C.SP + i);
            if (p * p >= hi) break;
            for (u64 t = first_idx(lo, p, __ldg(C.SPM + i)) + lane * p; t < nidx; t += 32 * p)
                atomicOr(&bm[t >> 5], 1u << (t & 31));
        }
        for (unsigned i = C.nspw + tid; i < C.nsp; i += NT) {
            u64 p = __ldg(C.SP + i);
            if (p * p >= hi) break;
            for (u64 t = first_idx(lo, p, __ldg(C.SPM + i)); t < nidx; t += p) atomicOr(&bm[t >> 5], 1u << (t & 31));
        }
        __syncthreads();
        PSTAGE(0);

        // Lattice points X > Y > 0, X + Y odd: rows X where 8Y >= X, columns Y
        // where 8Y < X, so no row or column holds more than a few points.  Rows
        // and then columns are cut into contiguous runs, one per thread.  In a
        // run of rows the bounds l <= Y <= h fall monotonically (yl, yh in
        // run_block()), and move by 32-bit differences: A = X^2 + (l-1)^2 - lo
        // < 0 and Bh = X^2 + h^2 - hi < 0, each within about 2X of 0; a point's
        // q - lo is Bh + len - (h^2 - Y^2).  Columns are the same with X, Y
        // swapped.
        {
            u64 Xa = isqrt_d(lo / 2), Xb = min(isqrt_d(hi - 2), isqrt_d((hi - 1) * 64 / 65) + 1);
            u64 nA = Xb >= Xa ? Xb + 1 - Xa : 0, nB = isqrt_d((hi - 1) / 65) + 1, ch = (nA + nB + NT - 1) / NT;
            u64 s = tid * ch, e = min(s + ch, nA + nB);
            const int len = (int)(hi - lo);
            // The run [a, b) of lines t = a, a + 1, ... (rows X or columns Y);
            // emit(X, Y, q - lo) for each point.
            auto walk = [&](u64 a, u64 b, bool rows, auto emit) {
                u64 t2 = a * a;
                unsigned t = (unsigned)a, l = t2 >= lo ? 0 : (unsigned)(isqrt_d(lo - t2 - 1) + 1);
                unsigned h = (unsigned)isqrt_d(hi - 1 - t2);
                int A = l ? (int)(i64)(t2 + (u64)(l - 1) * (l - 1) - lo) : 0, Bh = (int)(i64)(t2 + (u64)h * h - hi);
                for (; t < (unsigned)b; t++) {
                    while (l && A >= 0) A -= 2 * (int)l - 3, l--;
                    while (Bh >= 0) Bh -= 2 * (int)h - 1, h--;
                    unsigned lo2 = rows ? max(max(l, (t + 7) / 8), 1u) : max(l, 8 * t + 1);
                    unsigned hi2 = rows ? min(h, t - 1) : h;
                    lo2 += !((t + lo2) & 1);
                    for (unsigned o = lo2; o <= hi2; o += 2) {
                        unsigned d = (unsigned)(Bh + len) - (h - o) * (h + o);
                        if (rows) emit(t, o, d);
                        else emit(o, t, d);
                    }
                    if (l) A += 2 * (int)t + 1;
                    Bh += 2 * (int)t + 1;
                }
            };
            // Two walks: the first counts each thread's primes, a scan over the
            // block gives each its place, and the second writes them there.
            // (Appending through one shared counter was slower than walking
            // twice: its atomics serialise.)
            unsigned nl = 0;
            auto count = [&](unsigned, unsigned, unsigned d) { nl += ~bm[d >> 7] >> (d >> 2 & 31) & 1; };
            if (s < min(e, nA)) walk(Xa + s, Xa + min(e, nA), true, count);
            if (max(s, nA) < e) walk(1 + max(s, nA) - nA, 1 + e - nA, false, count);
            {
                unsigned v = nl;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    unsigned y = __shfl_up_sync(~0u, v, o);
                    if (lane >= o) v += y;
                }
                if (lane == 31) s_ws[wid] = v;
                __syncthreads();
                unsigned wb = 0, tot = 0;
                for (unsigned k = 0; k < NT / 32; k++) wb += k < wid ? s_ws[k] : 0, tot += s_ws[k];
                if (tid == 0) s_np = tot;
                nl = wb + v - nl;
            }
            auto put = [&](unsigned X, unsigned Y, unsigned d) {
                if (bm[d >> 7] >> (d >> 2 & 31) & 1) return;
                if (nl < PCAP) PL[nl] = make_uint2(X, Y);
                nl++;
            };
            if (s < min(e, nA)) walk(Xa + s, Xa + min(e, nA), true, put);
            if (max(s, nA) < e) walk(1 + max(s, nA) - nA, 1 + e - nA, false, put);
        }
        __syncthreads();
        PSTAGE(1);

        unsigned np = s_np;
        if (np > PCAP) {
            if (tid == 0) atomicOr(C.err, 1u);
            np = PCAP;
        }
        // Work items: chunk c of the cofactors, prime j; a warp shares c.
        const unsigned np32 = (np + 31) & ~31u, items = (U.i1 - U.i0 + CH - 1) / CH * np32;
        for (unsigned w = tid; w < items; w += NT) {
            unsigned c = w / np32, j = w - c * np32;
            if (j >= np) continue;
            uint2 xy = PL[j];
            unsigned ia = U.i0 + c * CH;
            item(xy.x, xy.y, ia, min(ia + CH, U.i1), nodes, sumk, pairs);
        }
        __syncthreads();
        PSTAGE(2);
    }
    for (int o = 16; o; o >>= 1) {
        nodes += __shfl_down_sync(~0u, nodes, o);
        sumk += __shfl_down_sync(~0u, sumk, o), pairs += __shfl_down_sync(~0u, pairs, o);
    }
#ifdef PROF
    if (tid == 0) PCNT(0, T[0]), PCNT(1, T[1]), PCNT(2, T[2]);
#endif
    if (tid == 0) C.nq[blockIdx.x] = s_nq;
    if (lane == 0) {
        atomicAdd(C.st + 0, (unsigned long long)nodes), atomicAdd(C.st + 1, (unsigned long long)sumk);
        atomicAdd(C.st + 2, (unsigned long long)pairs);
    }
}

// ---------------------------------------------------------------- leaves of the small-prime search

// qv of every prime p <= B, from its X, Y.
__global__ void prime_tab(unsigned np) {
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= np) return;
    uint2 xy = C.PXY[j];
    qv v = prime_ab(xy.x, xy.y);
    uint4 *o = (uint4 *)C.PQ + 2 * j;
    o[0] = make_uint4(v.a0l, v.a0h, v.b0l, v.b0h), o[1] = make_uint4(v.a2l, v.a2h, v.b2l, v.b2h);
}

// Warp items i0..i1 of a batch: each is up to 32 primes of one run (node), a
// lane each, so the lanes step through the node's tables together.
__global__ void __launch_bounds__(NT, MINB) leaves(unsigned i0, unsigned i1) {
    const unsigned lane = threadIdx.x & 31;
    if (threadIdx.x == 0) s_nq = 0;
    __syncthreads();
    for (;;) {
        unsigned w = 0;
        if (lane == 0) w = i0 + atomicAdd(C.ctr1, 1u);
        w = __shfl_sync(~0u, w, 0);
        if (w >= i1) break;
        uint2 it = __ldg(C.BI + w);
        uint4 r = __ldg(C.BRA + it.x);
        unsigned j = it.y + lane;
        if (j < r.x) {
            uint2 xy = __ldg(C.PXY + j);
            uint4 a = __ldg(C.PQ + 2 * j), b = __ldg(C.PQ + 2 * j + 1);
            qv v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            leafm(xy.x, xy.y, v, r, it.x | 1u << 31, C.BR, C.BMK);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) C.nq[blockIdx.x] = s_nq;
}

// Pass 2 for the leaves defer()red by the last launch, a warp each: the masks
// in shared memory, with a bit per offset for the big ones and one for those
// with fewer than half the primes; then every covering pair (as pass2()), the
// lanes taking the partners, four masks to a word.  Candidates go on in the
// block's region of the queue.
__global__ void __launch_bounds__(NT) heavy(void) {
    __shared__ uint32_t sK[NT / 32][GMAXK / 4], sG[NT / 32][GMAXK / 32], sS[NT / 32][GMAXK / 32];
    const unsigned lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nh = min(*C.nh, HQCAP);
    uint8_t *K = (uint8_t *)sK[wid];
    const uint32_t *K4 = sK[wid], *G = sG[wid], *S = sS[wid];
    if (threadIdx.x == 0) s_nq = C.nq[blockIdx.x];
    __syncthreads();
    for (;;) {
        unsigned w = 0;
        if (lane == 0) w = atomicAdd(C.ctrh, 1u);
        w = __shfl_sync(~0u, w, 0);
        if (w >= nh) break;
        uint4 c = C.HQ[w], r;
        const unsigned X = c.x, Y = c.y, i = c.z;
        const uint2 *R = C.R;
        const uint8_t *MK = C.MK;
        if (i >> 31) r = __ldg(C.BRA + (i & 0x7fffffff)), R = C.BR, MK = C.BMK;
        else r = __ldg(C.M + i);
        const unsigned cnt = r.z & 0xffff, full = r.z >> 16 & 0xff, off = r.y, k = 3 * cnt + 1;
        const unsigned B = __popc(full), half = (B + 1) / 2;
        qv v = prime_ab(X, Y);
        for (unsigned h = lane; h <= cnt; h += 32) {
            uint2 x = __ldg(R + off + h);
            unsigned zl = ne(x.x, K7), zh = ne(x.y, K7);
            unsigned m0 = gather(eq3(x.x, v.a0l, v.b0l, zl), eq3(x.y, v.a0h, v.b0h, zh)) & full;
            if (h == cnt) {
                K[3 * cnt] = (uint8_t)m0;  // the n term
            } else {
                K[h] = (uint8_t)m0, K[cnt + h] = (uint8_t)(__ldg(MK + off + h) & full);
                K[2 * cnt + h] = (uint8_t)(gather(eq3(x.x, v.a2l, v.b2l, zl), eq3(x.y, v.a2h, v.b2h, zh)) & full);
            }
        }
        __syncwarp();
        for (unsigned e0 = 0; e0 < k; e0 += 32) {
            unsigned e = e0 + lane, p = e < k ? __popc(K[e]) : half;
            unsigned g = __ballot_sync(~0u, e < k && p >= half), sm = __ballot_sync(~0u, p < half);
            if (lane == 0) sG[wid][e0 / 32] = g, sS[wid][e0 / 32] = sm;
        }
        __syncwarp();
        // Partners x of a big u: need within K[x] (a zero byte of need & ~K[x]);
        // if u has at most B - half primes, x is big and x < u, else x < u or x
        // small.  The bytes past k are neither.
        for (unsigned g0 = 0; g0 < k; g0 += 32)
            for (unsigned gm = G[g0 / 32]; gm; gm &= gm - 1) {
                unsigned u = g0 + __ffs(gm) - 1, need = full & ~(unsigned)K[u], n4 = need * 0x01010101u;
                const bool ob = B - __popc(K[u]) >= half;
                const uint32_t *F = ob ? G : S;
                const unsigned nw = ob ? (u + 3) / 4 : (k + 3) / 4;
                for (unsigned t0 = 0; t0 < nw; t0 += 32) {
                    unsigned t = t0 + lane, hit = 0;
                    if (t < nw) {
                        unsigned d = n4 & ~K4[t], z = ~(((d & 0x7f7f7f7fu) + 0x7f7f7f7fu) | d) & H8;
                        int c = (int)u - 4 * (int)t;
                        unsigned lt = c >= 4 ? H8 : c <= 0 ? 0 : H8 & ((1u << 8 * c) - 1);
                        unsigned f = spread(F[t >> 3] >> 4 * (t & 7) & 15);
                        hit = z & (ob ? lt & f : lt | f);
                    }
                    if (!__ballot_sync(~0u, hit)) continue;
                    for (unsigned q = warp_slots(__popc(hit), lane); hit; hit &= hit - 1, q++)
                        put(make_uint4(X, Y, i, u | (4 * t + (__ffs(hit) - 1) / 8) << 16), q);
                }
            }
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x == 0) C.nq[blockIdx.x] = s_nq;
}

// ---------------------------------------------------------------- host

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

template <typename T>
static T *upload(const T *h, size_t n) {
    T *d;
    CK(cudaMalloc(&d, (n ? n : 1) * sizeof(T)));
    if (n) CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Buffers of leaf runs in pinned memory, filled by the CPU threads under bl:
// one is open, full ones wait in bready for the GPU, the rest are in bfree.
typedef struct {
    longlong2 *h;
    uint4 *ra;
    uint2 *r, *it;
    u64 *rn;
    uint8_t *mk;
    unsigned ne, nr, ni;  // entries, runs, warp items
    u64 cost, leaves;
} batch_t;

static batch_t bufs[NBUF], *bfree[NBUF], *bready[NBUF], *bopen;
static int nfree, nready, bdone, bwait, bdead;
static int bavail;  // a buffer can take runs (also read without bl)
static pthread_mutex_t bl = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t bc = PTHREAD_COND_INITIALIZER;

// gpu_snapshot(): a request (sreq), answered by the GPU thread between
// launches once the buffers are empty (sdone, the counts in snap, signalled
// on sc); after the GPU is done, snap holds the final counts (sfin).
static int sreq, sdone, sfin, sloop;  // sloop: the loops answer (before: nothing done yet)
static u64 ustart;
static gpu_out snap;
static pthread_cond_t sc = PTHREAD_COND_INITIALIZER;
static u64 nsol, nrep;  // hits that were magic squares (in->hit)
static batch_t qmark;   // take(): a snapshot is wanted and every buffer is done

extern "C" int gpu_leaves(uint64_t n, int cnt, unsigned full, unsigned top1, const int64_t *h, const uint8_t *rho,
                          const uint16_t *mk, const uint8_t *nrho, uint32_t j0, uint32_t j1) {
    unsigned ni = (j1 - j0 + 31) / 32;
    if (!__atomic_load_n(&bavail, __ATOMIC_RELAXED) || 3 * (unsigned)cnt + 1 > GMAXK || ni > BITEM) return 0;
    u64 cost = (u64)(j1 - j0) * (cnt == 1 ? 4 : 2 + 3 * (u64)cnt);
    pthread_mutex_lock(&bl);
    batch_t *b = bopen;
    if (b && (b->ne + cnt + 1 > BENT || b->nr == BRUN || b->ni + ni > BITEM || b->cost + cost > BCOST)) {
        bready[nready++] = b, bopen = b = NULL;
        if (bwait) pthread_cond_signal(&bc);
    }
    if (!b && nfree && !bdead) {
        b = bopen = bfree[--nfree];
        b->ne = b->nr = b->ni = 0, b->cost = b->leaves = 0;
    }
    if (!b || bdead) {
        __atomic_store_n(&bavail, 0, __ATOMIC_RELAXED);
        pthread_mutex_unlock(&bl);
        return 0;
    }
    unsigned off = b->ne, r = b->nr;
    memcpy(b->h + off, h, (size_t)cnt * sizeof *b->h);
    for (int e = 0; e <= cnt; e++) {
        u64 x;
        memcpy(&x, e < cnt ? rho + 8 * e : nrho, 8);
        x &= 0x7f7f7f7f7f7f7f7full;  // 0xff (h = 0 mod p) as 0x7f
        memcpy(b->r + off + e, &x, 8);
        b->mk[off + e] = e < cnt ? (uint8_t)mk[e] : 0;
    }
    b->ra[r] = make_uint4(j1, off, (unsigned)cnt | full << 16 | (top1 & 15) << 24 | (top1 >> 8 & 15) << 28, mk[0] & 0xffu);
    b->rn[r] = n;
    for (unsigned s = 0; s < ni; s++) b->it[b->ni + s] = make_uint2(r, j0 + 32 * s);
    b->ne += cnt + 1, b->nr++, b->ni += ni, b->cost += cost, b->leaves += j1 - j0;
    if (bwait) pthread_cond_signal(&bc);
    pthread_mutex_unlock(&bl);
    return 1;
}

extern "C" void gpu_leaves_done(void) {
    pthread_mutex_lock(&bl);
    bdone = 1;
    pthread_cond_signal(&bc);
    pthread_mutex_unlock(&bl);
}

extern "C" void gpu_snapshot(gpu_out *out) {
    pthread_mutex_lock(&bl);
    if (!sloop && !sfin) {
        memset(out, 0, sizeof *out);
        out->udone = ustart;
        pthread_mutex_unlock(&bl);
        return;
    }
    sreq = 1, sdone = 0;
    pthread_cond_signal(&bc);
    while (!sdone && !sfin) pthread_cond_wait(&sc, &bl);
    *out = snap;
    sreq = 0;
    pthread_mutex_unlock(&bl);
}

static int snap_wanted(void) {
    pthread_mutex_lock(&bl);
    int w = sreq && !sdone;
    pthread_mutex_unlock(&bl);
    return w;
}

// The counts of the device and of out, with the units below u0 done, to
// snap; fin: for good.
static void answer(const unsigned long long *dst, const gpu_out *out, size_t u0, int fin) {
    pthread_mutex_lock(&bl);
    snap = *out;
    snap.nodes = dst[0], snap.sumk = dst[1], snap.pairs = dst[2], snap.fhits = dst[3], snap.qpass = dst[4];
    snap.sols = nsol, snap.reps = nrep, snap.udone = u0;
    sdone = 1, sfin |= fin;
    pthread_cond_broadcast(&sc);
    pthread_mutex_unlock(&bl);
}

// The next buffer for the GPU: a full one, else the open one, else wait for
// runs; NULL once the CPU is done and every run is taken.
static batch_t *take(void) {
    batch_t *b = NULL;
    pthread_mutex_lock(&bl);
    for (;;) {
        if (nready) {
            b = bready[0], nready--;
            memmove(bready, bready + 1, nready * sizeof *bready);
            break;
        }
        if (bopen) {
            b = bopen, bopen = NULL;
            break;
        }
        if (sreq && !sdone) {
            b = &qmark;
            break;
        }
        if (bdone) break;
        bwait = 1;
        pthread_cond_wait(&bc, &bl);
        bwait = 0;
    }
    pthread_mutex_unlock(&bl);
    return b;
}

static void give_back(batch_t *b) {
    pthread_mutex_lock(&bl);
    bfree[nfree++] = b;
    __atomic_store_n(&bavail, 1, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&bl);
}

// The hits of the last launch to the host (in->hit).  Returns 0; 2 if there
// were more than HITCAP, and then passes none on (the caller undoes the
// launch and does its work in parts); 1 on another error.
static int drain(const gpu_in *in, const cparams &P, u64 *hits) {
    unsigned hc[2];
    CK(cudaMemcpy(hc, P.nhits, sizeof(unsigned), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc + 1, P.err, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (hc[1] & 1) {
        fprintf(stderr, "GPU: prime list overflow\n");
        return 1;
    }
    if (hc[1]) return 2;
    if (hc[0]) {
        unsigned zero = 0;
        CK(cudaMemcpy(hits, P.hits, (size_t)hc[0] * 5 * sizeof(u64), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(P.nhits, &zero, sizeof zero, cudaMemcpyHostToDevice));
        for (unsigned i = 0; i < hc[0]; i++) {
            int k = in->hit(hits[5 * i], hits + 5 * i + 1);
            nsol += k == 1, nrep += k == 2;
        }
    }
    return 0;
}

// Undoes a launch whose hits overflowed: the counts (st0, from before it)
// and the hit buffer.
static void undo(const cparams &P, const unsigned long long *st0) {
    unsigned zero = 0;
    CK(cudaMemcpy(P.st, st0, 5 * sizeof *st0, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(P.nhits, &zero, sizeof zero, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(P.err, &zero, sizeof zero, cudaMemcpyHostToDevice));
}

#ifdef PROF
static double PK[2][2];  // seconds in heavy() and check(), after phase2() and after leaves()
#endif

// After a phase2() (ph = 0) or leaves() launch: pass 2 of its heavy leaves,
// then its candidates.
static void finish(const cparams &P, int grid, int ph) {
    CK(cudaGetLastError());
#ifdef PROF
    cudaEvent_t ev[3];
    for (int i = 0; i < 3; i++) CK(cudaEventCreate(ev + i));
    CK(cudaEventRecord(ev[0]));
#endif
    heavy<<<grid, NT>>>();
    CK(cudaGetLastError());
#ifdef PROF
    CK(cudaEventRecord(ev[1]));
#endif
    check<<<grid, NT>>>();
    CK(cudaGetLastError());
    CK(cudaMemsetAsync(P.nh, 0, sizeof(unsigned)));
    CK(cudaMemsetAsync(P.ctrh, 0, sizeof(unsigned)));
#ifdef PROF
    CK(cudaEventRecord(ev[2]));
#endif
    CK(cudaDeviceSynchronize());
#ifdef PROF
    float a, b;
    CK(cudaEventElapsedTime(&a, ev[0], ev[1]));
    CK(cudaEventElapsedTime(&b, ev[1], ev[2]));
    PK[ph][0] += a / 1e3, PK[ph][1] += b / 1e3;
    for (int i = 0; i < 3; i++) CK(cudaEventDestroy(ev[i]));
#else
    (void)ph;
#endif
}

// One buffer of leaf runs: copied to the GPU, given back, and done (in
// halves after too many hits).  Returns 1 on an error.
static int batch(const gpu_in *in, const cparams &P, int grid, batch_t *b, u64 *hits, gpu_out *out) {
    unsigned ni = b->ni;
    unsigned long long sb[5];
    CK(cudaMemcpy((void *)P.BH, b->h, (size_t)b->ne * sizeof *b->h, cudaMemcpyHostToDevice));
    CK(cudaMemcpy((void *)P.BR, b->r, (size_t)b->ne * sizeof *b->r, cudaMemcpyHostToDevice));
    CK(cudaMemcpy((void *)P.BMK, b->mk, b->ne, cudaMemcpyHostToDevice));
    CK(cudaMemcpy((void *)P.BRA, b->ra, (size_t)b->nr * sizeof *b->ra, cudaMemcpyHostToDevice));
    CK(cudaMemcpy((void *)P.BRN, b->rn, (size_t)b->nr * sizeof *b->rn, cudaMemcpyHostToDevice));
    CK(cudaMemcpy((void *)P.BI, b->it, (size_t)ni * sizeof *b->it, cudaMemcpyHostToDevice));
    out->leaves += b->leaves, out->runs += b->nr;
    give_back(b);
    for (unsigned a = 0, span = ni; a < ni;) {  // items a.., in halves after too many hits
        unsigned e = a + span < ni ? a + span : ni;
        CK(cudaMemcpy(sb, P.st, sizeof sb, cudaMemcpyDeviceToHost));
        CK(cudaMemset(P.ctr1, 0, sizeof(unsigned)));
        leaves<<<grid, NT>>>(a, e);
        finish(P, grid, 1);
        int r = drain(in, P, hits);
        if (r == 1) return 1;
        if (r == 2) {
            if (e - a <= 1) {
                fprintf(stderr, "GPU: more than %u hits in one warp item\n", HITCAP);
                return 1;
            }
            undo(P, sb);
            span = (e - a) / 2;
        } else {
            a = e;
        }
    }
    return 0;
}

static int run(const gpu_in *in, gpu_out *out);

extern "C" int gpu_run(const gpu_in *in, gpu_out *out) {
    int r = run(in, out);
    pthread_mutex_lock(&bl);
    bdead = 1, bavail = 0;  // no more runs (on an error, the caller gives up)
    pthread_mutex_unlock(&bl);
    unsigned long long st[5] = {out->nodes, out->sumk, out->pairs, out->fhits, out->qpass};
    answer(st, out, out->udone, 1);
    CK(cudaDeviceReset());
    return r;
}

static int run(const gpu_in *in, gpu_out *out) {
    double t0 = now();
    memset(out, 0, sizeof *out);
    pthread_mutex_lock(&bl);
    ustart = in->ustart;
    pthread_mutex_unlock(&bl);
    const u64 N = in->N, B = in->B;
    const size_t NM = in->nm;

    // Prefix sums over the cofactors: exact counts per leaf, and a cost
    // estimate in units of one h step for cutting the work.
    u64 *csk = (u64 *)malloc((NM + 1) * sizeof(u64)), *csp = (u64 *)malloc((NM + 1) * sizeof(u64));
    double *cw = (double *)malloc((NM + 1) * sizeof(double));
    csk[0] = csp[0] = 0, cw[0] = 0;
    unsigned maxcnt = 0;
    for (size_t i = 0; i < NM; i++) {
        unsigned cnt = in->mrec[4 * i + 2] & 0xffff;
        u64 k = 3 * (u64)cnt + 1;
        csk[i + 1] = csk[i] + k, csp[i + 1] = csp[i] + k * (k - 1) / 2;
        cw[i + 1] = cw[i] + (cnt == 1 ? 4.0 : 2.0 + 3.0 * cnt);
        if (cnt > maxcnt) maxcnt = cnt;
    }
    if (3 * maxcnt + 1 > GMAXK) {
        fprintf(stderr, "GPU: a cofactor has %u offsets, more than %d\n", 3 * maxcnt + 1, GMAXK);
        return 1;
    }

    size_t i17 = 0;
    while (i17 < in->nop && in->op[i17] <= 13) i17++;
    const uint32_t *sp = in->op + i17;
    unsigned nsp = (unsigned)(in->nop - i17), nspw = 0;
    u64 *spm = (u64 *)malloc((nsp + 1) * sizeof(u64));
    for (unsigned i = 0; i < nsp; i++) {
        spm[i] = UINT64_MAX / sp[i];
        if (sp[i] < W / 64) nspw = i + 1;
    }
    size_t npw = (PPER + W) / 32 + 2;
    uint32_t *patb = (uint32_t *)calloc(npw, sizeof(uint32_t));
    for (u64 j = 0; j < 32 * (u64)npw; j++) {
        u64 q = 1 + 4 * j;
        if (q % 3 == 0 || q % 5 == 0 || q % 7 == 0 || q % 11 == 0 || q % 13 == 0) patb[j >> 5] |= 1u << (j & 31);
    }

    // Units.  Segments grow with q (lo/16, so a segment holds dozens of
    // primes), up to 4W.  Small q pair with many cofactors, so a heavy
    // segment is split into cofactor ranges of about TU each.
    size_t nu = 0, ucap = 1 << 16;
    unit_t *U = (unit_t *)malloc(ucap * sizeof *U);
    u64 qhi = N / 5 + 1;
    size_t L = NM;
    for (u64 lo = B / 4 * 4 + 1 + (B % 4 ? 4 : 0); lo < qhi;) {  // smallest 1 (mod 4) above B
        u64 len = lo / 16 < 4ull * W ? (lo / 16 + 4) & ~3ull : 4ull * W, hi = lo + len < qhi ? lo + len : qhi;
        while (L && in->mrec[4 * (L - 1)] > N / lo) L--;
        double np = (hi - lo) / (2 * log((double)lo)) + 1;  // primes expected
        size_t pieces = (size_t)ceil(np * cw[L] / TU), ia = 0;
        if (pieces < 1) pieces = 1;
        if (pieces > L && L) pieces = L;
        for (size_t k = 1; k <= pieces; k++) {
            size_t ib = L;
            if (k < pieces) {  // first index with cw >= the k-th share
                double t = cw[L] * k / pieces;
                size_t a = ia, b = L;
                while (a < b) {
                    size_t mid = (a + b) / 2;
                    if (cw[mid] < t) a = mid + 1;
                    else b = mid;
                }
                ib = a;
                if (ib <= ia) continue;
            }
            if (nu == ucap) {
                ucap *= 2;
                U = (unit_t *)realloc(U, ucap * sizeof *U);
            }
            U[nu] = (unit_t){lo, (uint32_t)(hi - lo), (uint32_t)ia, (uint32_t)ib, 0};
            nu++;
            ia = ib;
        }
        lo = hi;
    }

    if (in->ucheck && in->ucheck != nu) {
        fprintf(stderr, "GPU: %zu work units, but the checkpoint has %llu (another build?)\n", nu,
                (unsigned long long)in->ucheck);
        return 1;
    }
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int bps = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, phase2, NT, 0));
    int grid = bps * prop.multiProcessorCount;

    cparams P;
    P.N = N;
    P.U = upload(U, nu);
    P.M = upload((const uint4 *)in->mrec, NM);
    P.R = upload((const uint2 *)in->rho, in->nent);
    P.MK = upload(in->mk, in->nent);
    P.H = upload((const longlong2 *)in->h, in->nent);
    P.CSK = upload(csk, NM + 1), P.CSP = upload(csp, NM + 1);
    P.AB = upload(in->ab, (size_t)4160);
    P.SP = upload(sp, nsp), P.SPM = upload(spm, nsp);
    P.nsp = nsp, P.nspw = nspw;
    P.PATB = upload(patb, npw);
    CK(cudaMalloc(&P.PL, (size_t)grid * PCAP * sizeof(uint2)));
    CK(cudaMalloc(&P.hits, (size_t)HITCAP * 5 * sizeof(u64)));
    unsigned zero[2] = {0, 0};
    P.ctr = upload(zero, 1), P.nhits = upload(zero, 1), P.err = upload(zero, 1), P.ctr1 = upload(zero, 1);
    P.nh = upload(zero, 1), P.ctrh = upload(zero, 1);
    CK(cudaMalloc(&P.HQ, (size_t)HQCAP * sizeof(uint4)));
    CK(cudaMalloc(&P.nq, grid * sizeof(unsigned)));
    CK(cudaMalloc(&P.Q, (size_t)QCAP * sizeof(uint4)));
    P.qb = QCAP / grid;
    unsigned long long st0[5] = {0};
    P.st = upload(st0, 5);
    free(csk), free(csp), free(spm), free(patb);

    // The primes <= B, and the buffers of leaf runs (pinned) with their copy
    // on the GPU.
    const size_t NPR = in->np;
    uint2 *pxy = (uint2 *)malloc((NPR + 1) * sizeof(uint2));
    for (size_t j = 0; j < NPR; j++) pxy[j] = make_uint2((unsigned)in->gx[j], (unsigned)in->gy[j]);
    P.PXY = upload(pxy, NPR);
    free(pxy);
    uint4 *pq;
    CK(cudaMalloc(&pq, (2 * NPR + 1) * sizeof(uint4)));
    P.PQ = pq;
    longlong2 *dh;
    uint4 *dra;
    uint2 *dr, *dit;
    u64 *drn;
    uint8_t *dmk;
    CK(cudaMalloc(&dh, (size_t)BENT * sizeof *dh));
    CK(cudaMalloc(&dr, (size_t)BENT * sizeof *dr));
    CK(cudaMalloc(&dmk, BENT));
    CK(cudaMalloc(&dra, (size_t)BRUN * sizeof *dra));
    CK(cudaMalloc(&drn, (size_t)BRUN * sizeof *drn));
    CK(cudaMalloc(&dit, (size_t)BITEM * sizeof *dit));
    P.BH = dh, P.BR = dr, P.BMK = dmk, P.BRA = dra, P.BRN = drn, P.BI = dit;
    CK(cudaMemcpyToSymbol(C, &P, sizeof P));
    prime_tab<<<(unsigned)(NPR + 255) / 256, 256>>>((unsigned)NPR);
    CK(cudaGetLastError());
    for (int i = 0; i < NBUF; i++) {
        batch_t *b = bufs + i;
        char *p;
        CK(cudaHostAlloc((void **)&p, (size_t)BENT * 25 + (size_t)BRUN * 24 + (size_t)BITEM * 8, cudaHostAllocDefault));
        b->h = (longlong2 *)p, p += (size_t)BENT * 16;
        b->ra = (uint4 *)p, p += (size_t)BRUN * 16;
        b->r = (uint2 *)p, p += (size_t)BENT * 8;
        b->rn = (u64 *)p, p += (size_t)BRUN * 8;
        b->it = (uint2 *)p, p += (size_t)BITEM * 8;
        b->mk = (uint8_t *)p;
    }
    pthread_mutex_lock(&bl);
    for (int i = 0; i < NBUF; i++) bfree[i] = bufs + i;
    nfree = NBUF, bavail = 1, sloop = 1;
    pthread_mutex_unlock(&bl);

    printf("GPU: %s, %d SMs x %d blocks of %d threads; %zu work units\n", prop.name, prop.multiProcessorCount, bps,
           NT, nu);
    fflush(stdout);

    // The large primes first, while the CPU threads do their own leaves (the
    // buffers fill and stay full).  Each launch takes units for LAUNCH_MS,
    // then finishes the ones it has.
    // A launch with too many hits is undone and redone taking half as many
    // units, then twice as many again after each launch that fits.
    u64 *hits = (u64 *)malloc((size_t)HITCAP * 5 * sizeof(u64));
    unsigned long long sb[5];  // the counts before a launch
    double tk = 0, last = now();
    size_t u0 = in->ustart < nu ? in->ustart : nu, launches = 0, span = nu;
    out->units = nu;
    unsigned c0 = (unsigned)u0;
    CK(cudaMemcpy(P.ctr, &c0, sizeof c0, cudaMemcpyHostToDevice));
    while (u0 < nu) {
        if (snap_wanted()) {  // the buffered leaf runs first
            for (batch_t *b; (b = take()) != &qmark;)
                if (batch(in, P, grid, b, hits, out)) return 1;
            CK(cudaMemcpy(sb, P.st, sizeof sb, cudaMemcpyDeviceToHost));
            answer(sb, out, u0, 0);
        }
        double ta = now();
        unsigned cap = (unsigned)(u0 + span < nu ? u0 + span : nu), c;
        CK(cudaMemcpy(sb, P.st, sizeof sb, cudaMemcpyDeviceToHost));
        phase2<<<grid, NT>>>(cap, (u64)(LAUNCH_MS * 1e6));
        finish(P, grid, 0);
        tk += now() - ta, launches++;
        CK(cudaMemcpy(&c, P.ctr, sizeof c, cudaMemcpyDeviceToHost));
        c = c < cap ? c : cap;  // units taken past cap were not done
        int r = drain(in, P, hits);
        if (r == 1) return 1;
        if (r == 2) {
            if (c - u0 <= 1) {
                fprintf(stderr, "GPU: more than %u hits in one unit\n", HITCAP);
                return 1;
            }
            undo(P, sb);
            span = (c - u0) / 2, c = (unsigned)u0;
        } else {
            u0 = c, span = 2 * span < nu ? 2 * span : nu;
        }
        CK(cudaMemcpy(P.ctr, &c, sizeof c, cudaMemcpyHostToDevice));
        if (now() - last > 60) {
            last = now();
            printf("  progress: GPU large primes %zu/%zu units, %.0fs\n", u0, nu, last - t0);
            fflush(stdout);
        }
    }
    double t2 = now() - t0, tl = 0;
    out->udone = u0;

    // Then the leaf runs, a buffer per launch.
    for (batch_t *b; (b = take());) {
        if (b == &qmark) {
            CK(cudaMemcpy(sb, P.st, sizeof sb, cudaMemcpyDeviceToHost));
            answer(sb, out, u0, 0);
            continue;
        }
        double ta = now();
        if (batch(in, P, grid, b, hits, out)) return 1;
        tl += now() - ta;
        if (now() - last > 60) {
            last = now();
            printf("  progress: GPU leaves %llu, %.0fs\n", (unsigned long long)out->leaves, last - t0);
            fflush(stdout);
        }
    }
    unsigned long long st[5];
    CK(cudaMemcpy(st, P.st, sizeof st, cudaMemcpyDeviceToHost));
    out->nodes = st[0], out->sumk = st[1], out->pairs = st[2], out->fhits = st[3], out->qpass = st[4];
    out->secs = now() - t0, out->secs2 = t2;
    printf("GPU: large primes in %zu launches, %.2fs in kernels, done at %.2fs; then %llu leaves of the small primes "
           "(%llu runs), %.2fs in kernels; %.2fs in all\n",
           launches, tk, t2, (unsigned long long)out->leaves,
           (unsigned long long)out->runs, tl, out->secs);
#ifdef PROF
    unsigned long long prf[16];
    CK(cudaMemcpyFromSymbol(prf, PRF, sizeof prf));
    int khz;
    CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
    double cap = tk * grid * khz * 1e3;
    printf("PROF block-cycles / (kernel time x blocks x clock): sieve %.3f lattice %.3f items %.3f\n", prf[0] / cap,
           prf[1] / cap, prf[2] / cap);
    printf("PROF pass 2: %llu leaves, %llu with > %d big, %llu swept\n", prf[3], prf[4], NS, prf[5]);
    printf("PROF heavy() and check(): %.2fs %.2fs after phase2(), %.2fs %.2fs after leaves()\n", PK[0][0], PK[0][1],
           PK[1][0], PK[1][1]);
#endif
    fflush(stdout);
    free(hits), free(U), free(cw);
    return 0;
}
