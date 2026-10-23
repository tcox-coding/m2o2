// mss_gpu.h - interface of the CUDA part of mss -g (mss_gpu.cu): the
// large-prime phase, and the leaves of the small-prime search.
#ifndef MSS_GPU_H
#define MSS_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The usable primes must be 7, 11, 13, 17, 19, 29, 31, 37, in this order
// (lane and mask bit i).  Every q here is a prime above 37, so the cofactor's
// full mask and top two popcounts hold for every leaf m*q.
typedef struct {
    uint64_t N, B;
    size_t nm;             // cofactors m <= N/(B+1), sorted by m
    const uint32_t *mrec;  // per cofactor: m, off, cnt | full << 16 | t1 << 24 | t2 << 28, mask of H(m)[0]
    size_t nent;           // entries of the tables below: H(m), then m itself, per cofactor
    const uint8_t *rho;    // 8 bytes per entry: projective residues, 0xff recoded as 0x7f
    const uint8_t *mk;     // per entry: the usable primes dividing its offset
    const int64_t *h;      // per entry: re, im
    const uint32_t *ab;    // per usable prime p (blocks of p*p, in order) and x, y < p:
                           // alpha0 | beta0 << 8 | alpha2 << 16 | beta2 << 24 of X + iY = x + iy mod p
    const uint32_t *op;    // odd primes up to isqrt(N/5) + 1
    size_t nop;
    size_t np;             // primes P[j] = gx[j]^2 + gy[j]^2 = 1 (mod 4) up to B
    const int64_t *gx, *gy;
    // A candidate pair (x, y as low and high words) with n^2 + x + y a square:
    // returns 1 for a magic square, 2 for one with a repeated entry, else 0.
    int (*hit)(uint64_t n, const uint64_t xy[4]);
    uint64_t ustart;       // work units already done (a resumed run)
    uint64_t ucheck;       // if not 0: the number of work units there must be (the checkpoint's)
} gpu_in;

typedef struct {
    uint64_t nodes, sumk, pairs;  // of the large-prime centres (the CPU counts the leaves it hands over)
    uint64_t fhits, qpass;        // of both
    uint64_t sols, reps;          // hits that were magic squares, with distinct or repeated entries
    uint64_t leaves, runs;        // leaves taken from the CPU, and their runs
    uint64_t units, udone;        // work units, and how many are done (from unit 0, ustart included)
    double secs, secs2;           // in all, and until the large primes were done
} gpu_out;

// Runs on its own thread: all centres m*q, q a prime in (B, N/5], then the
// leaf runs from gpu_leaves() until gpu_leaves_done().  Returns 0 on success.
// Every count is of this run only (from unit ustart on).
int gpu_run(const gpu_in *in, gpu_out *out);

// The counts so far, at a point where every leaf run handed over has been
// done, and the units below udone and no others.  The caller makes sure no
// run is handed over meanwhile.  Waits for the GPU's current launch.
void gpu_snapshot(gpu_out *out);

// The leaves n*P[j], j0 <= j < j1, of the CPU's search (every P[j] > 37, not
// dividing n), from n's half set: cnt entries h (re, im), their projective
// residues (8 bytes each) and offset masks, then n's residue; full and top1
// as for leaf() in mss.c.  Called by the CPU threads while gpu_run() runs;
// returns 0 if the GPU cannot take them now (the caller does them).
int gpu_leaves(uint64_t n, int cnt, unsigned full, unsigned top1, const int64_t *h, const uint8_t *rho,
               const uint16_t *mk, const uint8_t *nrho, uint32_t j0, uint32_t j1);

// No more leaf runs.
void gpu_leaves_done(void);

#ifdef __cplusplus
}
#endif

#endif
