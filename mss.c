// mss.c - exhaustive search for a 3x3 magic square of nine distinct squares.
//
// Every 3x3 magic square with centre c has the form (Lucas)
//
//     c-b     c+a+b   c-a
//     c-a+b   c       c+a-b
//     c+a     c-a-b   c+b
//
// The centre of a magic square of squares is a square, c = n^2, and every line
// through the centre is then a 3-term arithmetic progression of squares
// (n^2-d, n^2, n^2+d).  Writing the outer terms as (r-s)^2 and (r+s)^2 gives
// r^2 + s^2 = n^2 and d = 2rs, so the admissible offsets for centre n^2 are
//
//     D(n) = { 2rs : r^2 + s^2 = n^2, r > s > 0 }
//
// and a magic square of distinct squares with centre n^2 exists iff there are
// a > b in D(n), a != 2b, with both a+b and a-b in D(n).
//
// If a prime p = 2 or p = 3 (mod 4) divides n, then p | r and p | s for every
// representation, so all nine entries are divisible by p^2.  A primitive
// solution therefore has n built only from primes = 1 (mod 4).  We enumerate
// exactly those n <= N and generate D(n) from Gaussian-integer factorisations,
// so no representation is missed:
//   - centres whose prime factors are all <= sqrt(N): DFS over factorisations;
//   - centres m*q with a prime q > sqrt(N): q from a segmented sieve, paired
//     with every eligible cofactor m <= N/q.
//
// Near misses are recorded too: when three of the four offsets a, b, a+b, a-b
// lie in D(n), seven of the nine cells are squares (the two non-squares sit
// symmetrically about the centre).  If one of those two is also a square the
// square has eight square entries, which is reported loudly.
//
// With -s only solutions are sought, and far fewer pairs are looked at: every
// cell of a solution is a square mod each prime, which for p = 7, 11, 13, 17,
// 19, 29, 31, 37 forces p | ab (see prune_init), so only pairs of offsets that
// are jointly divisible by all of those primes are checked.  The offsets of
// most centres are then never computed at all, only their divisibility masks.
//
// Usage:  mss N [threads]      search all centres n <= N   (N < 2^32;
//                              mss-wide, built with -DWIDE: N < 2^50)
//         mss brute N          slow independent check for n <= N
//         mss selftest         check the near-miss logic on synthetic sets
//         -q (before the rest) skips the two pair tests that only the diagonal
//         near misses (types C, D) need: about 1.9x faster, and it still finds
//         every solution and every hourglass near miss (types A, B).
//         -s                   solutions only (about 5x faster than -q)
//         -v                   also count the pairs -s would check, by plain
//                              division on the offsets (a check of -s)
//         -g                   as -s, with the large-prime phase and most leaves
//                              of the small-prime search on the GPU (mss-gpu,
//                              built with -DGPU)
//         -t TIME              stop after TIME (seconds, or e.g. 90m, 6h), with
//                              a checkpoint to resume from
//         -c SECS              checkpoint every SECS seconds (default 600; 0: only
//                              at the time limit or on SIGINT/SIGTERM)
//
// A search writes its state to <program>-<mode>-<N>.ckpt in the current
// directory (see "checkpoints" below) and resumes from that file when run
// again the same way.  results.csv there holds a row per search, updated at
// each checkpoint; finds.csv and finds.txt get every square found.

#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __SSE2__
#include <immintrin.h>
#endif
#ifdef GPU
#include "mss_gpu.h"
#endif

typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned __int128 u128;
typedef __int128 i128;

// Offsets d = 2rs <= n^2, so u64 is exact for n < 2^32.  Build with -DWIDE for
// 128-bit offsets and larger N.
#ifdef WIDE
typedef u128 dval;
#define NMAX (1ull << 50)
#else
typedef u64 dval;
#define NMAX (1ull << 32)
#endif

static const char *u128s(u128 x, char *buf) {
    char tmp[48];
    int i = 0;
    do tmp[i++] = '0' + (int)(x % 10), x /= 10;
    while (x);
    for (int j = 0; j < i; j++) buf[j] = tmp[i - 1 - j];
    buf[i] = 0;
    return buf;
}

typedef struct { i64 re, im; } gi;

// Components of the product of Gaussian integers of norms A^2 and B^2 are
// bounded by A*B <= n, so i64 never overflows here.
static inline gi gmul(gi a, gi b) {
    return (gi){a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

enum { T_A, T_B, T_C, T_D, NTYPES };
static const char *type_name[NTYPES] = {
    "a,b,a+b in D (a-b line not square)",
    "a,b,a-b in D (a+b line not square)",
    "a,a+b,a-b in D (b line not square)",
    "b,a+b,a-b in D (a line not square)",
};

typedef struct {
    u64 nodes, sumk, chk, pairs, fhits, qpass, mcov;
    u64 n7[NTYPES], neg7, rep7, eight, sols, reps;
    u64 best_n[NTYPES];
    dval best_a[NTYPES], best_b[NTYPES];
} stats_t;

#define MAXDEPTH 24
#define HCAP (1 << 15)
#define MAXE2 64
#define FBITS 10  // filter slots ~ 2^FBITS per offset: few false hits
#define FMAXB 18  // largest filter: 2^18 one-byte slots (half the L2 per core)

typedef struct {
    gi *H[MAXDEPTH];
    dval *D;
    u64 *L;      // low 64 bits of D
    uint8_t *G;  // pair filter (bit 0: D, bit 1: +-2D), all zero between calls
    int fsh;     // filter hash shift for the current centre
    dval *T;     // exact hash set of D, all zero between calls
    int *S;      // slot of each D[i] in T
    int tb;      // log2 size of T for the current centre
    int tbuilt;  // T holds the current centre's D
    uint16_t *K; // -s: prime mask of each D[i]
    uint16_t *KS;  // -s: masks in popcount order
    int *O;      // -s: index of each sorted mask
    struct pres_s *R[MAXDEPTH];  // -s: residues of H[d]
    uint16_t *HK[MAXDEPTH];      // -s: prime masks of the offsets of H[d]
    struct prho_s *RH[MAXDEPTH]; // -s: projective residues of H[d], one spare
    int gpu_ok;                  // leaf runs may go to the GPU
    stats_t st;
} ctx_t;

static u64 N;
static int selftest_mode;  // synthetic D sets: count only, never report
static int quick;          // -q: skip the two patterns only diagonal near misses need
static int solve_only;     // -s: solutions only, with prime-divisibility pruning
static int verify;         // -v: also count the pairs -s would check, straight from D
static int use_gpu;        // -g: -s with the large-prime phase on the GPU
static int gpu_on;         // the GPU is running the large-prime phase
static u64 B;          // isqrt(N): primes above B divide a centre at most once
static u64 *P;         // primes = 1 (mod 4) up to B, with P[j] = GX[j]^2 + GY[j]^2
static i64 *GX, *GY;
static size_t NP;
static uint32_t *OP;   // all odd primes up to sqrt(N/5), for the segmented sieve
static size_t NOP;

// ---------------------------------------------------------------- arithmetic

static u64 isqrt64(u64 x) {
    u64 r = (u64)sqrtl((long double)x);
    while (r && (u128)r * r > x) r--;
    while ((u128)(r + 1) * (r + 1) <= x) r++;
    return r;
}

static int is_sq128(u128 x, u64 *root) {
    u64 r = (u64)sqrtl((long double)x);
    while (r && (u128)r * r > x) r--;
    while ((u128)(r + 1) * (r + 1) <= x) r++;
    if (root) *root = r;
    return (u128)r * r == x;
}

static u64 powmod(u64 b, u64 e, u64 m) {
    u128 r = 1, x = b % m;
    for (; e; e >>= 1) {
        if (e & 1) r = r * x % m;
        x = x * x % m;
    }
    return (u64)r;
}

// p = x^2 + y^2 for prime p = 1 (mod 4), via Cornacchia.
static void two_squares(u64 p, i64 *x, i64 *y) {
    u64 t;
    for (u64 c = 2;; c++) {
        t = powmod(c, (p - 1) / 4, p);
        if ((u128)t * t % p == p - 1) break;
    }
    u64 a = p, b = t;
    while ((u128)b * b > p) {
        u64 r = a % b;
        a = b;
        b = r;
    }
    u64 s;
    if (!is_sq128(p - (u128)b * b, &s)) {
        fprintf(stderr, "Cornacchia failed for %llu\n", (unsigned long long)p);
        exit(1);
    }
    *x = (i64)b;
    *y = (i64)s;
}

static void sieve_primes(u64 L1, u64 L2) {
    // Odd-only sieve to max(L1, L2): primes = 1 (mod 4) up to L1 go to P,
    // all odd primes up to L2 go to OP.
    u64 L = L1 > L2 ? L1 : L2, nb = L / 2 + 1;
    uint8_t *comp = calloc(nb / 8 + 1, 1);
    for (u64 i = 3; i * i <= L; i += 2)
        if (!(comp[i / 2 / 8] >> (i / 2 % 8) & 1))
            for (u64 j = i * i; j <= L; j += 2 * i) comp[j / 2 / 8] |= 1 << (j / 2 % 8);
    size_t cap = 1024, ocap = 1024;
    P = malloc(cap * sizeof *P);
    OP = malloc(ocap * sizeof *OP);
    for (u64 i = 3; i <= L; i += 2) {
        if (comp[i / 2 / 8] >> (i / 2 % 8) & 1) continue;
        if (i <= L2) {
            if (NOP == ocap) OP = realloc(OP, (ocap *= 2) * sizeof *OP);
            OP[NOP++] = (uint32_t)i;
        }
        if (i % 4 == 1 && i <= L1) {
            if (NP == cap) P = realloc(P, (cap *= 2) * sizeof *P);
            P[NP++] = i;
        }
    }
    free(comp);
    GX = malloc(NP * sizeof *GX);
    GY = malloc(NP * sizeof *GY);
#pragma omp parallel for schedule(static, 4096)
    for (size_t j = 0; j < NP; j++) two_squares(P[j], &GX[j], &GY[j]);
}

// ---------------------------------------------------------------- reporting

static void print_square(FILE *f, u64 n, dval a, dval b) {
    i128 c = (i128)n * n, A = a, B = b;
    i128 v[9] = {c - B, c + A + B, c - A, c - A + B, c, c + A - B, c + A, c - A - B, c + B};
    for (int i = 0; i < 9; i++) {
        u64 r;
        if (v[i] >= 0 && is_sq128((u128)v[i], &r))
            fprintf(f, "%16llu^2", (unsigned long long)r);
        else
            fprintf(f, " %17.6Le", (long double)v[i]);
        if (i % 3 == 2) fputc('\n', f);
    }
    // Belt and braces: re-verify all eight lines.
    static const int L[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
    int ok = 1;
    for (int l = 0; l < 8; l++) ok &= v[L[l][0]] + v[L[l][1]] + v[L[l][2]] == 3 * c;
    fprintf(f, "  all 8 lines sum to 3n^2: %s\n", ok ? "yes" : "NO (BUG)");
}

static void stamp(char *buf, size_t len) {  // local time, as 2026-10-21 18:34:05
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(buf, len, "%Y-%m-%d %H:%M:%S", &tm);
}

// A find goes to stdout, finds.txt and finds.csv.  (Work redone after a
// resume reports its finds again.)
static void report(const char *what, u64 n, dval a, dval b) {
    if (selftest_mode) return;
#pragma omp critical(report)
    {
        char ba[48], bb[48], ts[32];
        FILE *outs[2] = {stdout, fopen("finds.txt", "a")};
        for (int i = 0; i < 2; i++) {
            if (!outs[i]) continue;
            fprintf(outs[i], "\n*** %s: n = %llu, a = %s, b = %s\n", what, (unsigned long long)n,
                    u128s(a, ba), u128s(b, bb));
            print_square(outs[i], n, a, b);
            fflush(outs[i]);
        }
        if (outs[1]) fclose(outs[1]);
        FILE *f = fopen("finds.csv", "a");
        if (f) {
            stamp(ts, sizeof ts);
            if (ftell(f) == 0) fprintf(f, "time,kind,n,a,b\n");
            fprintf(f, "%s,%s,%llu,%s,%s\n", ts, what, (unsigned long long)n, u128s(a, ba), u128s(b, bb));
            fclose(f);
        }
    }
}

// ---------------------------------------------------------------- analysis
//
// Three of a, b, a+b, a-b in D(n) means that some pair x < y in D has one of
//   x + y = z in D       -> (a,b) = (y,x) [a-b missing, or present: a solution]
//                           and (a,b) = (z,x), (z,y) [a+b missing]
//   x + y = 2m, m in D   -> (a,b) = (m, m-x) [b missing]
//   y - x = 2h, h in D   -> (a,b) = (x+h, h) [a missing]
// Every pair is tested against a byte filter (bit 0: D, bit 1: +-2D) indexed
// by a multiplicative hash of the low 64 bits of the value.
// Sums and differences are exact mod 2^64, so the filters have no false
// negatives, and the +-2D slots make the difference test independent of the
// order of x and y, so D is never sorted.  The O(k^2) inner loop is branch-free
// 64-bit arithmetic even for 128-bit offsets; the rare filter hits are checked
// exactly against a hash set of D, built in pair_hit() on a centre's first
// hit, since most centres have none.

#define DMAX ((dval)~(dval)0)

static inline u64 fs(u64 v, int sh) { return (v * 0x9E3779B97F4A7C15ull) >> sh; }
static inline u64 hslot(u64 v, int tb) { return ((v >> 3) * 0x9E3779B97F4A7C15ull) >> (64 - tb); }

static int inD(const ctx_t *c, u128 v) {
    if (v == 0 || v > DMAX) return 0;
    u64 msk = (1ull << c->tb) - 1;
    for (u64 h = hslot((u64)v, c->tb);; h = (h + 1) & msk) {
        if (c->T[h] == v) return 1;
        if (c->T[h] == 0) return 0;
    }
}

// Seven squares with centre n^2, offsets (a, b); m is the offset whose line
// is not a square pair.  Check whether either of its cells is a square.
static void near(ctx_t *c, int type, u64 n, dval a, dval b, u128 m) {
    stats_t *s = &c->st;
    s->n7[type]++;
    u128 sq = (u128)n * n;
    if (m >= sq) {
        s->neg7++;
    } else if (n < s->best_n[type]) {
        s->best_n[type] = n;
        s->best_a[type] = a;
        s->best_b[type] = b;
    }
    if (selftest_mode) return;
    if (is_sq128(sq + m, NULL) || (m < sq && is_sq128(sq - m, NULL))) {
        s->eight++;
        report("EIGHT SQUARES", n, a, b);
    }
}

// Exact follow-up of a filter hit on the pair D[i], D[j].  Every configuration
// is reached from exactly one pair, and every solution is counted once, from
// the x + y = z pattern.
static void build_T(ctx_t *c, const dval *D, int k) {
    u64 tmask = (1ull << c->tb) - 1;
    for (int t = 0; t < k; t++) {
        u64 h = hslot((u64)D[t], c->tb);
        while (c->T[h]) h = (h + 1) & tmask;
        c->T[h] = D[t];
        c->S[t] = (int)h;
    }
    c->tbuilt = 1;
}

static __attribute__((noinline)) void pair_hit(ctx_t *c, u64 n, const dval *D, int k, int i, int j) {
    stats_t *s = &c->st;
    const uint8_t *G = c->G;
    if (!c->tbuilt) build_T(c, D, k);
    dval x = D[i], y = D[j];
    if (x > y) x = D[j], y = D[i];
    u64 gs = G[fs((u64)x + (u64)y, c->fsh)], gd = G[fs((u64)y - (u64)x, c->fsh)];
    u128 z = (u128)x + y;
    s->fhits++;
    if ((gs & 1) && inD(c, z)) {
        if (!inD(c, y - x)) near(c, T_A, n, y, x, y - x);
        else if (y - x == x) {
            s->reps++;
            report("nine squares with a repeated entry", n, y, x);
        } else {
            s->sols++;
            report("MAGIC SQUARE OF SQUARES", n, y, x);
        }
        if (!inD(c, z + x)) near(c, T_B, n, (dval)z, x, z + x);
        if (!inD(c, z + y)) near(c, T_B, n, (dval)z, y, z + y);
    }
    dval h = (y - x) / 2, m = x + h;
    if ((gs & 2) && inD(c, m) && !inD(c, h)) near(c, T_C, n, m, h, h);
    if ((gd & 2) && inD(c, h)) {
        if (x == h) s->rep7++;
        else if (!inD(c, m)) near(c, T_D, n, m, h, m);
    }
}

static void cover_count(ctx_t *c, u64 n, const dval *D, int k);

static void analyze(ctx_t *c, u64 n, const dval *D, int k) {
    stats_t *s = &c->st;
    if (verify) cover_count(c, n, D, k);
    s->nodes++;
    s->sumk += k;
    s->pairs += (u64)k * (k - 1) / 2;
    int bl = 64 - __builtin_clzll((u64)k), fb = bl + FBITS < FMAXB ? bl + FBITS : FMAXB;
    int sh = 64 - fb;
    uint8_t *G = c->G;
    u64 *lo = c->L;
    c->fsh = sh;
    c->tb = bl + 1;
    c->tbuilt = 0;
    for (int i = 0; i < k; i++) {
        u64 v = (u64)D[i], x = v ^ (u64)((u128)D[i] >> 64);
        s->chk += x * 0x9E3779B97F4A7C15ull ^ (x >> 29);
        lo[i] = v;
        G[fs(v, sh)] |= 1;
        if (!quick) G[fs(2 * v, sh)] |= 2, G[fs(-2 * v, sh)] |= 2;
    }
    if (quick) {
        // Solutions and types A, B need only x + y in D: one probe per pair.
        for (int i = 0; i + 1 < k; i++) {
            u64 x = lo[i];
#pragma GCC unroll 4
            for (int j = i + 1; j < k; j++)
                if (__builtin_expect(G[fs(x + lo[j], sh)] != 0, 0)) pair_hit(c, n, D, k, i, j);
        }
    } else {
        for (int i = 0; i + 1 < k; i++) {
            u64 x = lo[i];
#pragma GCC unroll 4
            for (int j = i + 1; j < k; j++) {
                u64 y = lo[j], hit = (G[fs(x + y, sh)] & 3) | (G[fs(y - x, sh)] & 2);
                if (__builtin_expect(hit != 0, 0)) pair_hit(c, n, D, k, i, j);
            }
        }
    }
    for (int i = 0; i < k; i++) {
        G[fs(lo[i], sh)] = 0;
        if (!quick) G[fs(2 * lo[i], sh)] = 0, G[fs(-2 * lo[i], sh)] = 0;
    }
    if (c->tbuilt)
        for (int i = 0; i < k; i++) c->T[c->S[i]] = 0;
}

static int cmp_dval(const void *x, const void *y) {
    dval a = *(const dval *)x, b = *(const dval *)y;
    return (a > b) - (a < b);
}

static inline dval offset(gi w) { return (dval)(2 * (u64)llabs(w.re)) * (u64)llabs(w.im); }

// ---------------------------------------------------------------- solutions only (-s)
//
// All nine cells of a magic square of squares are squares modulo every prime.
// For some primes p that forces p | ab: no residues a, b != 0 (mod p) make all
// nine Lucas cells squares mod p.  prune_init() finds such primes by brute
// force over all residues (it finds 7, 11, 13, 17, 19, 29, 31, 37, and for 13
// the property also holds when 13 | n), so the pair (b, a) of a solution must
// between them be divisible by every usable prime.  Each offset gets a mask of
// those primes dividing it (p | 2rs iff p | r or p | s), and only pairs whose
// masks cover the centre's full mask are checked exactly.  A covering pair
// has a member with at least half the primes and a partner with the rest, so
// far fewer than k^2/2 pairs are visited: analyze_s() pairs offsets in
// decreasing popcount order, leaf() scans from the ones with at least half.

#define NPC 12
static int NPS;                      // usable primes
static u64 PSP[NPC], PINV[NPC], PLIM[NPC], PFM[NPC];  // PFM: fastmod32 constant
static uint8_t PZERO[NPC];           // usable also for centres divisible by p

typedef struct {
    uint16_t v[16];
} res16;

static res16 RP, RMAG;    // per lane: p and floor(2^16/p) (1 and 0 past the usable primes)
static u64 P32[NPC];      // 2^32 mod p
static unsigned PZM;      // mask of the PZERO primes

static void prune_init(void) {
    static const int cand[NPC] = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    for (int ci = 0; ci < NPC; ci++) {
        int p = cand[ci], sq[64] = {0}, ok[64], every = 1;
        for (int t = 0; t < p; t++) sq[t * t % p] = 1;
        for (int cc = 0; cc < p; cc++) {
            ok[cc] = sq[cc];
            for (int a = 1; a < p && ok[cc]; a++)
                for (int b = 1; b < p && ok[cc]; b++) {
                    int d[8] = {-b, a + b, -a, b - a, a - b, a, -a - b, b}, all = 1;
                    for (int t = 0; t < 8; t++) all &= sq[((cc + d[t]) % p + p) % p];
                    if (all) ok[cc] = 0;
                }
            if (cc && sq[cc] && !ok[cc]) every = 0;
        }
        if (!every) continue;  // only primes usable for every centre they do not divide
        u64 inv = (u64)p;
        for (int t = 0; t < 6; t++) inv *= 2 - (u64)p * inv;  // p^-1 mod 2^64
        PSP[NPS] = (u64)p, PINV[NPS] = inv, PLIM[NPS] = UINT64_MAX / (u64)p, PZERO[NPS] = (uint8_t)ok[0];
        PFM[NPS] = UINT64_MAX / (u64)p + 1;
        NPS++;
        if (NPS == 8) break;  // 8 lanes per part
    }
    for (int i = 0; i < 8; i++) {
        u64 p = i < NPS ? PSP[i] : 1;
        RP.v[i] = RP.v[8 + i] = (uint16_t)p;
        RMAG.v[i] = RMAG.v[8 + i] = (uint16_t)(i < NPS ? 65536 / p : 0);
        if (i < NPS) P32[i] = (1ull << 32) % p, PZM |= (unsigned)PZERO[i] << i;
    }
}

// Usable primes for centre n, and the primes dividing an offset 2|re*im|.
static inline unsigned fullmask(u64 n) {
    unsigned f = 0;
    for (int i = 0; i < NPS; i++)
        if (PZERO[i] || n * PINV[i] > PLIM[i]) f |= 1u << i;
    return f;
}

// -v: the pairs of D(n) whose offsets are between them divisible by every
// usable prime, by direct division.  -s must check exactly these.
static void cover_count(ctx_t *c, u64 n, const dval *D, int k) {
    unsigned full = fullmask(n);
    uint16_t *K = c->KS;
    for (int i = 0; i < k; i++) {
        unsigned m = 0;
        for (int t = 0; t < NPS; t++) m |= (unsigned)(D[i] % PSP[t] == 0) << t;
        K[i] = (uint16_t)(m & full);
    }
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++) c->st.mcov += (K[i] | K[j]) == full;
}

// ---------------------------------------------------------------- residues (-s)
//
// pres: a Gaussian integer's real and imaginary parts mod each usable prime.
// rmul() gives the residues of h*f, and the prime mask of its offset, from h's
// residues and f's vectors F1 = (Re f | Im f), F2 = (-Im f | Re f): with h's
// real part in lanes i and 8+i and its imaginary part in the other operand,
// lane i is Re(h f) and lane 8+i is Im(h f), below 2*37^2 before reduction.
//
// The leaves only need masks, and those need no arithmetic.  For h, f not 0
// mod p, p | Re(h f) iff [Re h : Im h] = [Im f : Re f] in P^1(F_p), and
// p | Im(h f) iff [Re h : Im h] = [-Re f : Im f].  So h is kept as one byte per
// prime, rho = enc([Re h : Im h]) (u/v, or p for [1 : 0], or 0xff when h = 0
// mod p, which sets the bit for every f), and f as alpha = enc([Im f : Re f]),
// beta = enc([-Re f : Im f]): each bit of the mask is two byte compares.
// Lanes past the usable primes have their bits set, later dropped by the full
// mask.

typedef struct pres_s {
    uint8_t re[8], im[8];
} pres;

// x mod p for x < 2^32 (Lemire's fastmod), and for any 64-bit x.
static inline u64 fmod32(u64 x, int i) { return (u64)(((u128)(PFM[i] * x) * PSP[i]) >> 64); }
static inline u64 fmod64(u64 x, int i) { return fmod32(fmod32(x >> 32, i) * P32[i] + fmod32(x & 0xffffffffu, i), i); }

static inline uint8_t rmodp(i64 v, int i) {
    u64 r = fmod64((u64)(v < 0 ? -v : v), i);
    return (uint8_t)(v < 0 && r ? PSP[i] - r : r);
}

static void pres_of(gi w, pres *r) {
    memset(r, 0, sizeof *r);
    for (int i = 0; i < NPS; i++) r->re[i] = rmodp(w.re, i), r->im[i] = rmodp(w.im, i);
}

static void fvec(const pres *r, res16 *F1, res16 *F2) {
    memset(F1, 0, sizeof *F1);
    memset(F2, 0, sizeof *F2);
    for (int i = 0; i < NPS; i++) {
        uint16_t p = (uint16_t)PSP[i], re = r->re[i], im = r->im[i];
        F1->v[i] = re, F1->v[8 + i] = im;
        F2->v[i] = (uint16_t)(im ? p - im : 0), F2->v[8 + i] = re;
    }
}

// Mask of the primes dividing the offset of h itself.
static unsigned hmask(const pres *h) {
    unsigned m = 0;
    for (int i = 0; i < 8; i++) m |= (unsigned)(h->re[i] == 0 || h->im[i] == 0) << i;
    return m;
}

typedef struct prho_s {
    uint8_t v[8];
} prho;

typedef struct {
    uint8_t v[4][16];  // alpha, beta of conj(pi)^2, then of pi^2; each twice
} pab;

static uint8_t RHO[8][40][40];  // enc([u : v]) mod the i-th usable prime
static uint16_t INV[8][40];      // v^-1 mod the i-th usable prime (0 for v = 0)

static u64 FULLB[256];  // byte i is 1 iff bit i is set

static void rho_init(void) {
    for (int m = 0; m < 256; m++) {
        FULLB[m] = 0;
        for (int i = 0; i < 8; i++) FULLB[m] |= (u64)(m >> i & 1) << 8 * i;
    }
    memset(RHO, 0, sizeof RHO);
    for (int i = 0; i < 8; i++) RHO[i][0][0] = 0xff;
    for (int i = 0; i < NPS; i++) {
        int p = (int)PSP[i];
        for (int v = 1; v < p; v++) {
            int w = 1;
            while (v * w % p != 1) w++;
            INV[i][v] = (uint16_t)w;
            for (int u = 0; u < p; u++) RHO[i][u][v] = (uint8_t)(u * w % p);
        }
        for (int u = 1; u < p; u++) RHO[i][u][0] = (uint8_t)p;
    }
}

static inline void rho_of(const pres *r, prho *o) {
    for (int i = 0; i < 8; i++) o->v[i] = RHO[i][r->re[i]][r->im[i]];
}

#ifdef __SSE2__
// Bytes 0xff where rho is alpha, beta or 0xff: the primes dividing the offset.
static inline __m128i rho_eq(__m128i r, __m128i A, __m128i B) {
    return _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(r, A), _mm_cmpeq_epi8(r, B)), _mm_cmpeq_epi8(r, _mm_set1_epi8(-1)));
}
#endif

// Masks of h*f for f = (a, -b) (t = 0) or (a, b) (t = 2), scalar version.
static inline unsigned rho_mask(const prho *r, const pab *ab, int t) {
    unsigned m = 0;
    for (int i = 0; i < 8; i++)
        m |= (unsigned)(r->v[i] == ab->v[t][i] || r->v[i] == ab->v[t + 1][i] || r->v[i] == 0xff) << i;
    return m;
}

#ifdef __AVX2__
static inline __m256i rlanes(const pres *h, const res16 *F1, const res16 *F2) {
    __m256i e = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)h));
    __m256i a = _mm256_permute4x64_epi64(e, 0x44), b = _mm256_permute4x64_epi64(e, 0xee);
    return _mm256_add_epi16(_mm256_mullo_epi16(a, _mm256_loadu_si256((const __m256i *)F1->v)),
                            _mm256_mullo_epi16(b, _mm256_loadu_si256((const __m256i *)F2->v)));
}

static inline unsigned zmask(__m256i z) {
    unsigned mm = (unsigned)_mm256_movemask_epi8(_mm256_packs_epi16(z, z));
    return (mm | (mm >> 16)) & 0xff;
}
#endif

static inline unsigned rmul(const pres *h, const res16 *F1, const res16 *F2, pres *out) {
#ifdef __AVX2__
    __m256i x = rlanes(h, F1, F2), p = _mm256_loadu_si256((const __m256i *)RP.v);
    __m256i r = _mm256_sub_epi16(x, _mm256_mullo_epi16(_mm256_mulhi_epu16(x, _mm256_loadu_si256((const __m256i *)RMAG.v)), p));
    r = _mm256_sub_epi16(r, _mm256_and_si256(_mm256_cmpgt_epi16(r, _mm256_sub_epi16(p, _mm256_set1_epi16(1))), p));
    __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi16(r, r), 0x08);
    _mm_storeu_si128((__m128i *)out, _mm256_castsi256_si128(pk));
    return zmask(_mm256_cmpeq_epi16(r, _mm256_setzero_si256()));
#else
    pres o;
    unsigned m = 0;
    for (int l = 0; l < 16; l++) {
        int i = l & 7;
        uint8_t v = (uint8_t)((h->re[i] * F1->v[l] + h->im[i] * F2->v[l]) % RP.v[l]);
        if (l < 8) o.re[i] = v;
        else o.im[i] = v;
        if (!v) m |= 1u << i;
    }
    *out = o;
    return m;
#endif
}

// ---------------------------------------------------------------- centres
//
// A "leaf" centre n*p, p a prime not dividing n, is analysed straight from the
// half set H(n) and f = {conj(pi)^2, p, pi^2}: D(n p) = offsets of H*f[0],
// H*f[1], H*f[2] and n*f[0] (the elements extend() would build), so H(n p) is
// never stored.  With f = NULL the offsets are those of H itself.  With -s the
// offsets are only computed for centres that have a candidate pair.
typedef struct {
    const gi *H;
    int cnt;
    u64 n;
    const gi *f;
} leaf_t;

static inline dval leaf_offset(const leaf_t *lf, int i) {
    if (!lf->f) return offset(lf->H[i]);
    int cnt = lf->cnt, t = (i >= cnt) + (i >= 2 * cnt) + (i >= 3 * cnt);
    if (t == 3) return offset((gi){lf->f[0].re * (i64)lf->n, lf->f[0].im * (i64)lf->n});
    return offset(gmul(lf->H[i - t * cnt], lf->f[t]));
}

static int leaf_offsets(const leaf_t *lf, dval *D) {
    int k = 0;
    if (!lf->f) {
        for (int h = 0; h < lf->cnt; h++) D[k++] = offset(lf->H[h]);
        return k;
    }
    for (int t = 0; t < 3; t++)
        for (int h = 0; h < lf->cnt; h++) D[k++] = offset(gmul(lf->H[h], lf->f[t]));
    D[k++] = offset((gi){lf->f[0].re * (i64)lf->n, lf->f[0].im * (i64)lf->n});
    return k;
}

// v in D(n) iff n^2 + v and n^2 - v are both squares (they are (r + s)^2 and
// (r - s)^2, and conversely r, s = (A +- B)/2 for n^2 +- v = A^2, B^2).
static int inDn(u64 n, u128 v) {
    u128 sq = (u128)n * n;
    return v && v < sq && is_sq128(sq + v, NULL) && is_sq128(sq - v, NULL);
}

// Is v < 2^104 a square?  The double square root is within 1 of the true one,
// so floor(sqrt(v)) is r or r + 1.  (Residue tests do not help here: every
// offset is 4mk(m^2 - k^2), so n^2 +- v is a square mod 64, 9 and mostly 5,
// and mod a usable prime one of the pair's offsets is 0.)
static inline int is_sq_fast(u128 v) {
    double d = (double)(u64)(v >> 64) * 18446744073709551616.0 + (double)(u64)v;
    u64 r = (u64)__builtin_sqrt(d);
    return (u128)r * r == v || (u128)(r + 1) * (r + 1) == v;
}

static __attribute__((noinline)) void sol_pair(ctx_t *c, u64 n, const leaf_t *lf, int u, int v) {
    stats_t *s = &c->st;
    s->fhits++;
    dval du = leaf_offset(lf, u), dv = leaf_offset(lf, v), x = du < dv ? du : dv, y = du < dv ? dv : du;
    // x + y in D needs n^2 + x + y to be a square, which it almost never is.
    u128 sq = (u128)n * n, a = (u128)x + y;
    if (__builtin_expect(a >= sq || !is_sq_fast(sq + a), 1)) return;
    s->qpass++;
    if (!is_sq128(sq - a, NULL) || !inDn(n, y - x)) return;
    if (y - x == x) {
        s->reps++;
        report("nine squares with a repeated entry", n, y, x);
    } else {
        s->sols++;
        report("MAGIC SQUARE OF SQUARES", n, y, x);
    }
}

// K[i] = prime mask of the i-th offset of lf, already restricted to full =
// fullmask(n).  Only the offsets of candidate pairs are computed.
static void analyze_s(ctx_t *c, u64 n, const uint16_t *K, int k, unsigned full, const leaf_t *lf) {
    stats_t *s = &c->st;
    s->nodes++;
    s->sumk += k;
    s->pairs += (u64)k * (k - 1) / 2;
    int B = __builtin_popcount(full), cnt[NPC + 2] = {0}, nge[NPC + 2], pos[NPC + 2], top1 = 0, top2 = 0;
    for (int i = 0; i < k; i++) {
        int r = __builtin_popcount(K[i]);
        cnt[r]++;
        if (r > top2) top2 = r > top1 ? top1 : r, top1 = r > top1 ? r : top1;
    }
    if (top1 + top2 < B) return;  // no two offsets can cover all the primes
    nge[B + 1] = 0;
    for (int r = B; r >= 0; r--) pos[r] = nge[r + 1], nge[r] = nge[r + 1] + cnt[r];
    for (int i = 0; i < k; i++) {
        int p = pos[__builtin_popcount(K[i])]++;
        c->KS[p] = K[i];
        c->O[p] = i;
    }
    for (int pv = 1; pv < k; pv++) {
        unsigned mv = c->KS[pv];
        int need = B - __builtin_popcount(mv), lim = need <= 0 ? pv : nge[need] < pv ? nge[need] : pv;
        for (int pu = 0; pu < lim; pu++)
            if (__builtin_expect((c->KS[pu] | mv) == full, 0)) sol_pair(c, n, lf, c->O[pu], c->O[pv]);
    }
}

// Track the two largest popcounts t1 >= t2 (branch-free: they are
// unpredictable); merge2 folds in another such pair.
static inline void top2(unsigned r, unsigned *t1, unsigned *t2) {
    unsigned lo = r < *t1 ? r : *t1;
    *t2 = lo > *t2 ? lo : *t2;
    *t1 = r > *t1 ? r : *t1;
}

static inline void merge2(unsigned a1, unsigned a2, unsigned *t1, unsigned *t2) {
    unsigned lo = a1 < *t1 ? a1 : *t1, hi2 = a2 > *t2 ? a2 : *t2;
    *t2 = lo > hi2 ? lo : hi2;
    *t1 = a1 > *t1 ? a1 : *t1;
}

static unsigned top2_of(const uint16_t *HK, int cnt, unsigned pbit, unsigned full) {
    unsigned t1 = 0, t2 = 0;
    for (int h = 0; h < cnt; h++) top2((unsigned)__builtin_popcount((HK[h] | pbit) & full), &t1, &t2);
    return t1 | t2 << 8;
}

// Offsets v (of n) whose mask K[v] contains need: bit 2v of the result, for
// the 16 from v0 on.
#ifdef __AVX2__
static inline unsigned supsets16(const uint16_t *K, int v0, int n, __m256i nv) {
    __m256i kv = _mm256_loadu_si256((const __m256i *)(K + v0));
    unsigned hit = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_and_si256(kv, nv), nv));
    return hit & (n - v0 >= 16 ? ~0u : (1u << 2 * (n - v0)) - 1) & 0x55555555u;
}
#endif

// Every pair of offsets whose masks K cover full has a member among big[]
// (at least half of the B primes), and its partner then has at least
// B - popcount of that one.  Pairs of two big ones are taken once, from the
// one with the larger index.  A big u with at most B - half primes needs a
// big partner, so only the big ones are scanned (their masks KB); any other u
// is scanned against all offsets.  K and KB are readable up to 16 entries past
// the end.
static void cover_pairs(ctx_t *c, u64 np, const leaf_t *lf, const uint16_t *K, int k, unsigned full, unsigned half,
                        const int *big, int nb) {
    unsigned B = (unsigned)__builtin_popcount(full);
    uint16_t *KB = c->KS;
    for (int b = 0; b < nb; b++) KB[b] = K[big[b]];
    for (int b = 0; b < nb; b++) {
        int u = big[b];
        unsigned need = full & ~(unsigned)K[u], only_big = B - (unsigned)__builtin_popcount(K[u]) >= half;
#ifdef __AVX2__
        __m256i nv = _mm256_set1_epi16((short)need);
        if (only_big) {
            for (int v0 = 0; v0 < nb; v0 += 16)
                for (unsigned hit = supsets16(KB, v0, nb, nv); __builtin_expect(hit != 0, 0); hit &= hit - 1) {
                    int v = big[v0 + __builtin_ctz(hit) / 2];
                    if (v < u) sol_pair(c, np, lf, u, v);
                }
            continue;
        }
        for (int v0 = 0; v0 < k; v0 += 16)
            for (unsigned hit = supsets16(K, v0, k, nv); __builtin_expect(hit != 0, 0); hit &= hit - 1) {
                int v = v0 + __builtin_ctz(hit) / 2;
                if (v != u && (v < u || (unsigned)__builtin_popcount(K[v]) < half)) sol_pair(c, np, lf, u, v);
            }
#else
        if (only_big) {
            for (int w = 0; w < nb; w++)
                if (!(need & ~(unsigned)KB[w]) && big[w] < u) sol_pair(c, np, lf, u, big[w]);
            continue;
        }
        for (int v = 0; v < k; v++)
            if (!(need & ~(unsigned)K[v]) && v != u && (v < u || (unsigned)__builtin_popcount(K[v]) < half))
                sol_pair(c, np, lf, u, v);
#endif
    }
}

// Centre n*p from H(n) and the factors f of p (alpha, beta in ab): RH and HK
// hold the projective residues and offset masks of H(n), with an entry
// readable past the last; nrho is n's; pbit is the usable-prime bit of p (the
// offset of h*p is p^2 times that of h); full = fullmask(n p).  top1: the two
// largest popcounts of that h*p block (t1 | t2 << 8), or ~0u if not known.
static void leaf(ctx_t *c, u64 np, const leaf_t *lf, const prho *RH, const uint16_t *HK, unsigned pbit,
                 const prho *nrho, const pab *ab, unsigned full, unsigned top1) {
    if (__builtin_expect(3 * lf->cnt + 1 > HCAP, 0)) {
        fprintf(stderr, "HCAP exceeded\n");
        exit(1);
    }
    if (!solve_only) {
        analyze(c, np, c->D, leaf_offsets(lf, c->D));
        return;
    }
    int cnt = lf->cnt, k = 3 * cnt + 1;
    unsigned B = (unsigned)__builtin_popcount(full), t1, t2, w0, w2;
    c->st.nodes++, c->st.sumk += k, c->st.pairs += (u64)k * (k - 1) / 2;
    if (top1 == ~0u) top1 = top2_of(HK, cnt, pbit, full);
    t1 = top1 & 0xff, t2 = top1 >> 8;
    // Pass 1: the two largest popcounts, which for most centres show that no
    // two offsets can cover all the primes.  Two h at a time, a second one
    // past the end being masked off.
#ifdef __SSE2__
    __m128i A0 = _mm_loadu_si128((const __m128i *)ab->v[0]), B0 = _mm_loadu_si128((const __m128i *)ab->v[1]);
    __m128i A2 = _mm_loadu_si128((const __m128i *)ab->v[2]), B2 = _mm_loadu_si128((const __m128i *)ab->v[3]);
    __m128i FF = _mm_set1_epi8(-1), Z = _mm_setzero_si128(), M1 = Z, M2 = Z;
    __m128i FB = _mm_set1_epi64x((long long)FULLB[full]), FB1 = _mm_cvtsi64_si128((long long)FULLB[full]);
#define RHOE(r, A, B) _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(r, A), _mm_cmpeq_epi8(r, B)), _mm_cmpeq_epi8(r, FF))
    for (int h = 0; h < cnt; h += 2) {
        __m128i r = _mm_loadu_si128((const __m128i *)(RH + h)), fb = h + 1 < cnt ? FB : FB1;
        __m128i x = _mm_or_si128(_mm_sad_epu8(_mm_and_si128(RHOE(r, A0, B0), fb), Z),
                                 _mm_slli_epi64(_mm_sad_epu8(_mm_and_si128(RHOE(r, A2, B2), fb), Z), 16));
        M2 = _mm_max_epi16(M2, _mm_min_epi16(M1, x)), M1 = _mm_max_epi16(M1, x);
    }
    // The popcounts sit in 16-bit lanes 0, 1, 4, 5: fold the halves, then 1 into 0.
    for (int sh = 0; sh < 2; sh++) {
        __m128i b1 = sh ? _mm_srli_epi64(M1, 16) : _mm_srli_si128(M1, 8);
        __m128i b2 = sh ? _mm_srli_epi64(M2, 16) : _mm_srli_si128(M2, 8);
        M2 = _mm_max_epi16(_mm_min_epi16(M1, b1), _mm_max_epi16(M2, b2)), M1 = _mm_max_epi16(M1, b1);
    }
    merge2((unsigned)_mm_extract_epi16(M1, 0), (unsigned)_mm_extract_epi16(M2, 0), &t1, &t2);
    __m128i rn = _mm_loadl_epi64((const __m128i *)nrho);
    unsigned wn = (unsigned)_mm_movemask_epi8(RHOE(rn, A0, B0)) & full;
#else
    for (int h = 0; h < cnt; h++) {
        top2((unsigned)__builtin_popcount(rho_mask(RH + h, ab, 0) & full), &t1, &t2);
        top2((unsigned)__builtin_popcount(rho_mask(RH + h, ab, 2) & full), &t1, &t2);
    }
    unsigned wn = rho_mask(nrho, ab, 0) & full;
#endif
    top2((unsigned)__builtin_popcount(wn), &t1, &t2);
    if (t1 + t2 < B) return;

    // Pass 2: the masks, and the offsets with at least half the primes, which
    // every covering pair includes.  A second h past the end has mask 0, and
    // its K slots are overwritten by the next block.
    uint16_t *K = c->K;
    int *big = c->O, nb = 0;
    unsigned half = (B + 1) / 2;
    for (int h = 0; h < cnt; h += 2) {
        unsigned keep = h + 1 < cnt ? full * 0x101 : full;
#ifdef __SSE2__
        __m128i r = _mm_loadu_si128((const __m128i *)(RH + h));
        w0 = (unsigned)_mm_movemask_epi8(RHOE(r, A0, B0)) & keep;
        w2 = (unsigned)_mm_movemask_epi8(RHOE(r, A2, B2)) & keep;
#else
        w0 = (rho_mask(RH + h, ab, 0) | rho_mask(RH + h + 1, ab, 0) << 8) & keep;
        w2 = (rho_mask(RH + h, ab, 2) | rho_mask(RH + h + 1, ab, 2) << 8) & keep;
#endif
        unsigned m[4] = {w0 & 0xff, w0 >> 8, w2 & 0xff, w2 >> 8};
        int idx[4] = {h, h + 1, 2 * cnt + h, 2 * cnt + h + 1};
        for (int t = 0; t < 4; t++) {
            K[idx[t]] = (uint16_t)m[t];
            big[nb] = idx[t], nb += (unsigned)__builtin_popcount(m[t]) >= half;
        }
    }
#ifdef __SSE2__
#undef RHOE
#endif
    for (int h = 0; h < cnt; h++) {
        unsigned m1 = (HK[h] | pbit) & full;
        K[cnt + h] = (uint16_t)m1;
        big[nb] = cnt + h, nb += (unsigned)__builtin_popcount(m1) >= half;
    }
    K[3 * cnt] = (uint16_t)wn;
    big[nb] = 3 * cnt, nb += (unsigned)__builtin_popcount(wn) >= half;
    cover_pairs(c, np, lf, K, k, full, half, big, nb);
}

// Centre n with its half set H stored (offset masks HK with -s, restricted
// here to full = fullmask(n)).
static void process(ctx_t *c, u64 n, const gi *H, uint16_t *HK, int cnt, unsigned full) {
    if (cnt < 3) return;
    if (solve_only) {
        leaf_t lf = {H, cnt, n, NULL};
        for (int i = 0; i < cnt; i++) HK[i] &= (uint16_t)full;
        analyze_s(c, n, HK, cnt, full, &lf);
    } else {
        for (int i = 0; i < cnt; i++) c->D[i] = offset(H[i]);
        analyze(c, n, c->D, cnt);
    }
}

// ---------------------------------------------------------------- enumeration
//
// For n = prod p_i^e_i, with p_i = pi_i * conj(pi_i), the Gaussian integers of
// norm n^2 are (up to units) prod pi_i^k_i conj(pi_i)^(2e_i - k_i), 0 <= k_i <=
// 2e_i.  Conjugation pairs them up except for n itself, and each conjugate
// pair gives one offset 2|re*im|.  We store one representative per pair (the
// "half set" H): multiplying in p^e gives
//     H' = H * {f_k : 0 <= k <= 2e}  U  n * {f_k : 0 <= k < e},
// where f_k = pi^k conj(pi)^(2e-k).  |H| = (prod(2e_i + 1) - 1) / 2 = |D(n)|.

static int extend(const gi *H, int cnt, u64 n, const gi *f, int e, gi *out) {
    if ((2 * e + 1) * cnt + e > HCAP) {
        fprintf(stderr, "HCAP exceeded\n");
        exit(1);
    }
    int c2 = 0;
    for (int k = 0; k <= 2 * e; k++) {
        for (int i = 0; i < cnt; i++) out[c2++] = gmul(H[i], f[k]);
        if (k < e) out[c2++] = (gi){f[k].re * (i64)n, f[k].im * (i64)n};
    }
    return c2;
}

static void factors(size_t j, int e, gi *f) {
    gi pi = {GX[j], GY[j]}, pc = {GX[j], -GY[j]}, pw[MAXE2], pcw[MAXE2];
    pw[0] = pcw[0] = (gi){1, 0};
    for (int k = 1; k <= 2 * e; k++) pw[k] = gmul(pw[k - 1], pi), pcw[k] = gmul(pcw[k - 1], pc);
    for (int k = 0; k <= 2 * e; k++) f[k] = gmul(pw[k], pcw[2 * e - k]);
}

// f = {conj(pi)^2, p, pi^2} for p = X^2 + Y^2, pi = X + iY (X, Y < 2^31).
static void prime_factors(u64 X, u64 Y, gi *f) {
    i64 re = (i64)(X * X) - (i64)(Y * Y), im = (i64)(2 * X * Y);
    f[0] = (gi){re, -im}, f[1] = (gi){(i64)(X * X + Y * Y), 0}, f[2] = (gi){re, im};
}

// lanes: X mod each usable prime, one 16-bit lane each (0 past NPS).
#ifdef __AVX2__
typedef __m128i lanes;
#else
typedef struct {
    uint16_t v[8];
} lanes;
#endif

static lanes LP2;  // p^2 in the usable lanes

static inline lanes lanes_of(u64 X) {  // X < 2^32
    uint16_t t[8] = {0};
    for (int i = 0; i < NPS; i++) t[i] = (uint16_t)fmod32(X, i);
    lanes r;
    memcpy(&r, t, sizeof r);
    return r;
}

#ifdef __AVX2__
static inline __m128i lred(__m128i v) {  // v mod p for v < 2^16 (see rmul)
    __m128i p = _mm_loadu_si128((const __m128i *)RP.v), mag = _mm_loadu_si128((const __m128i *)RMAG.v);
    __m128i r = _mm_sub_epi16(v, _mm_mullo_epi16(_mm_mulhi_epu16(v, mag), p));
    return _mm_min_epu16(r, _mm_sub_epi16(r, p));
}


// The residue vectors of prime_factors() from x = X, y = Y mod each usable
// prime; returns fullmask(X^2 + Y^2).
static unsigned prime_vectors(lanes x, lanes y, res16 *F1, res16 *F2) {
    __m128i x2 = _mm_mullo_epi16(x, x), y2 = _mm_mullo_epi16(y, y), xy = _mm_mullo_epi16(x, y), z = _mm_setzero_si128();
    __m128i a = lred(_mm_sub_epi16(_mm_add_epi16(x2, LP2), y2)), b = lred(_mm_add_epi16(xy, xy));
    __m128i nb = lred(_mm_sub_epi16(_mm_loadu_si128((const __m128i *)RP.v), b)), c = lred(_mm_add_epi16(x2, y2));
#define PUT(F, lo, hi) _mm_storeu_si128((__m128i *)(F).v, lo), _mm_storeu_si128((__m128i *)((F).v + 8), hi)
    PUT(F1[0], a, nb), PUT(F2[0], b, a);
    PUT(F1[1], c, z), PUT(F2[1], z, c);
    PUT(F1[2], a, b), PUT(F2[2], nb, a);
#undef PUT
    unsigned zm = (unsigned)_mm_movemask_epi8(_mm_packs_epi16(_mm_cmpeq_epi16(c, z), z)) & 0xff;
    return (PZM | ~zm) & ((1u << NPS) - 1);
}

// alpha, beta of the factors of X^2 + Y^2 from x = X, y = Y mod each usable
// prime (as make_ab: alpha2 = b/a, alpha0 = -b/a, beta0 = a/b, beta2 = -a/b,
// or p for a zero denominator); returns fullmask(X^2 + Y^2).
static unsigned prime_ab(lanes x, lanes y, pab *ab) {
    __m128i x2 = _mm_mullo_epi16(x, x), y2 = _mm_mullo_epi16(y, y), xy = _mm_mullo_epi16(x, y), z = _mm_setzero_si128();
    __m128i p = _mm_loadu_si128((const __m128i *)RP.v);
    __m128i a = lred(_mm_sub_epi16(_mm_add_epi16(x2, LP2), y2)), b = lred(_mm_add_epi16(xy, xy));
    __m128i c = lred(_mm_add_epi16(x2, y2));
    uint16_t at[8], bt[8], ai[8], bi[8];
    _mm_storeu_si128((__m128i *)at, a);
    _mm_storeu_si128((__m128i *)bt, b);
    for (int i = 0; i < 8; i++) ai[i] = INV[i][at[i]], bi[i] = INV[i][bt[i]];
    __m128i za = _mm_cmpeq_epi16(a, z), zb = _mm_cmpeq_epi16(b, z), pa = _mm_and_si128(za, p), pb = _mm_and_si128(zb, p);
    __m128i a2 = lred(_mm_mullo_epi16(b, _mm_loadu_si128((const __m128i *)ai)));
    __m128i b0 = lred(_mm_mullo_epi16(a, _mm_loadu_si128((const __m128i *)bi)));
    __m128i a0 = _mm_or_si128(_mm_andnot_si128(za, lred(_mm_sub_epi16(p, a2))), pa);
    __m128i b2 = _mm_or_si128(_mm_andnot_si128(zb, lred(_mm_sub_epi16(p, b0))), pb);
    a2 = _mm_or_si128(_mm_andnot_si128(za, a2), pa), b0 = _mm_or_si128(_mm_andnot_si128(zb, b0), pb);
    _mm_storeu_si128((__m128i *)ab->v[0], _mm_packus_epi16(a0, a0));
    _mm_storeu_si128((__m128i *)ab->v[1], _mm_packus_epi16(b0, b0));
    _mm_storeu_si128((__m128i *)ab->v[2], _mm_packus_epi16(a2, a2));
    _mm_storeu_si128((__m128i *)ab->v[3], _mm_packus_epi16(b2, b2));
    unsigned zm = (unsigned)_mm_movemask_epi8(_mm_packs_epi16(_mm_cmpeq_epi16(c, z), z)) & 0xff;
    return (PZM | ~zm) & ((1u << NPS) - 1);
}
#else
// alpha, beta of (a, -b) and (a, b) from a, b mod each usable prime; zero:
// the primes dividing c, for the full mask.
static unsigned make_ab(const uint16_t *a, const uint16_t *b, const uint16_t *c, pab *ab) {
    memset(ab, 0, sizeof *ab);
    unsigned zm = 0;
    for (int i = 0; i < NPS; i++) {
        unsigned p = (unsigned)PSP[i], na = a[i] ? p - a[i] : 0, nb = b[i] ? p - b[i] : 0;
        ab->v[0][i] = RHO[i][nb][a[i]], ab->v[1][i] = RHO[i][a[i]][b[i]];
        ab->v[2][i] = RHO[i][b[i]][a[i]], ab->v[3][i] = RHO[i][na][b[i]];
        zm |= (unsigned)(c[i] == 0) << i;
    }
    for (int t = 0; t < 4; t++) memcpy(ab->v[t] + 8, ab->v[t], 8);
    return (PZM | ~zm) & ((1u << NPS) - 1);
}


static unsigned prime_vectors(lanes x, lanes y, res16 *F1, res16 *F2) {
    memset(F1, 0, 3 * sizeof *F1);
    memset(F2, 0, 3 * sizeof *F2);
    unsigned zm = 0;
    for (int i = 0; i < NPS; i++) {
        unsigned p = (unsigned)PSP[i], xi = x.v[i], yi = y.v[i], a = (xi * xi + p * p - yi * yi) % p;
        unsigned b = 2 * xi * yi % p, nb = (p - b) % p, cc = (xi * xi + yi * yi) % p;
        F1[0].v[i] = a, F1[0].v[8 + i] = nb, F2[0].v[i] = b, F2[0].v[8 + i] = a;
        F1[1].v[i] = cc, F2[1].v[8 + i] = cc;
        F1[2].v[i] = a, F1[2].v[8 + i] = b, F2[2].v[i] = nb, F2[2].v[8 + i] = a;
        zm |= (unsigned)(cc == 0) << i;
    }
    return (PZM | ~zm) & ((1u << NPS) - 1);
}

static unsigned prime_ab(lanes x, lanes y, pab *ab) {
    uint16_t a[8] = {0}, b[8] = {0}, c[8] = {0};
    for (int i = 0; i < NPS; i++) {
        unsigned p = (unsigned)PSP[i], xi = x.v[i], yi = y.v[i];
        a[i] = (uint16_t)((xi * xi + p * p - yi * yi) % p), b[i] = (uint16_t)(2 * xi * yi % p);
        c[i] = (uint16_t)((xi * xi + yi * yi) % p);
    }
    return make_ab(a, b, c, ab);
}
#endif

static gi (*PF)[3];          // per prime <= B: conj(pi)^2, p, pi^2
static res16 (*PF1)[3], (*PF2)[3];
static pab *PAB;
static uint16_t *PDROP;      // per prime <= B: usable-prime bit lost when p | n
static uint16_t *PBIT;       // per prime <= B: its usable-prime bit (or 0)

static void prime_tables(void) {
    PF = malloc(NP * sizeof *PF);
    PF1 = malloc(NP * sizeof *PF1);
    PF2 = malloc(NP * sizeof *PF2);
    PAB = malloc(NP * sizeof *PAB);
    rho_init();
    PDROP = calloc(NP, sizeof *PDROP);
    PBIT = calloc(NP, sizeof *PBIT);
    uint16_t p2[8] = {0};
    for (int i = 0; i < NPS; i++) p2[i] = (uint16_t)(PSP[i] * PSP[i]);
    memcpy(&LP2, p2, sizeof p2);
    for (size_t j = 0; j < NP; j++) {
        prime_factors((u64)GX[j], (u64)GY[j], PF[j]);
        if (solve_only) {
            lanes x = lanes_of((u64)GX[j]), y = lanes_of((u64)GY[j]);
            prime_vectors(x, y, PF1[j], PF2[j]);
            prime_ab(x, y, &PAB[j]);
        }
        for (int i = 0; i < NPS; i++)
            if (PSP[i] == P[j]) PBIT[j] = (uint16_t)(1u << i), PDROP[j] = PZERO[i] ? 0 : PBIT[j];
    }
}

// H(n p^e) at depth + 1 from H(n) at depth; with -s also its residues and the
// offset masks and projective residues (c->HK, c->RH).  F1, F2: vectors of f[0..2e].
static int extend_r(ctx_t *c, int depth, int cnt, u64 n, const pres *nres, const gi *f, const res16 *F1,
                    const res16 *F2, int e) {
    int c2 = extend(c->H[depth], cnt, n, f, e, c->H[depth + 1]);
    if (solve_only) {
        const pres *R = c->R[depth];
        pres *R2 = c->R[depth + 1];
        uint16_t *K2 = c->HK[depth + 1];
        for (int t = 0, k = 0; t <= 2 * e; t++) {
            for (int i = 0; i < cnt; i++, k++) K2[k] = (uint16_t)rmul(R + i, F1 + t, F2 + t, R2 + k);
            if (t < e) K2[k] = (uint16_t)rmul(nres, F1 + t, F2 + t, R2 + k), k++;
        }
        for (int k = 0; k < c2; k++) rho_of(R2 + k, c->RH[depth + 1] + k);
    }
    return c2;
}

// Factors of p^e for e >= 2, with their vectors (-s).
static void factors_r(size_t j, int e, gi *f, res16 *F1, res16 *F2) {
    factors(j, e, f);
    if (!solve_only) return;
    for (int k = 0; k <= 2 * e; k++) {
        pres r;
        pres_of(f[k], &r);
        fvec(&r, &F1[k], &F2[k]);
    }
}

#ifdef GPU
// The leaves n*P[j] from j0 on, up to P[j] > N/n or jhi, to the GPU (counted
// here, as leaf() would): returns the end of the run, or 0 if not taken.
static size_t gpu_leaf_run(ctx_t *c, int depth, u64 n, int cnt, unsigned full, const prho *nrho, unsigned top1,
                           size_t j0, size_t jhi) {
    u64 lim = N / n;
    size_t a = j0 + 1, b = jhi;
    while (a < b) {
        size_t mid = a + (b - a) / 2;
        if (P[mid] <= lim) a = mid + 1;
        else b = mid;
    }
    if (!gpu_leaves(n, cnt, full, top1, (const int64_t *)c->H[depth], (const uint8_t *)c->RH[depth], c->HK[depth],
                    nrho->v, (uint32_t)j0, (uint32_t)a))
        return 0;
    u64 L = a - j0, k = 3 * (u64)cnt + 1;
    c->st.nodes += L, c->st.sumk += L * k, c->st.pairs += L * (k * (k - 1) / 2);
    return a;
}
#endif

// All centres n * (primes P[jlo..jhi) and their powers) * (larger primes),
// from H(n) at depth (residues c->R[depth], nres for n; full = fullmask(n)).
// The leaves (P[j] P[j+1] > N/n) come last, and only the first of them can
// have a power; with the GPU on, those with P[j] > 37 go there in one run.
static void dfs(ctx_t *c, int depth, u64 n, int cnt, size_t jlo, size_t jhi, unsigned full, pres nres) {
    const gi *H = c->H[depth];
    u64 lim = N / n;
    prho nrho;
    rho_of(&nres, &nrho);
    unsigned top1 = solve_only ? top2_of(c->HK[depth], cnt, 0, full) : 0;  // for the leaves with PBIT[j] == 0
#ifdef GPU
    int gtry = c->gpu_ok;  // offer the run once
#endif
    for (size_t j = jlo; j < jhi && P[j] <= lim; j++) {
        u64 p = P[j], m = n * p;
        unsigned fm = full & ~(unsigned)PDROP[j];
        size_t jg = 0;  // the leaves P[j..jg) went to the GPU
        pres mres = nres;
        if (solve_only) rmul(&nres, &PF1[j][1], &PF2[j][1], &mres);
        if (j + 1 < NP && P[j + 1] <= N / m) {
            int c2 = extend_r(c, depth, cnt, n, &nres, PF[j], PF1[j], PF2[j], 1);
            process(c, m, c->H[depth + 1], c->HK[depth + 1], c2, fm);
            dfs(c, depth + 1, m, c2, j + 1, NP, fm, mres);
        } else if (cnt) {
#ifdef GPU
            if (gtry && p > 37) gtry = 0, jg = gpu_leaf_run(c, depth, n, cnt, full, &nrho, top1, j, jhi);
#endif
            if (!jg) {
                leaf_t lf = {H, cnt, n, PF[j]};
                leaf(c, m, &lf, c->RH[depth], c->HK[depth], PBIT[j], &nrho, &PAB[j], fm, PBIT[j] ? ~0u : top1);
            }
        }
        gi f[MAXE2];
        res16 F1[MAXE2], F2[MAXE2];
        for (int e = 2; p <= N / m; e++) {
            m *= p;
            factors_r(j, e, f, F1, F2);
            if (solve_only) rmul(&mres, &PF1[j][1], &PF2[j][1], &mres);
            int c2 = extend_r(c, depth, cnt, n, &nres, f, F1, F2, e);
            process(c, m, c->H[depth + 1], c->HK[depth + 1], c2, fm);
            if (j + 1 < NP && P[j + 1] <= N / m) dfs(c, depth + 1, m, c2, j + 1, NP, fm, mres);
        }
        if (jg) break;  // the rest of the loop's primes went to the GPU
    }
}

// ---------------------------------------------------------------- tasks
//
// A serial pre-pass walks the top of the tree.  Children whose subtree is big
// are expanded further; runs of small children are grouped into a task.
// Tasks are then run in parallel with dynamic scheduling.  The subtree of a
// child n' = n*p has about N/n' centres, and their pair searches grow with
// the square of their offset counts, so it is weighed as (N/n') d(n'^2)^2,
// with d(n'^2) = 3 d(n^2) the count for n' itself.  (Unweighted, a task under
// a node with many prime factors could run for minutes, and checkpoints wait
// for the tasks in hand.)

typedef struct {
    u64 n;
    size_t jlo, jhi;  // jhi == 0: just process node n itself
    int depth;
    uint32_t pj[MAXDEPTH];
    uint8_t pe[MAXDEPTH];
} task_t;

static task_t *tasks;
static size_t ntasks, tcap;
static u64 THR;

static void emit(const task_t *t) {
    if (ntasks == tcap) tasks = realloc(tasks, (tcap = tcap ? 2 * tcap : 1024) * sizeof *tasks);
    tasks[ntasks++] = *t;
}

static void pre(task_t *t, size_t jstart) {
    u64 n = t->n, lim = N / n, f = 3;
    for (int d = 0; d < t->depth; d++) f *= 2 * t->pe[d] + 1;
    u128 g = (u128)f * f;  // the weight of a child
    size_t j = jstart;
    while (j < NP && P[j] <= lim) {
        u64 p = P[j];
        if (lim / p * g >= THR) {
            u64 m = n;
            for (int e = 1; p <= N / m; e++) {
                m *= p;
                task_t c = *t;
                c.n = m;
                c.pj[c.depth] = (uint32_t)j;
                c.pe[c.depth] = (uint8_t)e;
                c.depth++;
                c.jlo = c.jhi = 0;
                emit(&c);
                pre(&c, j + 1);
            }
            j++;
        } else {
            task_t c = *t;
            c.jlo = j;
            for (u128 w = 0; j < NP && P[j] <= lim && w < THR; j++) w += lim / P[j] * g;
            c.jhi = j;
            emit(&c);
        }
    }
}

static void run_task(ctx_t *c, const task_t *t) {
    int cnt = 0;
    u64 n = 1;
    unsigned full = fullmask(1);
    pres nres;
    gi f[MAXE2];
    res16 F1[MAXE2], F2[MAXE2];
    pres_of((gi){1, 0}, &nres);
    for (int d = 0; d < t->depth; d++) {
        size_t j = t->pj[d];
        factors_r(j, t->pe[d], f, F1, F2);
        cnt = extend_r(c, d, cnt, n, &nres, f, F1, F2, t->pe[d]);
        for (int e = 0; e < t->pe[d]; e++) {
            n *= P[j];
            if (solve_only) rmul(&nres, &PF1[j][1], &PF2[j][1], &nres);
        }
        full &= ~(unsigned)PDROP[j];
    }
    if (n != t->n) {
        fprintf(stderr, "task rebuild mismatch\n");
        exit(1);
    }
    if (t->jhi == 0) process(c, n, c->H[t->depth], c->HK[t->depth], cnt, full);
    else dfs(c, t->depth, n, cnt, t->jlo, t->jhi, full, nres);
}

// ---------------------------------------------------------------- large primes
//
// A centre n <= N has at most one prime factor q > B = isqrt(N), to the first
// power, and then n = m*q with m <= N/q < B.  The DFS above only uses primes
// <= B; centres with such a q are handled here instead, so no prime table up
// to N/5 is needed.  q runs over a segmented sieve, and q = X^2 + Y^2 comes
// straight from enumerating the lattice points of each block's annulus.  Each
// q is paired with every eligible cofactor m <= N/q, whose half sets H(m) are
// precomputed.

typedef struct {
    u64 m;
    int cnt;
    size_t off;      // into MH, MK and MRHO, which hold cnt + 1 entries: H(m), then m
    unsigned full;   // fullmask(m)
    unsigned top1;   // the two largest popcounts of the offset masks of H(m) (-s)
} mrec_t;

static mrec_t *M;
static size_t NM, Mcap;
static gi *MH;
static prho *MRHO;    // projective residues (-s)
static uint16_t *MK;  // offset masks (-s)
static size_t MHn, MHcap;

#ifndef SEG
#define SEG (1u << 21)  // numbers per sieve block (SEG/4 bytes: only q = 1 mod 4 is kept)
#endif
#define PPER 15015      // 3*5*7*11*13

static uint8_t *PAT;  // PAT[j]: 1 + 4j has a factor 3..13, for j < PPER + SEG/4
static double *OPI;   // 1 / OP[i]

static void pat_init(void) {
    OPI = malloc((NOP + 1) * sizeof *OPI);
    for (size_t i = 0; i < NOP; i++) OPI[i] = 1.0 / OP[i];
    PAT = malloc(PPER + SEG / 4);
    for (u64 j = 0; j < PPER + SEG / 4; j++) {
        u64 q = 1 + 4 * j;
        PAT[j] = q % 3 == 0 || q % 5 == 0 || q % 7 == 0 || q % 11 == 0 || q % 13 == 0;
    }
}

static void gen_m(ctx_t *c, int depth, u64 n, int cnt, size_t j, u64 lim) {
    for (; j < NP && P[j] <= lim / n; j++) {
        u64 m = n;
        gi f[MAXE2];
        for (int e = 1; P[j] <= lim / m; e++) {
            m *= P[j];
            factors(j, e, f);
            int c2 = extend(c->H[depth], cnt, n, f, e, c->H[depth + 1]);
            if (NM == Mcap) M = realloc(M, (Mcap = Mcap ? 2 * Mcap : 1024) * sizeof *M);
            while (MHn + c2 + 1 > MHcap) {
                MHcap = MHcap ? 2 * MHcap : 4096;
                MH = realloc(MH, MHcap * sizeof *MH);
                MRHO = realloc(MRHO, MHcap * sizeof *MRHO);
                MK = realloc(MK, MHcap * sizeof *MK);
            }
            memcpy(MH + MHn, c->H[depth + 1], c2 * sizeof *MH);
            if (solve_only)
                for (int h = 0; h <= c2; h++) {
                    pres r;
                    pres_of(h < c2 ? c->H[depth + 1][h] : (gi){(i64)m, 0}, &r);
                    rho_of(&r, MRHO + MHn + h);
                    MK[MHn + h] = (uint16_t)hmask(&r);
                }
            M[NM++] = (mrec_t){m, c2, MHn, fullmask(m), solve_only ? top2_of(MK + MHn, c2, 0, fullmask(m)) : 0};
            MHn += c2 + 1;
            gen_m(c, depth + 1, m, c2, j + 1, lim);
        }
    }
}

static int cmp_mrec(const void *x, const void *y) {
    u64 a = ((const mrec_t *)x)->m, b = ((const mrec_t *)y)->m;
    return (a > b) - (a < b);
}

// All centres m*q with q prime in [lo, hi), lo = 1 (mod 4), hi <= N/5 + 1.
// Every q = X^2 + Y^2 with X + Y odd is 1 (mod 4), so the sieve only keeps
// those: comp[i] is for lo + 4i.  Multiples of 3..13 come from PAT.
static void run_block(ctx_t *c, u64 lo, u64 hi, uint8_t *comp) {
    u64 len = (hi - lo + 3) / 4;
    size_t i0 = 0;
    if (lo > 13) {
        memcpy(comp, PAT + (lo / 4) % PPER, len);
        while (i0 < NOP && OP[i0] <= 13) i0++;
    } else {
        memset(comp, 0, len);
    }
    for (size_t i = i0; i < NOP && (u64)OP[i] * OP[i] < hi; i++) {
        // lo mod p from the reciprocal: the quotient is off by at most 1.
        u64 p = OP[i];
        i64 r = (i64)(lo - (u64)((double)lo * OPI[i]) * p);
        r += r < 0 ? (i64)p : r >= (i64)p ? -(i64)p : 0;
        u64 s = lo + (r ? p - (u64)r : 0);
        if (s < p * p) s = p * p;
        while ((s & 3) != 1) s += p;
        for (; s < hi; s += 4 * p) comp[(s - lo) >> 2] = 1;
    }
    // Rows X of the annulus: yl is the least y >= 0 with X^2 + y^2 >= lo, yh
    // the largest with X^2 + y^2 < hi.  Both only fall as X grows, so they are
    // updated in place: for large N a block is thinner than one row, and most
    // rows hold no point at all.
    u64 X = isqrt64(lo / 2), yl = isqrt64(lo - X * X), yh = isqrt64(hi - 1 - X * X);
    if (yl * yl < lo - X * X) yl++;
    for (; X * X + 1 < hi; X++) {
        u64 X2 = X * X;
        while (yl && X2 + (yl - 1) * (yl - 1) >= lo) yl--;
        while (X2 + yh * yh >= hi) yh--;
        u64 ylo = yl ? yl : 1, yhi = yh < X ? yh : X - 1;
        if (!((X + ylo) & 1)) ylo++;
        if (ylo > yhi) continue;
        lanes xr;
        int xready = 0;
        for (u64 Y = ylo; Y <= yhi; Y += 2) {
            u64 q = X2 + Y * Y;
            if (comp[(q - lo) >> 2]) continue;
            gi f[3];
            pab ab;
            prime_factors(X, Y, f);
            u64 lim = (u64)((double)N / (double)q);  // N/q (within 1, so corrected)
            if (lim * q > N) lim--;
            else if ((lim + 1) * q <= N) lim++;
            unsigned fq = 0, qbit = 0;
            if (solve_only) {
                if (!xready) xr = lanes_of(X), xready = 1;
                fq = prime_ab(xr, lanes_of(Y), &ab);
            }
            if (q <= 37)
                for (int t = 0; t < NPS; t++) qbit |= (unsigned)(PSP[t] == q) << t;
            int plain = fq == (1u << NPS) - 1 && !qbit;  // so the cofactors' top1 holds
#ifdef __SSE2__
            __m128i A0 = _mm_loadu_si128((const __m128i *)ab.v[0]), B0 = _mm_loadu_si128((const __m128i *)ab.v[1]);
            __m128i A2 = _mm_loadu_si128((const __m128i *)ab.v[2]), B2 = _mm_loadu_si128((const __m128i *)ab.v[3]);
#endif
            for (size_t i = 0; i < NM && M[i].m <= lim; i++) {
                leaf_t lf = {MH + M[i].off, M[i].cnt, M[i].m, f};
                const prho *R = MRHO + M[i].off;
#ifdef __SSE2__
                // Prime m (the most common case): one load holds h and m, whose
                // compares give all four masks, and the six pairs are tested
                // directly.  Offsets: h f0, h q, h f2, m f0.
                if (M[i].cnt == 1 && plain && solve_only) {
                    __m128i r = _mm_loadu_si128((const __m128i *)R);
                    unsigned full = M[i].full, w0 = (unsigned)_mm_movemask_epi8(rho_eq(r, A0, B0));
                    unsigned m0 = w0 & full, m3 = w0 >> 8 & full, m1 = MK[M[i].off] & full;
                    unsigned m2 = (unsigned)_mm_movemask_epi8(rho_eq(r, A2, B2)) & full;
                    c->st.nodes++, c->st.sumk += 4, c->st.pairs += 6;
                    if (__builtin_expect(((m0 | m1) == full) | ((m0 | m2) == full) | ((m0 | m3) == full) |
                                             ((m1 | m2) == full) | ((m1 | m3) == full) | ((m2 | m3) == full),
                                         0)) {
                        unsigned mk[4] = {m0, m1, m2, m3};
                        for (int u = 0; u < 4; u++)
                            for (int v = u + 1; v < 4; v++)
                                if ((mk[u] | mk[v]) == full) sol_pair(c, M[i].m * q, &lf, u, v);
                    }
                    continue;
                }
#endif
                leaf(c, M[i].m * q, &lf, R, MK + M[i].off, qbit, R + M[i].cnt, &ab, M[i].full & fq,
                     plain ? M[i].top1 : ~0u);
            }
        }
    }
}

#ifdef GPU
// The large-prime phase on the GPU (mss_gpu.cu), run by its own host thread
// while the CPU threads take the small primes; then the leaves the CPU threads
// hand over (dfs()).  The GPU finds the candidate pairs with n^2 + a + b a
// square; gpu_hit() finishes sol_pair()'s check.
static gpu_out gres;
static int gfail;
static u64 gustart, gucheck;  // units done before, and how many there are (a resumed run)

static int gpu_hit(uint64_t n, const uint64_t xy[4]) {
    u128 x = (u128)xy[1] << 64 | xy[0], y = (u128)xy[3] << 64 | xy[2], sq = (u128)n * n, a = x + y;
    if (!is_sq128(sq - a, NULL) || !inDn(n, y - x)) return 0;
    if (y - x == x) {
        report("nine squares with a repeated entry", n, (dval)y, (dval)x);
        return 2;
    }
    report("MAGIC SQUARE OF SQUARES", n, (dval)y, (dval)x);
    return 1;
}

static const int GPU_PRIMES[8] = {7, 11, 13, 17, 19, 29, 31, 37};

static int gpu_usable(void) {
    if (NPS != 8 || B < 38) return 0;  // every q > 37: its own mask bit is never set
    for (int i = 0; i < 8; i++)
        if (PSP[i] != (u64)GPU_PRIMES[i]) return 0;
    return 1;
}

static void *gpu_thread(void *arg) {
    (void)arg;
    uint32_t *mrec = malloc(NM * 4 * sizeof *mrec), *ab = malloc(4160 * sizeof *ab);
    uint8_t *rho = malloc(MHn * 8), *mk = malloc(MHn);
    for (size_t i = 0; i < NM; i++) {
        uint32_t *r = mrec + 4 * i;
        r[0] = (uint32_t)M[i].m, r[1] = (uint32_t)M[i].off;
        r[2] = (uint32_t)M[i].cnt | M[i].full << 16 | (M[i].top1 & 15) << 24 | (M[i].top1 >> 8) << 28;
        r[3] = MK[M[i].off];
    }
    for (size_t e = 0; e < MHn; e++) {
        for (int b = 0; b < 8; b++) rho[8 * e + b] = MRHO[e].v[b] == 0xff ? 0x7f : MRHO[e].v[b];
        mk[e] = (uint8_t)MK[e];
    }
    // alpha, beta of X + iY from x = X, y = Y mod p, as make_ab().
    for (int i = 0, o = 0; i < 8; o += GPU_PRIMES[i] * GPU_PRIMES[i], i++) {
        int p = GPU_PRIMES[i];
        for (int x = 0; x < p; x++)
            for (int y = 0; y < p; y++) {
                int a = ((x * x - y * y) % p + p) % p, b = 2 * x * y % p, na = (p - a) % p, nb = (p - b) % p;
                ab[o + x * p + y] = (uint32_t)RHO[i][nb][a] | (uint32_t)RHO[i][a][b] << 8 |
                                    (uint32_t)RHO[i][b][a] << 16 | (uint32_t)RHO[i][na][b] << 24;
            }
    }
    gpu_in in = {N, B, NM, mrec, MHn, rho, mk, (const int64_t *)MH, ab, OP, NOP, NP, GX, GY, gpu_hit, gustart, gucheck};
    gfail = gpu_run(&in, &gres);
    if (gfail) {  // the counts would be incomplete: stop now rather than after the CPU threads
        fprintf(stderr, "GPU phase failed: stopping\n");
        exit(1);
    }
    free(mrec), free(ab), free(rho), free(mk);
    return NULL;
}
#endif

static void ctx_init(ctx_t *c) {
    memset(c, 0, sizeof *c);
    for (int d = 0; d < MAXDEPTH; d++) c->H[d] = malloc(HCAP * sizeof(gi));
    c->D = malloc(HCAP * sizeof(dval));
    c->L = malloc(HCAP * sizeof(u64));
    c->G = calloc(1ull << FMAXB, 1);
    c->T = calloc(4 * HCAP, sizeof(dval));
    c->S = malloc(HCAP * sizeof(int));
    c->K = calloc(HCAP + 16, sizeof(uint16_t));
    c->KS = calloc(HCAP + 16, sizeof(uint16_t));
    c->O = malloc(HCAP * sizeof(int));
    for (int d = 0; d < MAXDEPTH; d++) c->R[d] = malloc(HCAP * sizeof(pres)), c->HK[d] = malloc(HCAP * sizeof(uint16_t)),
                                        c->RH[d] = calloc(HCAP + 1, sizeof(prho));
    for (int t = 0; t < NTYPES; t++) c->st.best_n[t] = UINT64_MAX;
}

static void merge(stats_t *a, const stats_t *b) {
    a->nodes += b->nodes, a->sumk += b->sumk, a->chk += b->chk, a->pairs += b->pairs, a->fhits += b->fhits;
    a->mcov += b->mcov, a->qpass += b->qpass;
    a->neg7 += b->neg7, a->rep7 += b->rep7, a->eight += b->eight;
    a->sols += b->sols, a->reps += b->reps;
    for (int t = 0; t < NTYPES; t++) {
        a->n7[t] += b->n7[t];
        if (b->best_n[t] < a->best_n[t])
            a->best_n[t] = b->best_n[t], a->best_a[t] = b->best_a[t], a->best_b[t] = b->best_b[t];
    }
}

static void print_stats(const stats_t *s) {
    printf("centres examined (|D(n)| >= 3): %llu\n", (unsigned long long)s->nodes);
    if (solve_only) {
        printf("offsets total: %llu   pairs: %llu\n", (unsigned long long)s->sumk, (unsigned long long)s->pairs);
        printf("candidate pairs (offsets jointly divisible by every usable prime): %llu (%.3f%%)\n",
               (unsigned long long)s->fhits, s->pairs ? 100.0 * s->fhits / s->pairs : 0.0);
        printf("  of which n^2 + a + b is a square: %llu\n", (unsigned long long)s->qpass);
        printf("MAGIC SQUARES OF 9 DISTINCT SQUARES: %llu\n", (unsigned long long)s->sols);
        printf("9 squares with a repeated entry:     %llu\n", (unsigned long long)s->reps);
        printf("(near misses are not searched with -s)\n");
        return;
    }
    printf("offsets total: %llu   pairs checked: %llu   checksum: %016llx\n",
           (unsigned long long)s->sumk, (unsigned long long)s->pairs, (unsigned long long)s->chk);
    printf("pair filter hits checked exactly: %llu (%.3f%%)\n", (unsigned long long)s->fhits,
           s->pairs ? 100.0 * s->fhits / s->pairs : 0.0);
    if (verify)
        printf("pairs covering the usable primes, counted directly (= -s candidate pairs): %llu\n",
               (unsigned long long)s->mcov);
    printf("MAGIC SQUARES OF 9 DISTINCT SQUARES: %llu\n", (unsigned long long)s->sols);
    printf("9 squares with a repeated entry:     %llu\n", (unsigned long long)s->reps);
    printf("8 squares (square centre):           %llu\n", (unsigned long long)s->eight);
    u64 tot = 0;
    for (int t = 0; t < NTYPES; t++) tot += s->n7[t];
    printf("7-square near misses: %llu (of which %llu have a negative cell; %llu repeated-entry skipped)\n",
           (unsigned long long)tot, (unsigned long long)s->neg7, (unsigned long long)s->rep7);
    for (int t = 0; t < NTYPES; t++) {
        if (quick && (t == T_C || t == T_D)) {
            printf("  %-38s  not searched (-q)\n", type_name[t]);
            continue;
        }
        printf("  %-38s %12llu", type_name[t], (unsigned long long)s->n7[t]);
        if (s->best_n[t] != UINT64_MAX) printf("   smallest centre %llu^2", (unsigned long long)s->best_n[t]);
        putchar('\n');
    }
}

static void print_examples(const stats_t *s) {
    for (int t = 0; t < NTYPES; t++) {
        if (s->best_n[t] == UINT64_MAX) continue;
        printf("\nsmallest 7-square example, %s:\n", type_name[t]);
        print_square(stdout, s->best_n[t], s->best_a[t], s->best_b[t]);
    }
}

// ---------------------------------------------------------------- brute force

static int all_1mod4(u64 n) {
    if (n % 2 == 0) return 0;
    for (u64 p = 3; p * p <= n; p += 2)
        if (n % p == 0) {
            if (p % 4 == 3) return 0;
            while (n % p == 0) n /= p;
        }
    return n == 1 || n % 4 == 1;
}

// Independent count of the same configurations straight from the definition:
// build candidate (a, b) from every way two members of D(n) can play two of
// the roles a, b, a+b, a-b, then test each line by checking squares directly.
static void indep(u64 n, const dval *D, int k, u64 *seven, u64 *eight, u64 *nine) {
    // seven[t]: configurations of type t (indexed like T_A..T_D by the missing
    // offset a-b, a+b, b, a); in -q mode only types A and B are counted.
    static const int type_of_missing[4] = {T_D, T_C, T_B, T_A};
    size_t cap = (size_t)k * k * 6 + 1, m = 0;
    dval(*cand)[2] = malloc(cap * sizeof *cand);
    for (int x = 0; x < k; x++)
        for (int y = 0; y < k; y++) {
            if (x == y) continue;
            dval X = D[x], Y = D[y];
            dval c[6][2] = {{X, Y}, {X, Y - X}, {X, X - Y}, {Y - X, X}, {X + Y, X}, {(X + Y) / 2, (X - Y) / 2}};
            int ok[6] = {X > Y, Y > X, X > Y, Y > X, 1, X > Y};
            for (int r = 0; r < 6; r++)
                if (ok[r] && c[r][0] > c[r][1] && c[r][1] > 0 && c[r][0] != 2 * c[r][1])
                    cand[m][0] = c[r][0], cand[m][1] = c[r][1], m++;
        }
    // Dedupe.
    for (size_t i = 1; i < m; i++) {
        dval a0 = cand[i][0], a1 = cand[i][1];
        size_t j = i;
        while (j > 0 && (cand[j - 1][0] > a0 || (cand[j - 1][0] == a0 && cand[j - 1][1] > a1)))
            cand[j][0] = cand[j - 1][0], cand[j][1] = cand[j - 1][1], j--;
        cand[j][0] = a0, cand[j][1] = a1;
    }
    u128 sq = (u128)n * n;
    for (size_t i = 0; i < m; i++) {
        if (i && cand[i][0] == cand[i - 1][0] && cand[i][1] == cand[i - 1][1]) continue;
        u128 a = cand[i][0], b = cand[i][1];
        u128 off[4] = {a, b, a + b, a - b};
        int full = 0, part = 0, miss = 0;
        for (int o = 0; o < 4; o++) {
            int up = is_sq128(sq + off[o], NULL), dn = off[o] < sq && is_sq128(sq - off[o], NULL);
            if (up && dn) full++;
            else {
                miss = o;
                if (up || dn) part++;
            }
        }
        if (full == 4) (*nine)++;
        if (full == 3 && !(quick && miss < 2)) {
            seven[type_of_missing[miss]]++;
            if (part) (*eight)++;
        }
    }
    free(cand);
}

static int brute(u64 lim) {
    N = lim;
    ctx_t c;
    ctx_init(&c);
    u64 seven[NTYPES] = {0}, eight = 0, nine = 0;
    for (u64 n = 5; n <= lim; n++) {
        if (!all_1mod4(n)) continue;
        int k = 0;
        u128 sq = (u128)n * n;
        for (u64 r = 1; r < n; r++) {
            u64 s;
            if (is_sq128(sq - (u128)r * r, &s) && s > 0 && r > s) c.D[k++] = 2 * r * s;
        }
        if (k < 3) continue;
        qsort(c.D, k, sizeof *c.D, cmp_dval);
        indep(n, c.D, k, seven, &eight, &nine);
        analyze(&c, n, c.D, k);
    }
    print_stats(&c.st);
    int ok = eight == c.st.eight && nine == c.st.sols;
    printf("independent count: seven by type =");
    for (int t = 0; t < NTYPES; t++) {
        printf(" %llu", (unsigned long long)seven[t]);
        ok &= seven[t] == c.st.n7[t];
    }
    printf(", eight=%llu nine=%llu  -> %s\n", (unsigned long long)eight, (unsigned long long)nine,
           ok ? "AGREES" : "DISAGREES");
    return !ok;
}

// Exercise analyze() on random synthetic offset sets, comparing its counts
// against a direct enumeration of every (a, b).
static int selftest(void) {
    ctx_t c;
    ctx_init(&c);
    selftest_mode = 1;
    srand(12345);
    u64 want7[NTYPES] = {0}, want9 = 0;
    for (int trial = 0; trial < 4000; trial++) {
        // D holds even values (as the real D(n) does); index 2x <-> value x.
        int lim = 20 + rand() % 200, k = 0;
        uint8_t in[1024] = {0};
        for (int x = 2; x <= 2 * lim; x += 2)
            if (rand() % 4 == 0) in[x] = 1, c.D[k++] = (dval)x;
        if (k < 3) continue;
        for (int a = 2; a <= 2 * lim; a++)
            for (int b = 1; b < a; b++) {
                int cnt = in[a] + in[b] + in[a + b] + in[a - b];
                if (cnt == 4 && a != 2 * b) want9++;
                if (cnt != 3 || a == 2 * b) continue;
                int t = !in[a - b] ? T_A : !in[a + b] ? T_B : !in[b] ? T_C : T_D;
                if (!(quick && (t == T_C || t == T_D))) want7[t]++;
            }
        analyze(&c, 1, c.D, k);
    }
    int ok = want9 == c.st.sols && c.st.eight == 0;
    printf("seven by type: want/got");
    for (int t = 0; t < NTYPES; t++) {
        printf(" %llu/%llu", (unsigned long long)want7[t], (unsigned long long)c.st.n7[t]);
        ok &= want7[t] == c.st.n7[t];
    }
    printf(" | nine: want %llu got %llu | eight (must be 0): %llu\n", (unsigned long long)want9,
           (unsigned long long)c.st.sols, (unsigned long long)c.st.eight);
    printf("selftest %s\n", ok ? "PASSED" : "FAILED");
    return !ok;
}

// ---------------------------------------------------------------- checkpoints
//
// The work is a list of items: the tasks, then (without the GPU) the sieve
// blocks of the large primes; the GPU does its own work units in order.  The
// worker threads take items in order and add each one's counts to wsum once
// it is done.  A checkpoint holds the items done, wsum, and how many GPU units
// are done, with the GPU's counts, written to a new file renamed over the old
// one.  The GPU's counts include those of every leaf run it was handed, so
// while a checkpoint is wanted the items started do their leaves themselves;
// once the items that did hand runs over are done, the GPU finishes those
// runs (gpu_snapshot()) and the checkpoint is taken.  The workers never wait.
// The monitor thread takes one every ckpt_secs (-c), and at the time limit
// (-t) or on SIGINT or SIGTERM, after which the run ends at once.

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    char magic[8], mode[8];
    u64 N, stsize, ntasks, nlarge;     // nlarge: CPU sieve blocks, or GPU units (0: not known yet)
    u64 gpu, tdone, ldone;             // tasks done, and blocks or units done
    u64 leaves, runs, sessions;        // leaves the GPU took from the CPU, and their runs
    double secs;                       // search time of every session so far
    stats_t st;                        // the counts of the work done
} ckpt_t;

static const char *prog = "mss";       // program name, for file names and results.csv
static char mode[8], ckname[128];
static u64 *blk, qhi;                  // sieve blocks [blk[b], blk[b + 1]) of the CPU's large primes
static size_t nblocks;
static size_t nwork, nnext, ngpu, ndone_t, ndone_b;  // ngpu: items in hand that may give the GPU leaf runs
static uint8_t *wdone;                 // per item: done
static int pausing, mquit;             // pausing: a checkpoint is wanted
static pthread_mutex_t wl = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t mc = PTHREAD_COND_INITIALIZER;
static stats_t wsum, fresh;            // the counts of the items done; empty counts
static ckpt_t ck0, ck;                 // resumed from (or empty), and the last one written
static double ckpt_secs = 600, deadline, t_search, t_small;
static volatile sig_atomic_t sig_stop;

static void on_signal(int sig) {
    (void)sig;
    sig_stop = 1;
}

#ifdef GPU
static void add_gpu(stats_t *s, const gpu_out *g) {
    s->nodes += g->nodes, s->sumk += g->sumk, s->pairs += g->pairs, s->fhits += g->fhits, s->qpass += g->qpass;
    s->sols += g->sols, s->reps += g->reps;
}
#endif

// results.csv: one row per search (program, mode, N), replaced in place.
static void csv_row(const char *status, const ckpt_t *h) {
    const stats_t *s = &h->st;
    char key[96], row[1024], ts[32], line[2048], tmp[64];
    u64 seven = 0;
    for (int t = 0; t < NTYPES; t++) seven += s->n7[t];
    stamp(ts, sizeof ts);
    snprintf(key, sizeof key, "%s,%s,%llu,", prog, mode, (unsigned long long)N);
    snprintf(row, sizeof row, "%s%s,%llu,%llu,%llu,%llu,%.3f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%s\n",
             key, status, (unsigned long long)h->tdone, (unsigned long long)h->ntasks, (unsigned long long)h->ldone,
             (unsigned long long)h->nlarge, h->secs / 3600, (unsigned long long)h->sessions,
             (unsigned long long)s->nodes, (unsigned long long)s->sumk, (unsigned long long)s->pairs,
             (unsigned long long)s->fhits, (unsigned long long)s->qpass, (unsigned long long)s->sols,
             (unsigned long long)s->reps, (unsigned long long)s->eight, (unsigned long long)seven, ts);
    snprintf(tmp, sizeof tmp, "results.csv.%d", (int)getpid());
    FILE *in = fopen("results.csv", "r"), *out = fopen(tmp, "w");
    if (!out) {
        perror(tmp);
        if (in) fclose(in);
        return;
    }
    fputs("program,mode,N,status,tasks_done,tasks,large_done,large_total,search_hours,sessions,centres,offsets,"
          "pairs,candidate_pairs,square_sums,magic,repeated,eight_squares,seven_squares,updated\n",
          out);
    int put = 0;
    if (in && fgets(line, sizeof line, in))  // (its header)
        while (fgets(line, sizeof line, in)) {
            if (strncmp(line, key, strlen(key))) fputs(line, out);
            else if (!put) fputs(row, out), put = 1;
        }
    if (!put) fputs(row, out);
    if (in) fclose(in);
    if (fclose(out) || rename(tmp, "results.csv")) perror("results.csv");
}

static void ckpt_save(const ckpt_t *h) {
    char tmp[160];
    snprintf(tmp, sizeof tmp, "%s.tmp", ckname);
    size_t nb = nwork / 8 + 1;
    uint8_t *bits = calloc(nb, 1);
    for (size_t i = 0; i < nwork; i++) bits[i / 8] |= (uint8_t)(wdone[i] << (i % 8));
    FILE *f = fopen(tmp, "wb");
    int ok = f && fwrite(h, sizeof *h, 1, f) == 1 && fwrite(bits, 1, nb, f) == nb && fflush(f) == 0 &&
             fsync(fileno(f)) == 0;
    if (f) ok &= fclose(f) == 0;
    if (!ok || rename(tmp, ckname)) perror(ckname);
    free(bits);
}

// The empty checkpoint of this search, or the one in its file.
static void ckpt_load(void) {
    ckpt_t *h = &ck0;
    memset(h, 0, sizeof *h);
    memcpy(h->magic, "mss ck1", 8);
    memcpy(h->mode, mode, sizeof h->mode);
    h->N = N, h->stsize = sizeof(stats_t), h->ntasks = ntasks, h->nlarge = gpu_on ? 0 : nblocks, h->gpu = gpu_on;
    for (int t = 0; t < NTYPES; t++) h->st.best_n[t] = wsum.best_n[t] = fresh.best_n[t] = UINT64_MAX;
    snprintf(ckname, sizeof ckname, "%s-%s-%llu.ckpt", prog, mode, (unsigned long long)N);
    FILE *f = fopen(ckname, "rb");
    if (!f) return;
    ckpt_t r;
    size_t nb = nwork / 8 + 1;
    uint8_t *bits = malloc(nb);
    if (fread(&r, sizeof r, 1, f) != 1 || fread(bits, 1, nb, f) != nb || memcmp(r.magic, h->magic, 8) ||
        memcmp(r.mode, h->mode, sizeof r.mode) || r.N != N || r.stsize != h->stsize || r.ntasks != ntasks ||
        r.gpu != h->gpu || (!gpu_on && r.nlarge != nblocks)) {
        fprintf(stderr, "%s is not a checkpoint of this search (remove it to start over)\n", ckname);
        exit(1);
    }
    fclose(f);
    for (size_t i = 0; i < nwork; i++) {
        wdone[i] = bits[i / 8] >> (i % 8) & 1;
        if (wdone[i]) *(i < ntasks ? &ndone_t : &ndone_b) += 1;
    }
    free(bits);
    *h = ck = r;
    printf("resuming from %s: %zu/%zu tasks, %llu/%llu %s done, %.2fh searched in %llu session%s\n", ckname,
           ndone_t, ntasks, (unsigned long long)r.ldone, (unsigned long long)r.nlarge,
           gpu_on ? "GPU units" : "sieve blocks", r.secs / 3600, (unsigned long long)r.sessions,
           r.sessions == 1 ? "" : "s");
}

// Write the counts of the work done so far; stop: then end the run.
static void checkpoint(int stop) {
    double ta = now();
    pthread_mutex_lock(&wl);
    pausing = 1;
    while (ngpu) pthread_cond_wait(&mc, &wl);
    pthread_mutex_unlock(&wl);
    ckpt_t h = ck0;
#ifdef GPU
    gpu_out g = {0};
    if (gpu_on) {
        gpu_snapshot(&g);
        add_gpu(&h.st, &g);
        h.leaves += g.leaves, h.runs += g.runs;
        if (g.units) h.nlarge = g.units;
    }
#endif
    pthread_mutex_lock(&wl);
    merge(&h.st, &wsum);
    h.sessions++, h.secs += now() - t_search, h.tdone = ndone_t, h.ldone = ndone_b;
#ifdef GPU
    if (gpu_on) h.ldone = g.udone;
#endif
    ckpt_save(&h);
    csv_row(stop ? "stopped" : "running", &h);
    ck = h;
    pausing = stop;
    pthread_mutex_unlock(&wl);
    printf("  checkpoint: %llu/%llu tasks, %llu/%llu %s, %.0fs searched (took %.1fs)\n",
           (unsigned long long)h.tdone, (unsigned long long)h.ntasks, (unsigned long long)h.ldone,
           (unsigned long long)h.nlarge, gpu_on ? "GPU units" : "sieve blocks", h.secs, now() - ta);
    if (stop) {  // the items in hand (and the GPU's launch) are dropped: a resumed run does them
        printf("\nstopped after %.2fs, %.2fh searched in all; run the same command to resume.  So far:\n\n",
               now() - t_search, h.secs / 3600);
        print_stats(&h.st);
        fflush(stdout);
        _exit(3);
    }
    fflush(stdout);
}

static void *monitor(void *arg) {
    (void)arg;
    double next = ckpt_secs > 0 ? now() + ckpt_secs : INFINITY;
    for (;;) {
        struct timespec ts = {0, 200000000};
        nanosleep(&ts, NULL);
        pthread_mutex_lock(&wl);
        int quit = mquit;
        pthread_mutex_unlock(&wl);
        if (quit) return NULL;
        int stop = sig_stop || (deadline > 0 && now() >= deadline);
        if (!stop && now() < next) continue;
        if (stop) printf("%s: writing a checkpoint, then stopping\n", sig_stop ? "signal" : "time limit"), fflush(stdout);
        checkpoint(stop);
        next = now() + ckpt_secs;
    }
}

// A worker: items in order until none is left.
static void work(void) {
    ctx_t c;
    ctx_init(&c);
    uint8_t *comp = malloc(SEG / 4 + 1);
    static double last;
    pthread_mutex_lock(&wl);
    last = t_search;
    for (;;) {
        while (nnext < nwork && wdone[nnext]) nnext++;
        if (nnext == nwork) break;
        size_t i = nnext++;
        c.gpu_ok = gpu_on && !pausing;
        ngpu += c.gpu_ok;
        pthread_mutex_unlock(&wl);
        c.st = fresh;
        if (i < ntasks) run_task(&c, &tasks[i]);
        else run_block(&c, blk[i - ntasks], blk[i - ntasks + 1] < qhi ? blk[i - ntasks + 1] : qhi, comp);
        pthread_mutex_lock(&wl);
        merge(&wsum, &c.st);
        wdone[i] = 1;
        if (i >= ntasks) ndone_b++;
        else if (++ndone_t == ntasks) t_small = now() - t_search;
        if (c.gpu_ok && !--ngpu && pausing) pthread_cond_signal(&mc);
        if (now() - last > 60) {
            last = now();
            if (i < ntasks) printf("  progress: small primes %zu/%zu tasks, %.0fs\n", ndone_t, ntasks, last - t_search);
            else printf("  progress: large primes %zu/%zu blocks, %.0fs\n", ndone_b, nblocks, last - t_search);
            fflush(stdout);
        }
    }
    pthread_mutex_unlock(&wl);
    free(comp);
}

static double parse_time(const char *s) {  // seconds, or with a suffix s, m, h, d
    char *e;
    double v = strtod(s, &e);
    return v * (*e == 'm' ? 60 : *e == 'h' ? 3600 : *e == 'd' ? 86400 : 1);
}

// ---------------------------------------------------------------- main

int main(int argc, char **argv) {
    double t0 = now(), tlimit = 0;
    prog = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    for (; argc >= 2 && argv[1][0] == '-' && argv[1][1] && !argv[1][2]; argv++, argc--) {
        if (argv[1][1] == 'q') quick = 1;
        else if (argv[1][1] == 's') solve_only = 1;
        else if (argv[1][1] == 'v') verify = 1;
#ifdef GPU
        else if (argv[1][1] == 'g') use_gpu = solve_only = 1;
#endif
        else if ((argv[1][1] == 't' || argv[1][1] == 'c') && argc >= 3) {
            *(argv[1][1] == 't' ? &tlimit : &ckpt_secs) = parse_time(argv[2]);
            argv++, argc--;
        } else break;
    }
    prune_init();
    if (argc >= 3 && !strcmp(argv[1], "brute")) return brute(strtoull(argv[2], 0, 10));
    if (argc >= 2 && !strcmp(argv[1], "selftest")) return selftest();
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s [-q|-s|-v|-g] [-t TIME] [-c SECS] N [threads] | %s [-q|-s] selftest | %s [-q|-v] brute N\n",
                prog, prog, prog);
        return 1;
    }
    N = (u64)strtod(argv[1], 0);
    if (N >= NMAX) {
        fprintf(stderr, "N must be < %llu (build with -DWIDE for more)\n", NMAX);
        return 1;
    }
    if (argc >= 3) omp_set_num_threads(atoi(argv[2]));
    snprintf(mode, sizeof mode, "%s%s", use_gpu ? "g" : solve_only ? "s" : quick ? "q" : "full", verify ? "v" : "");
    struct sigaction sa = {0};
    sa.sa_handler = on_signal, sa.sa_flags = SA_RESTART | SA_RESETHAND;  // a second signal ends the run at once
    sigaction(SIGINT, &sa, NULL), sigaction(SIGTERM, &sa, NULL);

    B = isqrt64(N);
    sieve_primes(B, isqrt64(N / 5) + 1);
    prime_tables();
    printf("N = %llu: %zu primes = 1 mod 4 up to B = %llu (%.1fs)\n", (unsigned long long)N, NP,
           (unsigned long long)B, now() - t0);

    {
        ctx_t c;
        ctx_init(&c);
        gen_m(&c, 0, 1, 0, 0, N / (B + 1));
        qsort(M, NM, sizeof *M, cmp_mrec);
    }
    // Work per q grows like N/q (number of cofactors), so blocks start small
    // and grow geometrically up to SEG to keep the threads balanced.
    size_t bcap = 0;
    qhi = N / 5 + 1;
    pat_init();
    for (u64 lo = B / 4 * 4 + 1 + (B % 4 ? 4 : 0); lo < qhi;) {  // smallest 1 (mod 4) above B
        u64 len = lo / 64 < SEG ? (lo / 64 + 4) & ~3ull : SEG;
        if (nblocks == bcap) blk = realloc(blk, (bcap = bcap ? 2 * bcap : 1024) * sizeof *blk);
        blk[nblocks++] = lo;
        lo += len;
    }
    if (nblocks == bcap) blk = realloc(blk, (bcap + 1) * sizeof *blk);
    blk[nblocks] = qhi;
    printf("%zu cofactors m <= N/(B+1) for primes in (B, N/5]: %zu sieve blocks\n", NM, nblocks);

    THR = N / 800 > 1 ? N / 800 : 1;
    task_t root = {.n = 1};
    pre(&root, 0);
    printf("%zu tasks, %d threads\n", ntasks, omp_get_max_threads());
#ifdef GPU
    if (use_gpu && !gpu_usable()) printf("(GPU phase needs B >= 38 and the usual usable primes: on the CPU)\n");
    else if (use_gpu) gpu_on = 1;
#endif
    nwork = ntasks + (gpu_on ? 0 : nblocks);
    wdone = calloc(nwork + 1, 1);
    ckpt_load();
    fflush(stdout);
#ifdef GPU
    pthread_t gth;
    if (gpu_on) gustart = ck0.ldone, gucheck = ck0.nlarge, pthread_create(&gth, NULL, gpu_thread, NULL);
#endif

    t_search = now();
    if (tlimit > 0) deadline = t0 + tlimit;
    pthread_t mth;
    pthread_create(&mth, NULL, monitor, NULL);
#pragma omp parallel
    work();
#ifdef GPU
    if (gpu_on) {
        gpu_leaves_done();
        pthread_join(gth, NULL);
        if (gfail) return 1;
    }
#endif
    pthread_mutex_lock(&wl);
    mquit = 1;
    pthread_mutex_unlock(&wl);
    pthread_join(mth, NULL);

    ckpt_t h = ck0;  // the whole search
    h.sessions++, h.secs += now() - t_search, h.tdone = ndone_t, h.ldone = ndone_b;
    merge(&h.st, &wsum);
#ifdef GPU
    if (gpu_on) {
        add_gpu(&h.st, &gres);
        h.leaves += gres.leaves, h.runs += gres.runs, h.ldone = gres.udone, h.nlarge = gres.units;
    }
#endif
#ifdef GPU
    if (gpu_on)
        printf("search time %.2fs (CPU threads done at %.2fs; GPU: large primes %.2fs, then leaves until %.2fs)\n",
               now() - t_search, t_small, gres.secs2, gres.secs);
    else
#endif
    printf("search time %.2fs (centres with all primes <= B: %.2fs, with a prime > B: %.2fs)\n", now() - t_search,
           t_small, now() - t_search - t_small);
    if (ck0.sessions) printf("(resumed: %.2fh searched in %llu sessions in all)\n", h.secs / 3600, (unsigned long long)h.sessions);
    putchar('\n');
    print_stats(&h.st);
    print_examples(&h.st);
    csv_row("complete", &h);
    remove(ckname);
    return 0;
}
