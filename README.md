# Searching for a 3×3 magic square of squares

Can nine **distinct** perfect squares fill a 3×3 grid so that every row,
column and both diagonals have the same sum? This is a famous open problem
(LaBar 1984; Martin Gardner's $100 prize, 1996; Christian Boyer's €1000
prize). Nobody has found one, and nobody has proved it impossible.

This repository is an exhaustive search by the centre of the square. It
runs in C with OpenMP, with an optional CUDA port for NVIDIA GPUs.

**Result so far: no magic square of squares has a centre entry `n²` with
`n ≤ 10¹³`**, i.e. a centre up to `10²⁶`. A non-primitive solution would
give a primitive one with a smaller centre, so this rules out every
solution, not just primitive ones. A second search, to `n ≤ 10¹¹`, found
no 7- or 8-square near miss of the kinds described below.

* [Results](#results)
* [Building and running](#building-and-running)
* [Method](#method)
* [Validation](#validation)
* [Performance](#performance)
* [The GPU port](#the-gpu-port)
* [Long runs: checkpoints, time limit, results file](#long-runs-checkpoints-time-limit-results-file)
* [What was already known](#what-was-already-known)
* [Files](#files)

## Results

| search | centres `n` examined | time | log |
|--------|---------------------:|-----:|-----|
| solutions only, all `n ≤ 10¹³`, CPU | 424,854,595,931 | 49.6 min | [`runs/s_le_1e13.log`](runs/s_le_1e13.log) |
| the same on the GPU (stopped twice and resumed) | 424,854,595,931 | 6.2 min | [`runs/g_le_1e13.log`](runs/g_le_1e13.log) |
| solutions only, all `n ≤ 10¹²`, CPU | 43,423,993,614 | 4.2 min | [`runs/s_le_1e12.log`](runs/s_le_1e12.log) |
| the same on the GPU | 43,423,993,614 | 26 s | [`runs/g_le_1e12.log`](runs/g_le_1e12.log) |
| solutions and near misses, all `n ≤ 10¹¹` | 4,440,149,081 | 4.4 min | [`runs/n_le_1e11.log`](runs/n_le_1e11.log) |
| solutions and near misses, all `n < 2³²` (first version) | 196,614,507 | 2.3 min | [`runs/n_lt_2e32.log`](runs/n_lt_2e32.log) |

"Centres examined" counts the `n` with at least three offsets (see
[Method](#method)); every other `n` cannot be the centre of a solution. The
CPU and GPU runs agree on every count. [`results.csv`](results.csv) has the
same runs with all their counts.

* **Solutions.** The run to `10¹³` tested 2.13·10¹¹ candidate pairs of
  offsets, 0.149% of all 1.43·10¹⁴ pairs. For 1,453,353 of them the cell
  `n² + a + b` was a square too, but none had the remaining cells square as
  well. No square with a repeated entry turned up either.
* **Near misses.** The search to `10¹¹` found no centre with three of the
  four offsets `a, b, a+b, a−b` in `D(n)`, in any of the four arrangements.
  So no 7-square magic square of this form has a centre entry up to `10²²`.
  That includes the "magic hourglass", whose two non-square cells are in the
  middle row or column. No 8-square one exists in that range either.

Hardware for all timings: AMD Ryzen 7 2700 (8 cores, 16 threads) and an
NVIDIA RTX 2080 Ti.

## Building and running

Needs a 64-bit Linux system and GCC with OpenMP (the code uses `__int128` and
POSIX calls; only x86-64 Linux has been tested). AVX2 is used when present
(`-march=native`), with plain-C fallbacks. The GPU build also needs the CUDA
toolkit.

```bash
make                         # mss (64-bit offsets, N < 2^32) and mss-wide (128-bit, N < 2^50)
make gpu                     # mss-gpu: the 128-bit build plus the CUDA port (-g)
make test                    # correctness checks (about 2 min; with mss-gpu built, also -g)
make bench                   # timings
```

`make gpu` looks for CUDA in `/usr/local/cuda` and builds for `sm_75`; set
`CUDA=` and `CUDA_ARCH=` to change either.

```bash
./mss 4294967295 16          # all centres n < 2^32 on 16 threads, with near misses (~8 s)
./mss-wide -s 1e12 16        # solutions only, all centres n <= 10^12 (~4 min)
./mss-gpu -g 1e12 16         # the same with the GPU (~30 s)
./mss-gpu -g -t 6h 2e14 16   # at most 6 hours, then a checkpoint; the same command resumes
./mss -v 1e8 16              # also count the pairs -s would test, by direct division
./mss brute 100000           # slow, independent check
./mss selftest               # the near-miss logic on synthetic offset sets
```

`mss [options] N [threads]` searches every centre `n ≤ N`.

| option | |
|--------|---|
| (none) | solutions and near misses |
| `-q` | skips the two tests only diagonal near misses need: 1.9× faster, still finds every solution and hourglass |
| `-s` | solutions only, with divisibility pruning: about 5× faster than `-q` |
| `-g` | as `-s`, with most of the work on the GPU (`mss-gpu` only) |
| `-v` | also counts, by plain division, the pairs `-s` must test (a check of `-s`) |
| `-t TIME` | stop after `TIME` (seconds, or e.g. `90m`, `6h`), with a checkpoint |
| `-c SECS` | checkpoint every `SECS` seconds (default 600; `0`: only when stopping) |

Anything found is printed, and appended to `finds.txt` and `finds.csv` in the
working directory. Each search also writes its checkpoint and its row of
`results.csv` there (see [Long runs](#long-runs-checkpoints-time-limit-results-file)).

## Method

Every 3×3 magic square with centre `c` can be written (Lucas) as

```
c-b     c+a+b   c-a
c-a+b   c       c+a-b
c+a     c-a-b   c+b
```

In a magic square of squares the centre is a square `n²`, and each of the
four lines through it is a 3-term arithmetic progression of squares
`n²-d, n², n²+d`. Writing the outer roots as `r-s` and `r+s` turns this into
`r² + s² = n²` with `d = 2rs`. So the usable offsets for centre `n²` are

```
D(n) = { 2rs : r² + s² = n², r > s > 0 }
```

(four times the areas of the right triangles with hypotenuse `n`). A magic
square of distinct squares with centre `n²` exists **iff** there are `a > b`
in `D(n)`, `a ≠ 2b`, with both `a+b` and `a-b` in `D(n)`.

If 2 or a prime `p ≡ 3 (mod 4)` divides `n`, then `p` divides both `r` and
`s` in every representation, so all nine entries share the factor `p²`.
Dividing that out gives a smaller solution. So it is enough to search centres
`n` whose prime factors are all `≡ 1 (mod 4)`.

`mss.c` enumerates every such `n ≤ N` and builds `D(n)` exactly from
Gaussian-integer factorisations (`p = ππ̄`). That yields every representation
of `n²` as a sum of two squares, so the search is exhaustive, with no
sampling or heuristics.

* Centres whose prime factors are all `≤ √N` are covered by a depth-first
  search over their factorisations, split into tasks for the threads.
* Centres with a prime factor `q > √N` are covered by a segmented sieve.
  `q = X² + Y²` is read straight off the lattice points in each block, and
  each `q` is combined with every eligible cofactor `m ≤ N/q`. This needs no
  prime table up to `N`, so memory stays around 150 MB even at `N = 10¹³`.
* Nearly every centre is a "leaf" `n·p`, where `p` is its largest prime. Its
  offsets are those of `H(n)·{π̄², p, π²}` and `n·π̄²`, where `H(n)` holds one
  Gaussian integer per offset of `n`. So they are computed on the fly from
  the parent's set and never stored.

### The three modes

**Default: solutions and near misses.** Every pair of offsets is probed with
a branch-free byte filter for the patterns `x+y = z`, `x+y = 2m` and
`y-x = 2h`, and the rare hits are checked exactly. That finds every solution
and every **near miss**: a centre where three of the four required offsets
`a, b, a+b, a-b` lie in `D(n)`. That gives a magic square with seven square
entries, the two non-squares sitting on one line through the centre.
* If that line is the middle row or column, this is the **"magic hourglass"**
  that Buell, Pech and Morgenstern hunted for.
* Any such example would be a new 7-square magic square, which is Boyer's
  second open problem. The only known 7-square magic square, Bremner's, has
  its non-squares elsewhere.
* If either non-square cell were also a square, the square would have eight
  square entries, and the program reports that loudly.

**`-q`.** This skips the two probes that only the diagonal near misses need.

**`-s`: solutions only.** All nine cells of a solution are squares modulo
every prime. For `p = 7, 11, 13, 17, 19, 29, 31, 37` that is impossible unless
`p | ab`. The program proves this for each prime at start-up by trying all
residues. The proof needs `p ∤ n`, except for 13, where it always holds. So
in a solution, the two offsets `b, a` are between them divisible by every
one of those primes that applies. Each offset gets an 8-bit mask of the
primes dividing it, and only pairs whose masks cover all the usable primes
are tested: about 0.13% of all pairs. The masks need no arithmetic:
* For `h, f ≢ 0 (mod p)`, `p | Re(hf)` iff `[Re h : Im h] = [Im f : Re f]` on
  the projective line over `F_p`, and `p | Im(hf)` iff
  `[Re h : Im h] = [−Re f : Im f]`.
* So each Gaussian integer is kept as one byte per prime, and each mask bit is
  two byte comparisons, sixteen at a time with SIMD.
* Most centres are rejected because no two masks have enough bits between
  them. The offsets themselves are only computed for the candidate pairs.

**`-g`: `-s` on the GPU as well.** An NVIDIA GPU searches the centres with a
prime factor above `√N` (83% of them at `10¹¹`). It then takes over most of
the leaves of the CPU threads' search, which they hand over as they go. It
tests exactly the same pairs as `-s`; see [The GPU port](#the-gpu-port).

## Validation

`make test` (script [`tests.sh`](tests.sh), about 2 minutes) runs:

* **Brute force.** `./mss brute N` recomputes `D(n)` by direct trial over `r`
  for every eligible `n ≤ N`, re-derives every candidate `(a, b)`, and tests
  each line with direct square checks. For six values of `N` up to `10⁵`, its
  centre count, offset count, checksum, and near-miss and solution counts
  equal those of the fast search. The values of `N` give different
  `√N mod 4`, which moves where the sieve starts.
* **`-s` against a direct count.** `-v` counts, by plain division on the
  exact offsets, the pairs `-s` must test. It agrees with the candidate count
  of `-s` on brute-force `D(n)` up to `10⁵`, and on the fast search at `10⁸`.
  Separately, `-s` at `2·10⁹` gives the recorded 13,640,004 candidate pairs.
* **Recorded results.** At `10⁹` the default mode reproduces the recorded
  checksum. The 64-bit and 128-bit builds agree.
* **GPU.** When `mss-gpu` is built, `-g` must give every count of `-s`
  (centres, offsets, pairs, candidate pairs, cells `n² + a + b` that are
  squares, solutions) at `10⁶`, `10⁸`, `2·10⁹` and `2·10¹⁰`. At `10¹¹` it must
  give the recorded counts, and the GPU must have taken leaves of the
  small-prime search.
* **Checkpoints.** `-s` at `10¹¹` and `-g` at `10¹²` are stopped by the time
  limit twice and resumed. The final counts must equal an uninterrupted
  run's, and `results.csv` must show a complete search over three sessions.
* **Self-test.** `./mss selftest` runs the near-miss logic on 4000 random
  synthetic offset sets. It agrees with a direct enumeration of every
  `(a, b)` on all 1,085,847 seven-of-nine and all 68,879 nine-of-nine
  configurations.

Checked once more by hand:

* The current code reproduces the first, simple version bit for bit over all
  `n < 2³²`. That version used a full prime table and a pairwise binary
  search. Both give 196,614,507 centres, 27,316,185,118 offset pairs and
  checksum `aa6e06eb2e9aa366`, in both builds.
* At `n ≤ 2·10¹⁰` the result is checksum `070908fd54e71d50`, and `-s` finds
  195,290,344 candidate pairs.
* `-g` gives every count of `-s` at `10¹¹`, `10¹²` and `10¹³`. How many
  leaves go to the GPU changes from run to run; the counts do not.
  `compute-sanitizer` finds no memory errors, shared-memory races or barrier
  misuse in any of the kernels.
* `-g` runs stopped by the time limit or SIGTERM in either GPU phase, or
  killed with `kill -9` after a checkpoint, resume to the uninterrupted
  counts.
* A separate, deliberately naive sieve gives the same number of eligible
  centres and offsets at `10⁸` and `10⁹`. It factors every `n` with a
  smallest-prime-factor table.
* Anything reported is re-verified: all 8 line sums are recomputed in 128-bit
  arithmetic before printing.

## Performance

Search time with 16 threads, excluding the setup of a second or so:

| search                 | first version | now     |
|------------------------|--------------:|--------:|
| all `n < 2³²`, default | 24.0 s   | 7.9 s   |
| same, 128-bit build    | 34.7 s   | 8.3 s   |
| same, `-s`             |          | 0.84 s  |
| `n ≤ 2·10⁹`, default   | 10.3 s   | 3.4 s   |
| `n ≤ 2·10⁹`, `-q`      |          | 1.8 s   |
| `n ≤ 2·10⁹`, `-s`      |          | 0.40 s  |
| `n ≤ 2·10¹⁰`, default  |          | 44.7 s  |
| `n ≤ 2·10¹⁰`, `-s`     |          | 4.1 s   |
| `n ≤ 10¹¹`, `-s`       |          | 22 s    |
| `n ≤ 10¹²`, `-s`       |          | 252 s   |
| `n ≤ 2·10¹⁰`, `-g`     |          | 1.0 s   |
| `n ≤ 10¹¹`, `-g`       |          | 2.6 s   |
| `n ≤ 10¹²`, `-g`       |          | 26 s    |
| `n ≤ 10¹³`, `-g`       |          | 6.2 min |

On the CPU the `-s` time grows like `N^1.06`. With the GPU it has grown
10–15× per factor of 10 in `N` so far. `make bench` reruns the main CPU rows.

<details>
<summary>What made the difference on the CPU</summary>

Measured one change at a time:
* **Default mode.** Each offset gets one filter byte (a multiplicative hash,
  sized to the centre and to L2). Offsets of leaf centres are generated
  straight into the analysis, never stored.
* **`-s` mode.**
  * The projective-residue masks are byte compares, not multiplies; Zen 1 has
    one 16-bit multiply pipe.
  * The first pass over each centre only computes the two largest mask
    popcounts, with `psadbw`. That rules out 77% of centres.
  * Covering pairs are found by a 16-wide scan from the offsets that have at
    least half the primes.
  * The most common case, a prime cofactor `m`, needs one 16-byte load and
    six ORs.
  * The exact test of a candidate starts with one double-precision square
    root of `n² + a + b`. Residue tests do not help there: every offset is
    `4mk(m²−k²)`, so `n² ± v` is automatically a square mod 64, 9 and mostly
    5. Mod a usable prime it is a square too, because one offset is 0.
* **Sieve.** It keeps only numbers `≡ 1 (mod 4)`, pre-sieves 3 to 13 with a
  repeating pattern, and uses 512 KB blocks. Each block's lattice rows have
  monotone bounds, so for large `N`, where a block is thinner than one row,
  an empty row costs a few instructions.
* **Tasks.** The tree search is split into tasks by an estimate of their work,
  `N/(n·p) · d((n·p)²)²` for the subtree under a child `n·p`: pair searches
  grow with the square of the offset count. Before the square weight, a task
  under a node with several prime factors could run for minutes.

Profile-guided optimisation and other compiler flags made no measurable
difference. Simultaneous multithreading gives 1.34× (16 threads against 8).
</details>

## The GPU port

`make gpu` builds `mss-gpu`: the 128-bit build plus
[`mss_gpu.cu`](mss_gpu.cu). The GPU first searches the large primes, while
the CPU threads run the small-prime search on their own. Then it takes over
the leaves of that search: centres `n·p` that the tree search does not extend
further, which take 92% of its time on the CPU. The CPU threads keep the rest
of the tree.

| `N`       | `-s` (CPU, 16 threads) | `-g`, large primes only | `-g` | leaves done on the GPU |
|-----------|------:|------:|------:|------:|
| `2·10¹⁰`  | 4.1 s | 1.8 s | 1.0 s | 134 million |
| `10¹¹`    | 22 s  | 9.9 s | 2.6 s | 740 million (97%) |
| `10¹²`    | 252 s | 116 s | 26 s  | 7.0 billion |

At `10¹²` the GPU spends 15 s on the large primes and 10 s on the leaves,
and the CPU threads finish when it does.

How it works:
* **Work units.** A unit is a sieve segment of about 2¹⁷ numbers
  `q ≡ 1 (mod 4)`, the lattice points `X² + Y² = q` in it, and a range of
  cofactors `m`. The segment's composites are a bitmap in shared memory:
  3 to 13 from a pattern, larger primes marked by a warp or a thread each.
  The lattice is walked by rows near the diagonal and by columns near the
  axis, so that no row or column holds more than a few points. Each thread
  walks its run twice: once to count its primes, and once to write them
  where a scan over the block puts them. Segments of small `q`, which pair
  with many cofactors, are split by cofactor range.
* **Lockstep.** A thread takes one `q` and up to 32 cofactors. The threads of
  a warp take different `q` with the same cofactors, so they step through
  the cofactor tables together and those loads are broadcasts.
* **Same masks as the CPU.** A Gaussian integer is one byte per prime, four
  bytes to a 32-bit word. Every byte is below `0x80`, so `(r ^ a) + 0x7f`
  compares four bytes at once with no carries between them.
* **Leaves from the CPU.** The leaves of a tree node `n` are a run of
  consecutive primes `p`. They are the same computation as the large-prime
  phase, with `n` as the cofactor and `p` as the prime. So a CPU thread copies
  the node's offsets, residue bytes and masks into a pinned buffer once, with
  the run's range of primes. A warp takes 32 primes of one node, which again
  makes the node's loads broadcasts. The CPU counts these centres; the GPU
  finds and tests their candidate pairs.
* **Neither side waits.** The four buffers fill while the GPU is on the large
  primes. A CPU thread that finds them full does that node's leaves itself.
  Afterwards the GPU takes each buffer as it fills, or the open one if it
  would otherwise idle. So both sides stay busy and finish together.
* **Heavy leaves.** A leaf with more than 8 offsets that have half the primes
  needs a full pair search. It goes to a third kernel that gives it a whole
  warp: masks in shared memory, flags as bitmasks, four masks per word in the
  partner scan, and one atomic per warp for its candidates.
* **Candidates.** Pairs whose masks cover the primes are queued per block and
  tested by another kernel with the double-precision square root of
  `n² + a + b`. The few that pass go back to the CPU for the exact check and
  the report. A node's own pair `(a, b)` with `n² + a + b` square passes again
  in every leaf `n·p` of its run, as `(p²a, p²b)`, so one launch can return a
  great many. Their buffer holds 4 million. A launch that overflows it is
  undone and redone in halves.
* **Short kernels.** The GPU may also drive the display, so a kernel must not
  run for long. Blocks take units for 40 ms, a buffer of leaves is at most
  about 20 ms of work, and the host relaunches. The desktop can stutter
  during a run.

<details>
<summary>What made the difference on the GPU</summary>

Measured at `10¹¹`, from clock counters inside the kernels, kernel timings,
and adding or removing one piece at a time (the profiling counters need root):
* The first version took 3.5 s, and its blocks sat idle 64% of the time,
  waiting for each launch's slowest unit. Time-limited launches fixed that.
* The lattice rows near the axis hold hundreds of points each, so threads
  were badly unbalanced. Splitting into rows and columns cut the lattice from
  30% of block time to 6%.
* Candidate pairs were first tested where they were found, in divergent code,
  and pushed to the queue from many inlined call sites. The kernel outgrew
  the instruction cache. A second kernel for the test and one push site per
  path took it from 2.8 s to 1.5 s.
* The leaves took 1.9 s of GPU time at first, and two thirds of it went to
  the 3% of leaves that need a full pair search. A lane did each one alone
  while the rest of its warp waited. Tree nodes have up to 341 offsets, the
  large-prime cofactors far fewer. The heavy-leaf kernel took the leaves to
  1.0 s, and the large primes at `10¹²` from 25 s to 15 s.
* In that kernel, four masks per word and one atomic per warp took it from
  0.30 s to 0.23 s. It was limited by latency more than by its scan: only two
  of three blocks per SM fit in shared memory until the flags became
  bitmasks, which took it to 0.17 s.
* The lattice's primes were appended through one shared counter, whose
  atomics serialised. Walking the lattice twice instead, with no atomics,
  took the large primes from 1.30 s to 1.08 s at `10¹¹` and from 15.9 s to
  14.9 s at `10¹²`.

At `10¹²` the sieve and lattice take 3.7 s when run without the leaves.
Most of the large-prime kernel's 15 s is the leaves `m·q` themselves.
</details>

## Long runs: checkpoints, time limit, results file

A search saves its state to `<program>-<mode>-<N>.ckpt` in the working
directory every 10 minutes (`-c SECS` changes that). It does the same at the
time limit `-t TIME`, or on Ctrl-C or SIGTERM, and then stops with exit
status 3. Running the same command again resumes from the file. A search
that completes deletes it. A second Ctrl-C ends the program at once.

* **What is saved.** The work is a list of items: the tasks of the
  small-prime search, then (on the CPU) the large-prime sieve blocks. The GPU
  does its large-prime units in order. A checkpoint stores which items are
  done, the counts of exactly those items, and how many GPU units are done,
  with the GPU's counts. Work in progress is not saved, so a resumed run
  redoes it.
* **The GPU's leaves.** The GPU's counts include the leaf runs that tasks
  hand it. So when a checkpoint is due, tasks started from then on do their
  own leaves. Once the tasks that handed runs over are done, the GPU finishes
  those runs and the checkpoint is written. The workers never wait. A
  checkpoint takes as long as the longest task in hand: under a second at
  `10¹³`, up to tens of seconds at the start of larger searches. So a stop at
  the time limit can come a little late.
* **`results.csv`** has one row per search (program, mode and `N`), rewritten
  at each checkpoint (`running` or `stopped`) and at the end (`complete`).
  The columns are:
  * tasks and large-prime blocks or GPU units done, out of how many;
  * hours searched, over how many sessions;
  * the counts the program prints. Without `-s`, `candidate_pairs` is the
    filter hits; `eight_squares` and `seven_squares` are only counted without
    `-s`.
* **Finds** are printed, and appended to `finds.txt` and `finds.csv`. Work
  redone after a resume reports its finds again.

A checkpoint is only valid for the same program, build, mode and `N`; the
program refuses one that does not match.

## What was already known

From Christian Boyer's research summary
(<http://www.multimagie.com/English/SquaresOfSquaresSearch.htm>, last entries
2020):

* Lee Morgenstern (2006–07) showed that every entry of a solution must exceed
  10¹⁴ (all roots have at least 8 digits).
* Morgenstern (2013–14) searched "three 3-square APs with equal step", the
  structure every 3×3 magic square of squares contains. He found no solution
  with step `d ≤ 2.4·10²¹` (all APs) or `d ≤ 6·10²³` (primitive steps
  `4mn(m²−n²)`, `m < 2²⁴`). Here the steps are the Lucas offsets `a` and `b`,
  and both are below the centre `n²`. So that implies the centre of any
  solution exceeds `2.4·10²¹`, i.e. `n > 4.9·10¹⁰`.
* Many near misses are known. They include 7-square magic squares (Bremner;
  Rathbun found its rank-4 elliptic curve) and semi-magic squares with one
  bad diagonal (Sallows, Rubin's with diagonal ratio 1 + 7·10⁻¹⁷). The Lucas
  family is proved unable to give 8 magic lines.

Everything below centre `4.9·10¹⁰` re-confirms that earlier work by an
independent method. The runs beyond that go past the bounds in that summary,
though later unpublished searches may well have gone further.

## Files

| file | |
|------|---|
| [`mss.c`](mss.c) | the search: enumeration, the three modes, tasks, checkpoints |
| [`mss_gpu.cu`](mss_gpu.cu), [`mss_gpu.h`](mss_gpu.h) | the CUDA port used by `-g` |
| [`Makefile`](Makefile) | builds, `make test`, `make bench` |
| [`tests.sh`](tests.sh) | the regression tests |
| [`results.csv`](results.csv) | one row per completed search, with every count |
| [`runs/`](runs) | the full output of each search in the results table |
