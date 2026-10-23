#!/bin/sh
# Regression tests for mss (run by `make test`, about a minute on 16 threads).
# Every check compares two independent computations, or one against counts
# recorded from the original, simpler program.  They run in a temporary
# directory, where the searches write their checkpoints and results.csv.
R=$(cd "$(dirname "$0")" && pwd) || exit 1
T=${THREADS:-16}
fail=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp" || exit 1

ok() { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fail=1; }
same() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 ($2 vs $3)"; fi; }
field() { grep "$1" "$2" | head -1; }

echo "self-test of the pair analysis on synthetic offset sets"
for m in "" -q; do
    "$R"/mss $m selftest > "$tmp/st" 2>&1 && ok "selftest $m" || bad "selftest $m"
done

echo "fast search vs brute force (D(n) by trial, every candidate re-derived)"
# The N vary B = isqrt(N) mod 4, which sets where the large-prime sieve starts.
for n in 1000 1234 5000 30000 77777 100000; do
    "$R"/mss brute $n > "$tmp/b" 2>&1 || bad "brute $n disagrees with its own direct count"
    "$R"/mss $n $T > "$tmp/f"
    for key in 'centres examined' 'offsets total' '7-square near misses' 'MAGIC' 'EIGHT\|8 squares'; do
        same "N=$n: $key" "$(field "$key" "$tmp/b")" "$(field "$key" "$tmp/f")"
    done
    "$R"/mss -q $n $T > "$tmp/q"
    same "N=$n: -q offsets" "$(field 'offsets total' "$tmp/b")" "$(field 'offsets total' "$tmp/q")"
    # -s must check exactly the pairs whose offsets are jointly divisible by
    # every usable prime, here counted by plain division on brute-force D(n).
    "$R"/mss -v brute $n > "$tmp/v"
    "$R"/mss -s $n $T > "$tmp/s"
    same "N=$n: -s candidates" "$(field 'counted directly' "$tmp/v" | sed 's/.*: //')" \
        "$(field 'candidate pairs' "$tmp/s" | sed 's/.*: \([0-9]*\).*/\1/')"
done

echo "larger N: recorded counts and -s against the direct count"
"$R"/mss 1e9 $T > "$tmp/a"
same "1e9 checksum (recorded)" "$(field 'offsets total' "$tmp/a")" \
    "offsets total: 437593604   pairs checked: 5317728049   checksum: 883d7b4c8cdc1295"
"$R"/mss-wide 1e9 $T > "$tmp/w"
same "1e9 128-bit build" "$(field 'offsets total' "$tmp/a")" "$(field 'offsets total' "$tmp/w")"
same "1e9 near misses, 128-bit build" "$(field '7-square' "$tmp/a")" "$(field '7-square' "$tmp/w")"
"$R"/mss -v 1e8 $T > "$tmp/v"
"$R"/mss -s 1e8 $T > "$tmp/s"
same "1e8 -s candidates = direct count" "$(field 'counted directly' "$tmp/v" | sed 's/.*: //')" \
    "$(field 'candidate pairs' "$tmp/s" | sed 's/.*: \([0-9]*\).*/\1/')"
"$R"/mss -s 2e9 $T > "$tmp/s"
same "2e9 -s candidates (recorded)" "$(field 'candidate pairs' "$tmp/s" | sed 's/.*: \([0-9]*\).*/\1/')" 13640004
"$R"/mss-wide -s 2e9 $T > "$tmp/s2"
same "2e9 -s, 128-bit build" "$(field 'candidate pairs' "$tmp/s")" "$(field 'candidate pairs' "$tmp/s2")"

if [ -x "$R"/mss-gpu ]; then
    echo "GPU (-g) against the CPU (-s): every count"
    for n in 1e6 1e8 2e9 2e10; do
        "$R"/mss-wide -s $n $T > "$tmp/c"
        "$R"/mss-gpu -g $n $T > "$tmp/g" 2>&1 || bad "-g $n failed: $(tail -1 "$tmp/g")"
        for key in 'centres examined' 'offsets total' 'candidate pairs' 'of which' 'MAGIC' 'repeated'; do
            same "N=$n -g: $key" "$(field "$key" "$tmp/c")" "$(field "$key" "$tmp/g")"
        done
    done
    # Below about 10^10 the CPU threads finish before the GPU starts on their
    # leaves; at 10^11 it takes most of them.  Counts recorded from -s.
    "$R"/mss-gpu -g 1e11 $T > "$tmp/g" 2>&1 || bad "-g 1e11 failed: $(tail -1 "$tmp/g")"
    same "N=1e11 -g: leaves on the GPU" "$(grep -c 'then [1-9][0-9]* leaves' "$tmp/g")" 1
    same "N=1e11 -g: centres (recorded)" "$(field 'centres examined' "$tmp/g")" \
        "centres examined (|D(n)| >= 3): 4440149081"
    same "N=1e11 -g: offsets (recorded)" "$(field 'offsets total' "$tmp/g")" \
        "offsets total: 49350733738   pairs: 906281915180"
    same "N=1e11 -g: candidates (recorded)" "$(field 'candidate pairs' "$tmp/g" | sed 's/.*: \([0-9]*\).*/\1/')" \
        1222703907
    same "N=1e11 -g: n^2 + a + b squares (recorded)" "$(field 'of which' "$tmp/g" | sed 's/.*: //')" 17281
fi

# A search stopped by its time limit (exit status 3) and resumed, perhaps
# more than once, must count exactly what one uninterrupted run does.
echo "checkpoints: searches stopped and resumed"
ck() {  # ck NAME RECORDED-OUTPUT PROGRAM ARGS: stopped at 2 s and 5 s, then resumed to the end
    name=$1 ref=$2 prog=$3
    shift 3
    "$prog" -t 2 "$@" > "$tmp/k1" 2>&1
    s1=$?
    "$prog" -t 5 "$@" > "$tmp/k2" 2>&1
    s2=$?
    "$prog" "$@" > "$tmp/k3" 2>&1
    s3=$?
    same "$name: exit statuses (stopped, stopped, done)" "$s1 $s2 $s3" "3 3 0"
    same "$name: resumed twice" "$(grep -c '^resuming' "$tmp/k2" "$tmp/k3" | tr '\n' ' ')" "$tmp/k2:1 $tmp/k3:1 "
    for key in 'centres examined' 'offsets total' 'candidate pairs' 'of which' 'MAGIC' 'repeated'; do
        same "$name: $key" "$(field "$key" "$ref")" "$(field "$key" "$tmp/k3")"
    done
    [ -e "$tmp"/*.ckpt ] && bad "$name: checkpoint left after the end"
    grep -q ",complete,.*,3,$(field 'centres examined' "$ref" | sed 's/.*: //')," results.csv ||
        bad "$name: results.csv has no complete row of 3 sessions"
}
"$R"/mss-wide -s 1e11 $T > "$tmp/ref" 2>&1
ck "-s 1e11" "$tmp/ref" "$R"/mss-wide -s 1e11 $T
[ -x "$R"/mss-gpu ] && ck "-g 1e12 (recorded)" "$R"/runs/g_le_1e12.log "$R"/mss-gpu -g 1e12 $T

[ $fail = 0 ] && echo "all tests passed" || echo "SOME TESTS FAILED"
exit $fail
