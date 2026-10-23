CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra -Wpointer-arith
THREADS ?= 16

all: mss mss-wide

mss: mss.c
	$(CC) $(CFLAGS) -o $@ $< -lm

mss-wide: mss.c
	$(CC) $(CFLAGS) -DWIDE -o $@ $< -lm

# 128-bit build with -g: the large-prime phase and most leaves of the small-prime
# search on an NVIDIA GPU (CUDA toolkit).
CUDA ?= /usr/local/cuda
CUDA_ARCH ?= sm_75
gpu: mss-gpu
mss-gpu: mss.c mss_gpu.cu mss_gpu.h
	$(CUDA)/bin/nvcc -O3 -arch=$(CUDA_ARCH) -c -o mss_gpu.o mss_gpu.cu
	$(CC) $(CFLAGS) -DWIDE -DGPU -o $@ mss.c mss_gpu.o -L$(CUDA)/lib64 -Wl,-rpath,$(CUDA)/lib64 -lcudart -lstdc++ -lpthread -lm

# Correctness: brute force, direct counts, recorded results and resumed runs;
# with mss-gpu built (make gpu), also -g against -s.
test: all
	THREADS=$(THREADS) ./tests.sh

# Speed: each mode at N = 2e9, then the solutions-only search at 2e10 and 1e11.
bench: all
	@for m in "" -q -s; do printf 'mss %-3s 2e9:   ' "$$m"; ./mss $$m 2e9 $(THREADS) | grep 'search time'; done
	@printf 'mss-wide -s 2e10: '; ./mss-wide -s 2e10 $(THREADS) | grep 'search time'
	@printf 'mss-wide -s 1e11: '; ./mss-wide -s 1e11 $(THREADS) | grep 'search time'

clean:
	rm -f mss mss-wide mss-gpu mss_gpu.o

.PHONY: all gpu test bench clean
