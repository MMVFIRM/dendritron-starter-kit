# Build the Dendritron RBF forward-pass HIP kernel for AMD GPUs (gfx803+).
#
# On ROCm 6.x hosts where only gcc-11 libstdc++ headers are installed, pass:
#   make GCC_INSTALL_DIR=/usr/lib/gcc/x86_64-linux-gnu/11

HIPCC ?= hipcc
GCC_INSTALL_DIR ?= /usr/lib/gcc/x86_64-linux-gnu/11
HIPFLAGS = --gcc-install-dir=$(GCC_INSTALL_DIR) -O3 -fPIC -shared

.PHONY: all test bench clean

all: libdendritron_rbf.so

libdendritron_rbf.so: dendritron_rbf.hip.cpp
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: libdendritron_rbf.so
	python3 correctness_test.py

bench: libdendritron_rbf.so
	python3 bench_gpu_vs_cpu.py

clean:
	rm -f libdendritron_rbf.so
