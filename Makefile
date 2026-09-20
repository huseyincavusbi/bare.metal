# bare.metal Makefile
# Inference: make baremetal      (no BAREMETAL_TRAIN)
# Training:  make baremetal-train (with -DBAREMETAL_TRAIN)

BINARY  := build/baremetal
BINARY_TRAIN := build/baremetal-train
BENCH_BIN := build/bench/bench

TEST_SRCS := $(wildcard test/*_test.c)
TEST_BINS := $(patsubst test/%.c,build/test/%,$(TEST_SRCS))

CC      := clang
CFLAGS  := -std=c11 -Wall -Wextra -Werror -O2 -I include -I src
CFLAGS_TRAIN := $(CFLAGS) -DBAREMETAL_TRAIN
LDFLAGS := -framework Metal -framework Foundation

OBJC_SRCS := src/backend/metal/device.m \
             src/backend/metal/dispatch.m
C_SRCS    := src/utils/log.c \
             src/baremetal/context.c \
             src/baremetal/model.c \
             src/baremetal/checkpoint.c \
             src/baremetal/checkpoint_st.c \
             src/baremetal/api.c \
             src/baremetal/tokenizer.c \
             src/baremetal/graph.c \
             src/baremetal/compiler.c \
             src/baremetal/scheduler.c \
             src/baremetal/session.c \
              src/baremetal/quant.c \
              src/baremetal/sampler.c \
              src/baremetal/trainer.c \
             src/kernels/registry.c

TOOL_SRCS := tools/cli.c tools/run.c

OBJS      := $(patsubst %.c,build/%.o,$(C_SRCS)) $(patsubst %.m,build/%.o,$(OBJC_SRCS))
TOOL_OBJS := $(patsubst %.c,build/%.o,$(TOOL_SRCS))

OBJS_TRAIN      := $(patsubst %.c,build/train/%.o,$(C_SRCS)) $(patsubst %.m,build/train/%.o,$(OBJC_SRCS))
TOOL_OBJS_TRAIN := $(patsubst %.c,build/train/%.o,$(TOOL_SRCS))

METAL_SRC := src/kernels/forward.metal src/kernels/backward.metal src/anneal/adamw.metal
METAL_BIN := build/kernels/default.metallib

.PHONY: all clean baremetal baremetal-train dirs tests bench

all: dirs baremetal

baremetal: dirs $(BINARY)

baremetal-train: dirs $(BINARY_TRAIN)

tests: dirs $(TEST_BINS)

bench: dirs $(BENCH_BIN)

$(BENCH_BIN): bench/bench.c $(OBJS) $(METAL_BIN)
	@mkdir -p build/bench
	$(CC) $(CFLAGS) -o $@ bench/bench.c $(OBJS) $(LDFLAGS)
	@mkdir -p build/bench/kernels
	@ln -sf ../../kernels/default.metallib build/bench/kernels/default.metallib
	@echo "  Built: $@"

dirs:
	@mkdir -p build/src/utils
	@mkdir -p build/src/baremetal
	@mkdir -p build/src/backend/metal
	@mkdir -p build/tools
	@mkdir -p build/kernels
	@mkdir -p build/src/kernels
	@mkdir -p build/test
	@mkdir -p build/train/src/utils
	@mkdir -p build/train/src/baremetal
	@mkdir -p build/train/src/backend/metal
	@mkdir -p build/train/tools
	@mkdir -p build/train/src/kernels

$(BINARY): $(OBJS) $(TOOL_OBJS) $(METAL_BIN)
	$(CC) $(CFLAGS) $(OBJS) $(TOOL_OBJS) -o $@ $(LDFLAGS)
	@echo "  Built: $@ (inference)"

$(BINARY_TRAIN): $(OBJS_TRAIN) $(TOOL_OBJS_TRAIN) $(METAL_BIN)
	$(CC) $(CFLAGS_TRAIN) $(OBJS_TRAIN) $(TOOL_OBJS_TRAIN) -o $@ $(LDFLAGS)
	@echo "  Built: $@ (training)"

build/test/%: test/%.c $(OBJS_TRAIN) $(METAL_BIN)
	@mkdir -p build/test
	$(CC) $(CFLAGS_TRAIN) -c $< -o $@.o
	$(CC) $(CFLAGS_TRAIN) $@.o $(OBJS_TRAIN) -o $@ $(LDFLAGS)
	@mkdir -p build/test/kernels
	@ln -sf ../../kernels/default.metallib build/test/kernels/default.metallib
	@echo "  Built: $@"

build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

build/train/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_TRAIN) -c $< -o $@

build/%.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

build/train/%.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_TRAIN) -fobjc-arc -c $< -o $@

$(METAL_BIN): $(METAL_SRC)
	@mkdir -p build/kernels
	xcrun -sdk macosx metal -c src/kernels/forward.metal -o build/kernels/forward.air
	xcrun -sdk macosx metal -c src/kernels/backward.metal -o build/kernels/backward.air
	xcrun -sdk macosx metal -c src/anneal/adamw.metal -o build/kernels/adamw.air
	xcrun -sdk macosx metallib build/kernels/forward.air build/kernels/backward.air build/kernels/adamw.air -o $(METAL_BIN)
	@echo "  Compiled Metal shaders → $(METAL_BIN)"

clean:
	rm -rf build
	@echo "  Cleaned"
