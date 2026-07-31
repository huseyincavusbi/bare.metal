# bare.metal Makefile
# Inference: make baremetal
# Training:  make baremetal-train (adds BAREMETAL_TRAIN=1)

BINARY  := build/baremetal
BINARY_TRAIN := build/baremetal-train

TEST_SRCS := $(wildcard test/*_test.c)
TEST_BINS := $(patsubst test/%.c,build/test/%,$(TEST_SRCS))

CC      := clang
CFLAGS  := -std=c11 -Wall -Wextra -O2 -I include -I src -DBAREMETAL_TRAIN
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

OBJS      := $(patsubst %.c,build/%.o,$(C_SRCS))
OBJS      += $(patsubst %.m,build/%.o,$(OBJC_SRCS))
TOOL_OBJS := $(patsubst %.c,build/%.o,$(TOOL_SRCS))

METAL_SRC := src/kernels/forward.metal src/kernels/backward.metal src/anneal/adamw.metal
METAL_BIN := build/kernels/default.metallib

.PHONY: all clean baremetal baremetal-train dirs tests

all: dirs baremetal

baremetal: dirs $(BINARY)

baremetal-train: dirs $(BINARY_TRAIN)

tests: dirs $(TEST_BINS)

dirs:
	@mkdir -p build/src/utils
	@mkdir -p build/src/baremetal
	@mkdir -p build/src/backend/metal
	@mkdir -p build/tools
	@mkdir -p build/kernels
	@mkdir -p build/src/kernels
	@mkdir -p build/test

$(BINARY): $(OBJS) $(TOOL_OBJS) $(METAL_BIN)
	$(CC) $(CFLAGS) $(OBJS) $(TOOL_OBJS) -o $@ $(LDFLAGS)
	@echo "  Built: $@"

$(BINARY_TRAIN): $(OBJS) $(TOOL_OBJS) $(METAL_BIN)
	$(CC) $(CFLAGS) $(OBJS) $(TOOL_OBJS) -o $@ $(LDFLAGS)
	@echo "  Built: $@ (with training)"

build/test/%: test/%.c $(OBJS) $(METAL_BIN)
	$(CC) $(CFLAGS) -c $< -o $@.o
	$(CC) $(CFLAGS) $@.o $(OBJS) -o $@ $(LDFLAGS)
	@mkdir -p build/test/kernels
	@ln -sf ../../kernels/default.metallib build/test/kernels/default.metallib
	@echo "  Built: $@"

build/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: %.m
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

$(METAL_BIN): $(METAL_SRC)
	xcrun -sdk macosx metal -c src/kernels/forward.metal -o build/kernels/forward.air
	xcrun -sdk macosx metal -c src/kernels/backward.metal -o build/kernels/backward.air
	xcrun -sdk macosx metal -c src/anneal/adamw.metal -o build/kernels/adamw.air
	xcrun -sdk macosx metallib build/kernels/forward.air build/kernels/backward.air build/kernels/adamw.air -o $(METAL_BIN)
	@echo "  Compiled Metal shaders → $(METAL_BIN)"

clean:
	rm -rf build
	@echo "  Cleaned"
