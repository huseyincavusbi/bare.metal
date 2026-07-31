# bare.metal

From-scratch LLM inference + training engine for Apple Silicon. C11 + Metal Shading Language, zero runtime dependencies.

## Features

- SmolLM2 135M / 360M, bit-exact vs PyTorch
- Training with AdamW, mixed precision (bf16/fp16/fp32)
- Q8 quantized inference (2.4x faster)
- Flash Attention, graph fusion, fused kernels

## Requirements

- macOS 13.0+ on Apple Silicon (M1 or later)

## Build & Run

```bash
make baremetal        # inference
make baremetal-train  # training

./build/baremetal run data/smollm2-135m "Once upon a time" --steps 8 --temp 0.7 --top-k 40 --seed 42
./build/baremetal-train train data/smollm2-135m test/grad/g4_text.txt ./checkpoints --steps 500 --seq-len 64 --lr 3e-4
```

## License

MPL-2.0
