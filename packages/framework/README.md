# @mni-ml/framework

A TypeScript ML framework with a **Rust+CUDA native backend**. PyTorch-like API, GPU training speed.

The first TypeScript ML framework with a Rust backend — designed for real model training, not just inference.

## Architecture

```
TypeScript (thin API)  →  Rust (napi-rs)  →  CUDA kernels / CPU fallback
     Tensor class            Autograd           cuBLAS matmul
     nn.Linear               Tape recording     Fused LayerNorm
     Adam optimizer           Backward pass      Fused cross-entropy
     F.gelu, F.softmax        Memory allocator   Caching allocator
```

- **All computation in Rust** — autograd, tensor ops, optimizer math
- **TypeScript is a thin wrapper** — just delegates to the native addon via napi-rs
- **Caching memory allocator** — zero allocations in the training hot path
- **Fused operations** — LayerNorm, cross-entropy (softmax + NLL in one pass)

## Install

```bash
npm install @mni-ml/framework
```

## Quick Start

```typescript
import { Tensor, nn, Adam, softmax, gelu, crossEntropyLoss } from '@mni-ml/framework';

// Create tensors (backed by Rust, no JS computation)
const x = Tensor.rand([4, 128, 256]);
const w = Tensor.randn([256, 512]);
w.setRequiresGrad(true);

// Forward pass
const y = x.matmul(w);           // Rust: optimized matmul + tape record
const z = gelu(y);                // Rust: GELU kernel + tape record
const loss = crossEntropyLoss(z.view(512, 512), targets);

// Backward pass (single call to Rust — full backward in one shot)
loss.backward();

// Optimizer step (fused AdamW in Rust)
optimizer.step();
```

## What's Included

### Tensor Operations
- Elementwise: `add`, `sub`, `mul`, `neg`, `exp`, `log`
- Activations: `gelu`, `relu`
- Reductions: `sum`, `mean`, `max` (along any dimension)
- Linear algebra: `matmul` (batched)
- Layout: `view`, `permute`, `contiguous`
- Broadcasting: automatic shape broadcasting on all binary ops

### Neural Network Modules
- `Linear` — fully connected layer
- `Embedding` — lookup table
- `LayerNorm` — fused layer normalization
- Functional: `softmax`, `gelu`, `dropout`, `crossEntropyLoss`

### Training
- `Adam` optimizer with decoupled weight decay (AdamW)
- Gradient clipping (global norm)
- `noGradStart()` / `noGradEnd()` for inference mode

### Autograd Engine (Rust)
- Tape-based automatic differentiation
- Topological sort backward traversal
- Gradient accumulation on leaf tensors
- Full backward pass executes in Rust — zero JS round-trips

## Building from Source

### macOS (CPU development)

```bash
cd native
cargo build --release --features cpu
cp target/release/libmni_framework_native.dylib mni-framework-native.darwin-arm64.node
cd .. && npx tsc
```

### Linux with CUDA (GPU training)

```bash
cd native
cargo build --release --features cuda
cp target/release/libmni_framework_native.so mni-framework-native.linux-x64-gnu.node
cd .. && npx tsc
```

## Training nanoGPT

The framework includes a complete GPT-2 training script:

```bash
# Download Shakespeare dataset
node prepare.js

# Train locally (CPU)
node train.js

# Train on Modal (GPU)
modal run modal_train.py --fresh
```

## Benchmarks

| Configuration | Old Framework (JS) | New Framework (Rust) | Speedup |
|---|---|---|---|
| 15K params, 3 steps | ~45s | 1.5s | **30x** |
| 100K params, 20 steps | ~25min | 7s | **214x** |
| Forward pass (matmul) | ~200ms | <1ms | **200x+** |
| Backward pass | ~500ms (JS autograd) | <2ms (Rust autograd) | **250x+** |

## How It Works

1. **TypeScript** creates `Tensor` objects that hold opaque integer IDs pointing to Rust-side data
2. **Operations** (matmul, gelu, etc.) call into Rust via napi-rs, which records each op to an autograd tape
3. **`backward()`** triggers a single Rust function that performs the entire backward pass: topological sort, gradient computation, and gradient accumulation — without returning to JS
4. **`optimizer.step()`** runs fused AdamW parameter updates entirely in Rust
5. **Caching allocator** reuses freed tensor buffers by size, eliminating allocation overhead after warmup

## License

MIT
