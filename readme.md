## Experiments with flash cosine similarity attention

Main repo: https://github.com/lucidrains/flash-cosine-sim-attention

```python
# O = softmax(scale * (Q * K^T) - scale) * V
def plain_impl(Q: TensorType['b', 'i', 'd'],
               K: TensorType['b', 'j', 'd'],
               V: TensorType['b', 'j', 'd'],
               scale=8) -> TensorType['b', 'i', 'j']:
    C = einsum('... i d, ... j d -> ... i j', Q, K)
    C = (C * scale - scale).softmax(dim = -1)
    O = einsum('... i j, ... j d -> ... i d', C, V)
    return O
```

Implements ideas from https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

The goal is to match pytorch performance while keeping the cuda code simple to understand.

Current results on RTX2070:
```
--------------------------------------------------------------------------------
batch: 32       head dim: 64    V dim: 64               dtype: torch.float32
--------------------------------------------------------------------------------
seq_len: 64     slower: 0.64x   kernel:   0.086ms       baseline:   0.134ms
seq_len: 128    slower: 1.02x   kernel:   0.142ms       baseline:   0.140ms
seq_len: 256    slower: 1.13x   kernel:   0.390ms       baseline:   0.347ms
seq_len: 512    slower: 1.18x   kernel:   1.432ms       baseline:   1.209ms
seq_len: 1024   slower: 1.02x   kernel:   5.140ms       baseline:   5.042ms
seq_len: 2048   slower: 0.94x   kernel:  17.211ms       baseline:  18.235ms
seq_len: 4096   slower: 0.00x   kernel:  69.789ms       baseline:       OOM
seq_len: 8192   slower: 0.00x   kernel: 279.815ms       baseline:       OOM
--------------------------------------------------------------------------------
batch: 32       head dim: 64    V dim: 64               dtype: torch.float16
--------------------------------------------------------------------------------
seq_len: 64     slower: 0.56x   kernel:   0.078ms       baseline:   0.140ms
seq_len: 128    slower: 0.59x   kernel:   0.097ms       baseline:   0.164ms
seq_len: 256    slower: 1.17x   kernel:   0.209ms       baseline:   0.178ms
seq_len: 512    slower: 1.02x   kernel:   0.562ms       baseline:   0.552ms
seq_len: 1024   slower: 0.95x   kernel:   1.931ms       baseline:   2.032ms
seq_len: 2048   slower: 0.80x   kernel:   7.147ms       baseline:   8.928ms
seq_len: 4096   slower: 0.82x   kernel:  27.829ms       baseline:  34.134ms
seq_len: 8192   slower: 0.00x   kernel: 109.877ms       baseline:       OOM
```

Results on A100 (old):
```
slower: 0.320x   seq_len: 64    fused kernel: 0.048     baseline: 0.152
slower: 0.352x   seq_len: 128   fused kernel: 0.053     baseline: 0.152
slower: 0.574x   seq_len: 256   fused kernel: 0.123     baseline: 0.215
slower: 0.673x   seq_len: 512   fused kernel: 0.303     baseline: 0.450
slower: 0.746x   seq_len: 1024  fused kernel: 0.960     baseline: 1.288
slower: 0.793x   seq_len: 2048  fused kernel: 3.616     baseline: 4.562
slower: 0.733x   seq_len: 4096  fused kernel: 13.212    baseline: 18.014
                 seq_len: 8192  fused kernel: 51.390    baseline: OOM
                 seq_len: 16384 fused kernel: 202.860   baseline: OOM
                 seq_len: 32768 fused kernel: 815.142   baseline: OOM
```

## Building the image to run in HPC cluster

On local computer (need root access):

```bash
sudo singularity build container.sif container.def
```

Run interactive shell:
```bash
singularity shell --nv container.sif
```

Install & run:
```bash
python3 setup.py install --user
python3 benchmark.py
```