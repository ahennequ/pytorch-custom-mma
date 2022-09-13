## Experiment with custom attention

```python
# C = exp(scale * (Q * B^T)) * V
def plain_impl(Q: TensorType['b', 'i', 'd'],
               K: TensorType['b', 'j', 'd'],
               V: TensorType['b', 'j', 'd'],
               scale=8) -> TensorType['b', 'i', 'j']:
    C = einsum('... i d, ... j d -> ... i j', Q, K).float()
    C = torch.exp(C * scale)
    O = einsum('... i j, ... j d -> ... i d', C, V).float()
    return O
```

Implements ideas from https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

The goal is to match pytorch performance while keeping the cuda code simple to understand.

Current results on A100:
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

Current results on RTX2070:
```
slower: 0.433x   seq_len: 64    fused kernel: 0.051     baseline: 0.118
slower: 1.092x   seq_len: 128   fused kernel: 0.132     baseline: 0.120
slower: 1.516x   seq_len: 256   fused kernel: 0.456     baseline: 0.301
slower: 1.567x   seq_len: 512   fused kernel: 1.629     baseline: 1.039
slower: 1.498x   seq_len: 1024  fused kernel: 5.687     baseline: 3.797
slower: 1.452x   seq_len: 2048  fused kernel: 21.326    baseline: 14.690
                 seq_len: 4096  fused kernel: 85.707    baseline: OOM
                 seq_len: 8192  fused kernel: 365.019   baseline: OOM
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