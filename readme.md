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

Results on RTX2070:
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

Results on A100:
```
--------------------------------------------------------------------------------
batch: 32       head dim: 64    V dim: 64               dtype: torch.float32
--------------------------------------------------------------------------------
seq_len: 64     slower: 0.50x   kernel:   0.082ms       baseline:   0.165ms
seq_len: 128    slower: 0.56x   kernel:   0.092ms       baseline:   0.164ms
seq_len: 256    slower: 0.88x   kernel:   0.160ms       baseline:   0.182ms
seq_len: 512    slower: 0.65x   kernel:   0.325ms       baseline:   0.496ms
seq_len: 1024   slower: 0.73x   kernel:   1.084ms       baseline:   1.489ms
seq_len: 2048   slower: 0.63x   kernel:   3.362ms       baseline:   5.371ms
seq_len: 4096   slower: 0.57x   kernel:  12.065ms       baseline:  21.270ms
seq_len: 8192   slower: 0.00x   kernel:  46.413ms       baseline:       OOM
seq_len: 16384  slower: 0.00x   kernel: 180.894ms       baseline:       OOM
seq_len: 32768  slower: 0.00x   kernel: 744.898ms       baseline:       OOM
--------------------------------------------------------------------------------
batch: 32       head dim: 64    V dim: 64               dtype: torch.float16
--------------------------------------------------------------------------------
seq_len: 64     slower: 0.41x   kernel:   0.066ms       baseline:   0.160ms
seq_len: 128    slower: 0.38x   kernel:   0.077ms       baseline:   0.201ms
seq_len: 256    slower: 0.63x   kernel:   0.102ms       baseline:   0.161ms
seq_len: 512    slower: 1.05x   kernel:   0.170ms       baseline:   0.162ms
seq_len: 1024   slower: 0.71x   kernel:   0.372ms       baseline:   0.521ms
seq_len: 2048   slower: 0.77x   kernel:   1.427ms       baseline:   1.851ms
seq_len: 4096   slower: 0.69x   kernel:   4.944ms       baseline:   7.167ms
seq_len: 8192   slower: 0.63x   kernel:  18.202ms       baseline:  29.042ms
seq_len: 16384  slower: 0.00x   kernel:  68.439ms       baseline:       OOM
seq_len: 32768  slower: 0.00x   kernel: 262.238ms       baseline:       OOM
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