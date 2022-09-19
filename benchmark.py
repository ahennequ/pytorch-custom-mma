import torch
import argparse
from torchtyping import TensorType
from torch import einsum

from torch.cuda import synchronize, Event
from functools import wraps, partial
import torch.nn.functional as F

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--only-forwards', default = True, action = 'store_true')
parser.add_argument('--only-backwards', default = False, action = 'store_true')
args = parser.parse_args()

assert not (args.only_forwards and args.only_backwards)

torch.manual_seed(0)

# constants
TEST_SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

BATCH_SIZE = 32
DIM = 64

TEST_FORWARDS = not args.only_backwards
TEST_BACKWARDS = not args.only_forwards

timer = partial(Event, enable_timing = True)

def benchmark(
    fn,
    *,
    num_times = 10,
    warmup_iters = 10,
    forwards = True,
    backwards = False
):
    assert forwards or backwards

    @wraps(fn)
    def inner(*args, **kwargs):
        # warmup
        for _ in range(warmup_iters):
            loss = fn(*args, **kwargs).sum()
            #loss.backward()

        # average across number of function calls
        all_measured_times_ms = 0.

        for _ in range(num_times):
            start_event = timer()
            end_event = timer()

            if forwards:
                start_event.record()

            o = fn(*args, **kwargs)

            if not backwards:
                end_event.record()

            if not forwards:
                start_event.record()

            if backwards:
                loss = o.sum()
                loss.backward()
                end_event.record()

            synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
            all_measured_times_ms += elapsed_time_ms

        return all_measured_times_ms / num_times

    return inner

# O = softmax(scale * (Q * K^T) - scale) * V
def plain_impl(Q: TensorType['b', 'i', 'd'],
               K: TensorType['b', 'j', 'd'],
               V: TensorType['b', 'j', 'd'],
               scale=8) -> TensorType['b', 'i', 'j']:
    C = einsum('... i d, ... j d -> ... i j', Q, K)
    C = (C * scale - scale).softmax(dim = -1)
    O = einsum('... i j, ... j d -> ... i d', C, V)
    return O

import pytorch_custom_mma_cuda

class MMACudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=8):
        C, l = pytorch_custom_mma_cuda.forward(Q, K, V, scale)
        ctx.save_for_backward(Q, K, V, C, l)
        return C
    @staticmethod
    def backward(ctx, grad_c):
        #Q, K, V, C, l = ctx.saved_tensors # TODO
        return None, None

# O = softmax(scale * (Q * K^T) - scale) * V
def cuda_impl(Q: TensorType['b', 'i', 'd'],
              K: TensorType['b', 'j', 'd'],
              V: TensorType['b', 'j', 'd'],
              scale=8) -> TensorType['b', 'i', 'j']:
    return MMACudaFunction.apply(Q, K, V, scale)

plain_fn = benchmark(
    plain_impl,
    forwards = TEST_FORWARDS,
    backwards = TEST_BACKWARDS
)

cuda_fn = benchmark(
    cuda_impl,
    forwards = TEST_FORWARDS,
    backwards = TEST_BACKWARDS
)

def allclose(a, b, atol = 1e-2):
    diff = (a - b).abs().amax()
    return diff <= atol

def l2norm(t):
    return F.normalize(t, dim = -1)

def bench(dtype=torch.float32):
    print("Benchmark", dtype)
    for seq_len in TEST_SEQUENCE_LENGTHS:
        Q = torch.randn(BATCH_SIZE, seq_len, DIM, dtype=dtype).cuda().requires_grad_()
        K = torch.randn(BATCH_SIZE, seq_len, DIM, dtype=dtype).cuda().requires_grad_()
        V = torch.randn(BATCH_SIZE, seq_len, DIM, dtype=dtype).cuda().requires_grad_()
        #V = torch.ones(BATCH_SIZE, seq_len, DIM, dtype=dtype).cuda().requires_grad_()

        Q, K = map(l2norm, (Q, K))

        if (seq_len <= 2048):
            # assert correctness
            C_plain = plain_impl(Q, K, V)
            C_cuda  = cuda_impl(Q, K, V)

            #print(C_plain, C_plain.shape)
            #torch.set_printoptions(profile="full")
            #print(C_cuda, C_cuda.shape)
            #torch.set_printoptions(profile="default") # reset
            assert allclose(C_plain, C_cuda)

        # benchmark
        fused_time = cuda_fn(Q, K, V)
        baseline_time = plain_fn(Q, K, V) if seq_len <= 2048 else 10000

        times_slower = fused_time / baseline_time

        print(f'slower: {times_slower:.3f}x\t seq_len: {seq_len}\tfused kernel: {fused_time:.3f}\tbaseline: {baseline_time:.3f}')

bench(dtype=torch.float32)
bench(dtype=torch.float16)