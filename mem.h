#pragma once

namespace mem {
    template<typename T>
    struct shared_fragment {
        T* smem;
        int N;
        int M;
        int stride;

        __device__ shared_fragment(char* shared_base, int N, int M)
          : smem(reinterpret_cast<T*>(shared_base)), N(N), M(M), stride(M + (sizeof(T) == 2 ? 8 : 0)) { }

        template<typename accessor>
        __device__ void load(accessor gmem, int tile_x, int tile_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % M;
                int y = i / M;
                smem[y * stride + x] = gmem[y + tile_y * N][x + tile_x * M];
            }
        }

        template<typename accessor>
        __device__ void load_transpose(accessor gmem, int tile_x, int tile_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int y = i % M;
                int x = i / M;
                smem[y * stride + x] = gmem[x + tile_y * N][y + tile_x * M];
            }
        }

        __device__ T& operator()(int x, int y) {
            return smem[y * stride + x];
        }

        template<typename accessor>
        __device__ void store(accessor gmem, int tile_x, int tile_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % M;
                int y = i / M;
                gmem[y + tile_y * N][x + tile_x * M] = smem[y * stride + x];
            }
        }

        __device__ unsigned size() {
            return N * stride;
        }

        __device__ char* next() {
            return reinterpret_cast<char*>(smem + size());
        }
    };
} // namespace mem