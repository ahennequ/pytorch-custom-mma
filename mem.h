#pragma once

namespace mem {
    template<typename T, int N_tile, int M_tile>
    struct shared_fragment {
        static constexpr int N = N_tile;
        static constexpr int M = M_tile;
        static constexpr int stride = M + (sizeof(T) == 2 ? 8 : 1);
        static constexpr int size = N * stride;
        
        T* smem;

        __device__ shared_fragment(char* shared_base)
          : smem(reinterpret_cast<T*>(shared_base)) { }

        template<typename accessor>
        __device__ void load(accessor gmem, int tile_x, int tile_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % M;
                int y = i / M;
                smem[y * stride + x] = gmem[y + tile_y][x + tile_x];
            }
        }

        template<typename accessor>
        __device__ void load_transpose(accessor gmem, int tile_x, int tile_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % N;
                int y = i / N;
                smem[x * stride + y] = gmem[y + tile_y][x + tile_x];
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
                gmem[y + tile_y][x + tile_x] = smem[y * stride + x];
            }
        }

        __device__ char* next() {
            return reinterpret_cast<char*>(smem + size);
        }
    };
} // namespace mem