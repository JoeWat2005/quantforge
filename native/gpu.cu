#include "model.h"
#include <cuda_runtime.h>
__global__ void simulate(Parameters p, std::uint64_t start, std::uint64_t n, int stride, double* out) {
    auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) scenario(p, start + i, out + i * stride, stride);
}
API int qf_gpu_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}
// Each call is deliberately bounded by host batch size and device free memory.
API int qf_simulate(const Parameters* p, std::uint64_t start, std::uint64_t count,
                    int vram_mib, int stride, double* output) {
    if (!p || !output || p->assets < 1 || p->assets > MAX_ASSETS || p->steps < 1 ||
        (stride != 1 && stride != p->assets + 1)) return -1;
    size_t free_bytes = 0, total = 0;
    auto err = cudaMemGetInfo(&free_bytes, &total);
    if (err != cudaSuccess) return int(err);
    size_t bytes = count * stride * sizeof(double);
    if (vram_mib < 1 || bytes > free_bytes / 2 || bytes > size_t(vram_mib) * 1024 * 1024) return -2;
    double* device = nullptr;
    err = cudaMalloc(&device, bytes);
    if (err != cudaSuccess) return int(err);
    simulate<<<unsigned((count + 255) / 256), 256>>>(*p, start, count, stride, device);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaMemcpy(output, device, bytes, cudaMemcpyDeviceToHost);
    auto release = cudaFree(device);
    return int(err == cudaSuccess ? release : err);
}
