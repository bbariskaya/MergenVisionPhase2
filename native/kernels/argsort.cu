#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace mergenvision {

// Sorts a *writable* copy of scores in-place; returns order indices.
extern "C" int mergenvision_argsort_descending(
    float* d_scores,  // modified in-place
    int* d_order,
    int n,
    cudaStream_t stream)
{
    thrust::device_ptr<float> keys(d_scores);
    thrust::device_ptr<int> vals(d_order);
    thrust::sequence(thrust::cuda::par.on(stream), vals, vals + n, 0);
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        keys,
        keys + n,
        vals,
        thrust::greater<float>());
    return cudaGetLastError();
}

} // namespace mergenvision
