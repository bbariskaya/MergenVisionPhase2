#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>

namespace mergenvision {

struct ScoreIndex {
    float score;
    int idx;
};

struct ScoreIndexDesc {
    __host__ __device__ bool operator()(const ScoreIndex& a, const ScoreIndex& b) const {
        if (a.score != b.score) return a.score > b.score;
        return a.idx < b.idx;
    }
};

// Deterministic descending argsort with explicit (score, index) tie-break.
extern "C" int mergenvision_argsort_descending(
    const float* d_scores,  // read-only
    int* d_order,
    int n,
    cudaStream_t stream)
{
    thrust::device_ptr<const float> scores(d_scores);

    ScoreIndex* d_pairs = nullptr;
    cudaError_t err = cudaMalloc(&d_pairs, sizeof(ScoreIndex) * n);
    if (err != cudaSuccess) return err;
    thrust::device_ptr<ScoreIndex> pairs_p(d_pairs);

    thrust::for_each_n(
        thrust::cuda::par.on(stream),
        thrust::make_counting_iterator(0),
        n,
        [scores, pairs_p] __device__ (int i) {
            pairs_p[i] = {scores[i], i};
        });

    thrust::sort(
        thrust::cuda::par.on(stream),
        pairs_p,
        pairs_p + n,
        ScoreIndexDesc());

    thrust::device_ptr<int> order(d_order);
    thrust::for_each_n(
        thrust::cuda::par.on(stream),
        thrust::make_counting_iterator(0),
        n,
        [pairs_p, order] __device__ (int i) {
            ScoreIndex p = pairs_p[i];
            order[i] = p.idx;
        });

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        cudaFree(d_pairs);
        return sync_err;
    }
    cudaFree(d_pairs);
    return cudaGetLastError();
}

} // namespace mergenvision
