#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int mergenvision_retinaface_decode_batch(
    const float* d_loc,
    const float* d_conf,
    const float* d_landms,
    const float* d_priors,
    int batch,
    int num_anchors,
    float conf_threshold,
    float variance0,
    float variance1,
    int max_candidates,
    float* d_out_boxes,
    float* d_out_scores,
    float* d_out_landmarks,
    int* d_counters,
    cudaStream_t stream);

int mergenvision_argsort_descending(
    const float* d_scores,
    int* d_order,
    int n,
    cudaStream_t stream);

int mergenvision_nms(
    const float* d_boxes,
    const float* d_scores,
    const int* d_order,
    int n,
    float iou_threshold,
    float score_threshold,
    uint8_t* d_keep,
    cudaStream_t stream);

int mergenvision_scale_clip_compact_xy(
    const float* d_boxes,
    const float* d_landmarks,
    const float* d_scores,
    const int* d_order,
    const uint8_t* d_keep,
    int n,
    float scale_x,
    float scale_y,
    int img_w,
    int img_h,
    float score_threshold,
    float* d_out_boxes,
    float* d_out_landmarks,
    float* d_out_scores,
    int* d_out_count,
    cudaStream_t stream);

int mergenvision_l2_normalize(
    const float* d_input,
    float* d_output,
    int rows,
    int cols,
    float epsilon,
    int* d_status,
    cudaStream_t stream);

int mergenvision_similarity_transform(
    const float* d_landmarks,
    float* d_matrices,
    int n,
    int size,
    int* d_status,
    cudaStream_t stream);

int mergenvision_warp_align(
    const uint8_t* d_src,
    int h,
    int w,
    const float* d_matrices,
    int n,
    float* d_dst,
    cudaStream_t stream);

cudaError_t mergenvision_warp_align_rgba_pitch(
    const uint8_t* const* d_surface_ptrs,
    const int* d_surface_indices,
    const int* d_pitches,
    const int* d_widths,
    const int* d_heights,
    const float* d_matrices,
    int n,
    float* d_dst,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
