#include "maxrows.cuh"

static __global__ void max_rows_f32_kernel(const float * __restrict__ x, float * __restrict__ dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float max_val = -FLT_MAX;
    for (int i = col; i < ncols; i += blockDim.x) {
        max_val = fmaxf(max_val, x[row * ncols + i]);
    }

    max_val = warp_reduce_max(max_val);
    if (blockDim.x > WARP_SIZE) {
        __shared__ float s_max[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_max[warp_id] = max_val;
        }
        __syncthreads();
        max_val = -FLT_MAX;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            max_val = s_max[lane_id];
        }
        max_val = warp_reduce_max(max_val);
    }

    if (col == 0) {
        dst[row] = max_val;
    }
}

void max_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        max_rows_f32_kernel<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        max_rows_f32_kernel<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    }
}

void ggml_cuda_op_max_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        max_rows_f32_kernel<<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        max_rows_f32_kernel<<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    }
}
