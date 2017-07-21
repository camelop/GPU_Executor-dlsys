#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */
__global__ void reduce_sum_axis_zero_kernel(const float* input, float* output, int length, int size) {
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        int sum = 0;
        int upper = size * length;
        for (int i=0; i<upper; i+=length) sum += input[y+i];
        output[y] = sum;
}

__global__ void broadcast_to_kernel(const float* input, float* output, int length) {
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        int val = input[y];
        int s = y * length;
        for (int i=0; i<length; i++) output[s+i] = val;
}

__global__ void array_set_kernel(int size, float* array, float value) {
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        array[y] = value;
}

__global__ void matrix_softmax_kernel(int nrow, int ncol,
                                      const float* input,
                                      float* output) {
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= nrow) return;
        input += y * ncol;
        output += y * ncol;
        float maxval = *input;
        for (int x=1; x < ncol; ++x) maxval = max(maxval, input[x]);
        float sum = 0;
        for (int x = 0; x < ncol; ++x) sum += exp(input[x] - maxval);
        for (int x = 0; x < ncol; ++x) output[x] = exp(input[x]) / sum;
}


// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
        // Dynamic shared memory, size provided at kernel launch.
        extern __shared__ float loss_per_row[];
        // Two dimensional thread blocks.
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= nrow) {
                return;
        }
        input_a += y * ncol;
        input_b += y * ncol;
        float maxval = *input_a;
        // Find max for a row.
        for (int x = 1; x < ncol; ++x) {
                maxval = max(maxval, input_a[x]);
        }
        // Deduct by max for a row, and raise to exp.
        float sum = 0;
        for (int x = 0; x < ncol; ++x) {
                sum += exp(input_a[x] - maxval);
        }
        // Compute per-row loss.
        float loss = 0;
        for (int x = 0; x < ncol; ++x) {
                loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
        }
        loss_per_row[y] = loss;
        __syncthreads();
        // Compute reduce_mean across rows.
        float mean_loss = 0;
        // Use a single thread to reduce mean across rows.
        if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
                for (int i = 0; i < nrow; ++i) {
                        mean_loss += loss_per_row[i];
                }
                mean_loss /= nrow;
                output[0] = mean_loss;
        }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
        int size = arr->shape[0];
        float *array = (float *)arr->data;
        float val = value;
        dim3 threads;
        if (size <= 1024) {
                threads.x = size;
        } else {
                if (size < 1024 * 1024) {
                        threads.x = 1024;
                        threads.y = size / 1024;
                } else {
                        threads.x = 1024;
                        threads.y = 1024;
                        threads.z = size / 1024 / 1024;
                }
        }
        array_set_kernel<<<1, threads >>>(size, array, val);
        return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
        int size = input->shape[0];
        int ndim = input->ndim;
        int length = 1;
        for (int i=1; i<ndim; i++) length*=input->shape[i];
        const float* input_data = (const float*) input->data;
        float* output_data = (float*) output->data;
        dim3 threads;
        threads.x = nrow % 1024;
        int nblocks = nrow / 1024;
        broadcast_to_kernel <<< nblocks, threads >>> (input_data, output_data, length);
        return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
        int size = input->shape[0];
        int ndim = input->ndim;
        int length = 1;
        for (int i=1; i<ndim; i++) length*=input->shape[i];
        const float* input_data = (const float*) input->data;
        float* output_data = (float*) output->data;
        dim3 threads;
        threads.x = length % 1024;
        int nblocks = length / 1024;
        reduce_sum_axis_zero_kernel <<< nblocks, threads >>> (input_data, output_data, length, size);
        return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
        /* TODO: Your code here */
        return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
        /* TODO: Your code here */
        return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
        /* TODO: Your code here */
        return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
        /* TODO: Your code here */
        return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
        /* TODO: Your code here */
        // Hint: use cublas
        // cublas assume matrix is column major
        return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
        /* TODO: Your code here */
        return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
        /* TODO: Your code here */
        return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
        /* DONE: My code here */
        assert(input->ndim == 2);
        assert(output->ndim == 1);
        int nrow = input->shape[0];
        assert(nrow <= 1024 * 4);
        int ncol = input->shape[1];
        const float *input_data = (const float *)input->data;
        float *output_data = (float *)output->data;
        dim3 threads;
        threads.x = nrow % 1024;
        int nblocks = nrow / 1024;
        matrix_softmax_kernel<<<nblocks, threads>>>(
                nrow, ncol, input_data, output_data);
        return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
        assert(input_a->ndim == 2);
        assert(input_b->ndim == 2);
        assert(output->ndim == 1);
        assert(input_a->shape[0] == input_b->shape[0] &&
               input_a->shape[1] == input_b->shape[1]);
        int nrow = input_a->shape[0];
        // Maximum x- or y-dimension of a block = 1024
        // But we need 'nrow' shared memory, and max shared memory is 48KB.
        // Conservatively allow max 16KB shared memory.
        assert(nrow <= 1024 * 4);
        int ncol = input_a->shape[1];
        const float *input_data_a = (const float *)input_a->data;
        const float *input_data_b = (const float *)input_b->data;
        float *output_data = (float *)output->data;
        dim3 threads;
        threads.x = nrow % 1024;
        int nblocks = nrow / 1024;
        // 1 block, each block with 'threads' number of threads with 'nrow' shared
        // memory size
        matrix_softmax_cross_entropy_kernel<<<nblocks, threads >>>(
                nrow, ncol, input_data_a, input_data_b, output_data);
        return 0;
}
