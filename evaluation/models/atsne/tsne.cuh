#pragma once
#include <curand.h>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "handle_cuda_err.hpp"
#include "testing.hpp"

namespace qvis {
namespace tsne {
const float float_max = std::numeric_limits<float>::max();
template <typename T>
void __global__ kernel_fill(T *dst, T value, unsigned N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < N; idx += gridDim.x * blockDim.x) {
        dst[idx] = value;
    }
}

template <int D>
__global__ void kernel_calc_gradient(unsigned N, const float *Y, const float *W, float *grad_output) {
    unsigned     tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float        grad[D], delta[D];
    for (; idx < N; idx += gridDim.x * blockDim.x) { // iter point
#pragma unroll
        for (int d = 0; d < D; d++) {
            grad[d] = 0.0f;
        }
        for (unsigned i = 0; i < N; i++) {
            if (i == idx) {
                continue;
            }

            float dist = 0;
#pragma unroll
            for (int d = 0; d < D; d++) {
                delta[d] = Y[idx * D + d] - Y[i * D + d];
                dist += delta[d] * delta[d];
            }
            dist = W[i * N + idx] / (1 + dist); // W[i * N + idx] == W[idx * N + i]
#pragma unroll
            for (int d = 0; d < D; d++) {
                grad[d] += dist * delta[d];
            }
        }
#pragma unroll
        for (int d = 0; d < D; d++) {
            grad_output[idx * D + d] += grad[d];
        }
    }
}

template <int D>
__global__ void kernel_calc_negative_gradient(unsigned N, const float *Y, float *grad_output, float *sumQ_output) {
    unsigned         tid        = threadIdx.x;
    unsigned int     idx        = blockIdx.x * blockDim.x + tid;
    float            sumQthread = 0;
    __shared__ float sumQblock;
    float            grad[D], delta[D]; // local grad, and tmp delta, should be register

    for (; idx < N; idx += gridDim.x * blockDim.x) { // iter point
#pragma unroll
        for (int d = 0; d < D; d++) {
            grad[d] = 0.0f;
        }
        for (unsigned i = 0; i < N; i++) {
            if (i == idx) {
                continue;
            }
            float dist = 0;
#pragma unroll
            for (int d = 0; d < D; d++) {
                delta[d] = Y[idx * D + d] - Y[i * D + d];
                dist += delta[d] * delta[d];
            }
            dist = 1 / (1 + dist);
            sumQthread += dist;
#pragma unroll
            for (int d = 0; d < D; d++) {
                grad[d] += dist * dist * delta[d];
            }
        }
#pragma unroll
        for (int d = 0; d < D; d++) {
            grad_output[idx * D + d] += grad[d];
        }
    }
    // dealwith sumQ
    __syncthreads();
    if (tid == 0) {
        sumQblock = 0;
    }
    __syncthreads();
    atomicAdd(&sumQblock, sumQthread);
    __syncthreads();
    if (tid == 0) {
        sumQ_output[blockIdx.x] += sumQblock;
    }
}

template <int MaxDim = 1024>
__global__ void kernel_query_distances(int dim, float *base, float *query, float *result, int numbase, int numquery) {
    unsigned int tid = threadIdx.x;
    // unsigned int idx = blockIdx.x * blockDim.x + tid;
    unsigned int baseid = blockIdx.x;
    assert(dim <= MaxDim);
    __shared__ float current_base[MaxDim];
    for (; baseid < numbase; baseid += gridDim.x) {
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            current_base[i] = base[baseid * dim + i];
        }
        __syncthreads();
        for (int q = tid; q < numquery; q += blockDim.x) {
            float r = 0;
            for (int i = 0; i < dim; i++) {
                float d = current_base[i] - query[q * dim + i];
                r += d * d;
            }
            result[q * numbase + baseid] = r;
        }
    }
}

template <int MaxDim = 1024>
__global__ void kernel_query_distances_highdim(int dim, float *base, float *query, float *result, int numbase,
                                               int numquery) {
    unsigned int tid = threadIdx.x;
    // unsigned int idx = blockIdx.x * blockDim.x + tid;
    unsigned int     baseid = blockIdx.x;
    __shared__ float current_base[MaxDim];
    for (; baseid < numbase; baseid += gridDim.x) {
        for (int dbase = 0; dbase < dim; dbase += MaxDim) {
            __syncthreads();
            for (int i = tid; i < MaxDim && i + dbase < dim; i += blockDim.x) {
                current_base[i] = base[baseid * dim + dbase + i];
            }
            __syncthreads();
            for (int q = tid; q < numquery; q += blockDim.x) {
                float r = 0;
                for (int i = 0; i < MaxDim && i + dbase < dim; i++) {
                    float d = current_base[i] - query[q * dim + dbase + i];
                    r += d * d;
                }
                result[q * numbase + baseid] += r;
            }
        }
    }
}
__global__ void kernel_calculate_p(unsigned N, const float *distance, float perplexity, float *W_out) {
    const float  logp = log(perplexity);
    unsigned     tid  = threadIdx.x;
    unsigned int idx  = blockIdx.x * blockDim.x + tid;
    for (; idx < N; idx += gridDim.x * blockDim.x) { // iter point
        float beta = 1e-5;
        float minf = -1;
        float maxf = 1;
        float sumf;
        for (int iter = 0; iter < 200; iter++) {
            sumf    = 0;
            float H = 0;
            for (int i = 0; i < N; i++) {
                float d = distance[idx * N + i];
                float t = exp(-beta * d);
                sumf += t;
                H += beta * (d * t);
            }
            H           = (H / sumf) + log(sumf);
            float Hdiff = H - logp;
            if (fabs(Hdiff) < 1e-5) break;
            if (Hdiff > 0) {
                minf = beta;
                if (maxf < 0)
                    beta *= 2;
                else
                    beta = (beta + maxf) / 2;
            } else {
                maxf = beta;
                if (minf < 0)
                    beta /= 2;
                else
                    beta = (minf + beta) / 2;
            }
            if (beta > float_max) beta = float_max;
        }
        for (int i = 0; i < N; i++) {
            W_out[idx * N + i] = exp(-beta * distance[idx * N + i]) / sumf;
        }
    }
}

__global__ void kernel_calculate_p_symmetric(unsigned N, float *W_out) {
    unsigned     tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    for (; idx < N; idx += gridDim.x * blockDim.x) { // iter point
        for (unsigned i = 0; i < N; i++) {
            if (i <= idx) {
                unsigned row = idx, col = i;
                float    avg         = (W_out[row * N + col] + W_out[col * N + row]) / 2 / N;
                W_out[row * N + col] = avg;
                W_out[col * N + row] = avg;
            }
        }
    }
}

template <int D>
void __global__ kernel_update_Y(unsigned N, float *Y, float *grads, float *grads_old, float *grads_neg, float *gains,
                                float sumQ, float momentum, float learning_rate) {

    unsigned     tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    for (; idx < D * N; idx += gridDim.x * blockDim.x) {
        float grad     = grads[idx] - (grads_neg[idx] / sumQ);
        float grad_old = grads_old[idx];

        float gain = ((grad >= 0) != (grad_old >= 0)) ? (gains[idx] + 0.2) : (gains[idx] * 0.8 + 0.01);
        gains[idx] = gain;

        grad           = momentum * grad_old - learning_rate * grad * gain;
        grads_old[idx] = grad;
        Y[idx] += grad;
    }
}

template <int D>
struct tSNE {
    int       N;
    int       num_block;
    const int ThreadPerBlock = 128;

    float *data_device;
    float *distances;
    float *W;
    float *Y;
    float *grad_pos, *grad_neg, *grad_old;
    float *gains;
    float *neg_sumQ_blocks, *neg_sumQ_blocks_host;

    tSNE(int N, int dim, float *data) {
        this->N = N;
        HANDLE_ERROR(cudaMalloc((void **)&data_device, N * dim * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&W, N * N * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&distances, N * N * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&Y, N * D * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&grad_pos, N * D * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&grad_neg, N * D * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&grad_old, N * D * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&gains, N * D * sizeof(float)));

        num_block = min((N + ThreadPerBlock - 1) / ThreadPerBlock, 50);

        HANDLE_ERROR(cudaMalloc((void **)&neg_sumQ_blocks, num_block * sizeof(float)));
        neg_sumQ_blocks_host = new float[num_block];

        // copy data to device
        HANDLE_ERROR(cudaMemcpy(data_device, data, sizeof(float) * N * dim, cudaMemcpyHostToDevice));
    }

    void calc_W(float dim, float perplexity) {
        // generate distances
        if (dim <= 1024) {
            kernel_query_distances<<<num_block, ThreadPerBlock>>>(dim, data_device, data_device, distances, N, N);
        } else {
            HANDLE_ERROR(cudaMemset(distances, 0, sizeof(float) * N * N));
            kernel_query_distances_highdim<<<num_block, ThreadPerBlock>>>(dim, data_device, data_device, distances, N,
                                                                          N);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        // generate W
        kernel_calculate_p<<<num_block, ThreadPerBlock>>>(N, distances, perplexity, W);
        HANDLE_ERROR(cudaDeviceSynchronize());

        kernel_calculate_p_symmetric<<<num_block, ThreadPerBlock>>>(N, W);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    void clear_grad() {
        // clear grad_old
        HANDLE_ERROR(cudaMemset(grad_old, 0, sizeof(float) * N * D));
        kernel_fill<<<32, ThreadPerBlock>>>(gains, 1.0f, N * D);
    }

    void do_iter(float learning_rate, float momentum) {
        HANDLE_ERROR(cudaMemset(grad_pos, 0, sizeof(float) * N * D));
        HANDLE_ERROR(cudaMemset(grad_neg, 0, sizeof(float) * N * D));
        HANDLE_ERROR(cudaMemset(neg_sumQ_blocks, 0, sizeof(float) * num_block));

        kernel_calc_gradient<D><<<num_block, ThreadPerBlock>>>(N, Y, W, grad_pos);
        kernel_calc_negative_gradient<D><<<num_block, ThreadPerBlock>>>(N, Y, grad_neg, neg_sumQ_blocks);
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(
            cudaMemcpy(neg_sumQ_blocks_host, neg_sumQ_blocks, sizeof(float) * num_block, cudaMemcpyDeviceToHost));
        float neg_sumQ = 0;

        for (int i = 0; i < num_block; i++) {
            neg_sumQ += neg_sumQ_blocks_host[i];
        }

        kernel_update_Y<D><<<num_block, ThreadPerBlock>>>(N, Y, grad_pos, grad_old, grad_neg, gains, neg_sumQ,
                                                          momentum, learning_rate);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    int tsne(float *result, float learning_rate, int max_iter = 1000, float momentum = 0.8,
             std::function<void(int, float *)> callback = [](int, float *) -> void {}) {
        HANDLE_ERROR(cudaDeviceSynchronize());

        // gengerate Y
        curandGenerator_t gen;
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
        CURAND_CALL(curandGenerateNormal(gen, Y, N * D, 0.0, 2.0));

        clear_grad();

        HANDLE_ERROR(cudaDeviceSynchronize());
        callback(0, Y);

        for (int iter = 1; iter <= max_iter; iter++) {
            do_iter(learning_rate, momentum);
            callback(iter, Y);
        }
        HANDLE_ERROR(cudaMemcpy(result, Y, sizeof(float) * N * D, cudaMemcpyDeviceToHost));
        return 0;
    }
    void free_intermediate() {
        HANDLE_ERROR(cudaFree(data_device));
        HANDLE_ERROR(cudaFree(distances));
        HANDLE_ERROR(cudaFree(Y));
    }
    void free_all() {
        free_intermediate();
        delete[] neg_sumQ_blocks_host;
        HANDLE_ERROR(cudaFree(W));
        HANDLE_ERROR(cudaFree(grad_pos));
        HANDLE_ERROR(cudaFree(grad_neg));
        HANDLE_ERROR(cudaFree(grad_old));
        HANDLE_ERROR(cudaFree(gains));
        HANDLE_ERROR(cudaFree(neg_sumQ_blocks));
    }
};

template <int D>
int tsne(unsigned N, int dim, float *data, float *result, float perplexity, float learning_rate, int vis_iter = 1000,
         float momentum = 0.8, std::function<void(int, float *)> callback = [](int, float *) -> void {}) {
    tSNE<D> grad_tSNE(N, dim, data);
    grad_tSNE.calc_W(dim, perplexity);
    grad_tSNE.tsne(result, learning_rate, vis_iter, momentum, callback);
    grad_tSNE.free_all();
    return 0;
}

} // namespace tsne
} // namespace qvis

namespace qvis {
namespace tsne {
namespace test {

void generate_3d_grad(int N, int M, std::vector<float> &data, std::vector<unsigned> &label) {
    data.resize(N * M * 3);
    label.resize(N * M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            data[(i * M + j) * 3]     = i;
            data[(i * M + j) * 3 + 1] = j;
            data[(i * M + j) * 3 + 2] = 0;
            label[i * M + j]          = i;
        }
    }
}
void save_result(const char *filename, unsigned num, unsigned D, float *data, unsigned *label) {
    std::ofstream out(filename);
    for (unsigned i = 0; i < num; i++) {
        for (unsigned j = 0; j < D; j++) {
            out << data[i * D + j] << "\t";
        }
        out << label[i] << "\n";
    }
}
void test_tsne(int N = 100, int M = 100, int vis_iter = 2000, float perplexity = 4000, float learning_rate = 100) {
    // generate data
    std::vector<float>    data;
    std::vector<unsigned> label;
    generate_3d_grad(N, M, data, label);
    //
    // save_path
    std::string save_path = "/tmp/test_qvis_tsne.txt";

    float *                           tsne_result    = new float[N * M * 2];
    auto                              last_timepoint = qvis::test::now();
    std::function<void(int, float *)> save_function  = [&](int iter, float *data) -> void {
        printf("iter %d\n", iter);
        printf("%s %7.4lf ms\n", "", qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));
        last_timepoint = qvis::test::now();

        HANDLE_ERROR(cudaMemcpy(tsne_result, data, sizeof(float) * N * M * 2, cudaMemcpyDeviceToHost));
        char num_buffer[10];
        sprintf(num_buffer, "%08d", iter);
        std::string save_intermedit_path = save_path + "." + num_buffer;
        if (iter % 1 == 0) {
            save_result(save_intermedit_path.c_str(), N * M, 2, tsne_result, label.data());
        }
    };

    qvis::tsne::tsne<2>(N * M, 3, data.data(), tsne_result, perplexity, learning_rate, vis_iter, 0.8, save_function);
    save_result(save_path.c_str(), N * M, 2, tsne_result, label.data());
}

} // namespace test
} // namespace tsne
} // namespace qvis
