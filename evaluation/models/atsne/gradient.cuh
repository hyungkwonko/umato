#pragma once

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#include "graph.hpp"
#include "handle_cuda_err.hpp"
#include "testing.hpp"

namespace qvis {

// CUDA Utils
const unsigned  warp_size = 32;
__forceinline__ __device__ unsigned get_lane_id() {
    unsigned ret;
    asm("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned get_warp_id() {
    unsigned ret;
    asm("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

// GPU kernel: calculate centers of cluster
// N: number of points
// CenterPerBlock < 6144 for D = 2 and 48k shared memory per block
template <int D, int ClusterPerBlock, int ThreadPerBlock>
__global__ static void kernel_calc_centers(unsigned N,
                                           unsigned NC, // number of points, number of clusters, number of threadblock
                                           const float *   Y, // N *D, low dimension
                                           const unsigned *cluser_cap,
                                           const unsigned *cluster, // capacity of cluster, which cluster point belong
                                           float *         centers_out // NC * D centers
) {
    static_assert(ClusterPerBlock * D * sizeof(float) <= 48 * 1024, "Too many center(s) per block");
    assert(ThreadPerBlock == blockDim.x);

    __shared__ float centers[ClusterPerBlock * D];

    unsigned center_base = blockIdx.x * ClusterPerBlock;
    unsigned tid         = threadIdx.x;
    for (; center_base < NC; center_base += gridDim.x * ClusterPerBlock) {
        // clear local centers
        for (unsigned i = 0; i < ClusterPerBlock * D; i += blockDim.x) {
            if (i + tid < ClusterPerBlock * D) {
                centers[i + tid] = 0.0f;
            }
        }
        __syncthreads();

        // accumulate centers
        for (unsigned i = 0; i < N * D; i += blockDim.x) { // Scan Y
            if (i + tid < N * D) {
                unsigned cluster_id = cluster[(i + tid) / D];
                // cluster belong to this block
                if (cluster_id >= center_base && cluster_id < center_base + ClusterPerBlock) {
                    // The following line will lead to wrong result
                    // centers[(cluster_id % ClusterPerBlock) * D + (i + tid) % D] += Y[i + tid];
                    atomicAdd(centers + (cluster_id % ClusterPerBlock) * D + (i + tid) % D, Y[i + tid]);
                }
            }
            __syncthreads();
        }

        // output
        for (unsigned i = 0; i < ClusterPerBlock * D && center_base * D + i < NC * D; i += blockDim.x) {
            if (i + tid < ClusterPerBlock * D && center_base * D + i + tid < NC * D) {
                centers_out[center_base * D + i + tid] = centers[i + tid] / cluser_cap[center_base + (i + tid) / D];
            }
        }

    } // end loop centerbase
}

// GPU kernel: calculate centers of cluster using register
// N: number of points
// CenterPerBlock < 6144 for D = 2 and 48k shared memory per block
template <int D, int ThreadPerBlock>
__global__ static void kernel_calc_centers_register(
    unsigned N, unsigned NC,                             // number of points, number of clusters, number of threadblock
    const float *   Y,                                   // N *D, low dimension
    const unsigned *cluser_cap, const unsigned *cluster, // capacity of cluster, which cluster point belong
    float *centers_out                                   // NC * D centers
) {
    assert(ThreadPerBlock == blockDim.x);

    float            centers[D];
    __shared__ float block_centers[D];

    unsigned center_base = blockIdx.x;
    unsigned tid         = threadIdx.x;
    for (; center_base < NC; center_base += gridDim.x) {
        // clear local centers
#pragma unroll
        for (unsigned i = 0; i < D; i++) {
            centers[i] = 0.0f;
        }

        // accumulate centers
        for (unsigned i = 0; i < N * D; i += ThreadPerBlock) { // Scan Y
            if (i + tid < N * D) {
                unsigned cluster_id = cluster[(i + tid) / D];
                // cluster belong to this block
                if (cluster_id == center_base) {
#pragma unroll
                    for (int d = 0; d < D; d++) {
                        centers[d] += ((i + tid) % D == d) * Y[i + tid];
                    }
                }
            }
        }

        // output
        for (unsigned d = 0; d < D; d++) {
            block_centers[d] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (unsigned d = 0; d < D; d++) {
            atomicAdd(block_centers + d, centers[d]);
        }
        __syncthreads();

        for (unsigned i = 0; i < D; i += ThreadPerBlock) {
            if (i + tid < D) {
                centers_out[center_base * D + i + tid] =
                    block_centers[i + tid] / cluser_cap[center_base + (i + tid) / D];
            }
        }

    } // end loop centerbase
}

template <int D, typename IndexType, bool DebugGrad = false>
__global__ void kernel_update_sgd(
    float *                      Y,           // low dimension points,
    const unsigned *             sgd_mapping, // random index of batchs
    const float *                centers,     // centers
    const Graph<IndexType, true> G,           // graph between Y and Y, degree first
    const Graph<float, true>     W,        // weight(p_ij - q_ij) between Y and Y, same dimension as NG, degree first
    const float                  np_coeff, // coefficient of neighbor positive
    const float                  nn_coeff, // coefficient of neighbor negative
    const Graph<IndexType, true> CG,       // graph between Y and center, degree first
    const float                  cp_coeff, // coefficient of center positive
    const Graph<float, true>     CW, // weight(p_ij - q_ij) between Y and center, same dimension as NG, degree first
    const Graph<IndexType, true> NG, // negitive graph between Y and Y, degree first
    const float                  sn_coeff,      // coefficient of negative sampling negative
    const float                  sumQ,          // sum[(1 + (x - y)^2)^-1]
    const float                  learning_rate, // learning rate
    float *                      sumQ_output,   // SumQ of block
    float *positive_grad, float *negative_grad) {
    __shared__ float sumQblock;
    float            sumQthread = 0;
    float            grad[D], delta[D]; // local grad, and tmp delta, should be register

    const unsigned  batch_size = warp_size;
    const unsigned &n          = G.n();
    unsigned        tid        = threadIdx.x;

    // unsigned lane_id = get_lane_id();
    // unsigned warp_id = get_warp_id();
    unsigned lane_id = tid % 32;
    unsigned warp_id = tid / 32;
    unsigned batchs  = (n + batch_size - 1) / batch_size;
    unsigned bid     = blockIdx.x * blockDim.x / warp_size + warp_id; // batch id,

    for (; bid < batchs; bid += blockDim.x * gridDim.x / warp_size) { // scan batchs
        unsigned idx = sgd_mapping[bid] * batch_size + lane_id;
        if (idx >= n) {
            continue;
        }
#pragma unroll
        for (int d = 0; d < D; d++) {
            grad[d] = 0.0f;
        }

        // scan knn
        for (int i = 0; i < G.d(); i++) {
            int nn = G[i][idx];     // degree first
            if (nn == 0xffffffff) { // not enough point in IVF, we will get this
                continue;
            }
            float dist = 0;
#pragma unroll
            for (int d = 0; d < D; d++) {
                delta[d] = Y[idx * D + d] - Y[nn * D + d];
                dist += delta[d] * delta[d];
            }
            // positive and negative grad of KNN
            float ndist = nn_coeff / (1 + dist);
#pragma unroll
            for (int d = 0; d < D; d++) {
                float pos = np_coeff * W[i][idx] / (1 + dist) * delta[d];
                float neg = ndist * ndist / sumQ * delta[d];
                grad[d] += pos;
                grad[d] -= neg;
                if (DebugGrad) {
                    positive_grad[idx * D + d] += pos;
                    negative_grad[idx * D + d] -= neg;
                }
            }
            sumQthread += ndist; // thread sumQ
        }                        // end knn

        // scan centers
        for (int i = 0; i < CG.d(); i++) {
            int nn = CG[i][idx];    // degree first
            if (nn == 0xffffffff) { // not enough point in IVF, we will get this
                continue;
            }
            float dist = 0;
#pragma unroll
            for (int d = 0; d < D; d++) {
                delta[d] = Y[idx * D + d] - centers[nn * D + d];
                dist += delta[d] * delta[d];
            }
            // positive grad of K center
#pragma unroll
            for (int d = 0; d < D; d++) {
                float pos = cp_coeff * CW[i][idx] / (1 + dist) * delta[d];
                ;
                grad[d] += pos;
                if (DebugGrad) {
                    positive_grad[idx * D + d] += pos;
                }
            }
        } // end centers

        // scan negative sampling
        for (int i = 0; i < NG.d(); i++) { // scan neighbor
            int nn = NG[i][idx];           // degree first
            if (nn == 0xffffffff) {        // not enough point in IVF, we will get this
                continue;
            }
            float dist = 0;
#pragma unroll
            for (int d = 0; d < D; d++) {
                delta[d] = Y[idx * D + d] - Y[nn * D + d];
                dist += delta[d] * delta[d];
            }

            dist = sn_coeff * 1 / (1 + dist);

            if (dist != 1.0) {
                sumQthread += dist; // thread sumQ
            }

#pragma unroll
            for (int d = 0; d < D; d++) {
                float neg = dist * dist * delta[d] / sumQ;
                grad[d] -= neg;
                if (DebugGrad) {
                    negative_grad[idx * D + d] -= neg;
                }
            }

        } // end negative sampling

        // update
        float length = 0.0;
#pragma unroll
        for (int d = 0; d < D; d++) {
            length += grad[d] * grad[d];
        }
        if (length > 25) { // clip grad
#pragma unroll
            for (int d = 0; d < D; d++) {
                // grad clip
                Y[idx * D + d] -= grad[d] * 5 / sqrtf(length) * learning_rate;
            }
        } else {
            for (int d = 0; d < D; d++) {
                Y[idx * D + d] -= grad[d] * learning_rate;
            }
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

template <int D, typename IndexType, bool MulCoeff = false>
__global__ void kernel_calc_gradient(
    const float *                Y,      // low dimension points
    const float *                Target, // low dimension points, or centers
    const Graph<IndexType, true> NG,     // graph between Y and Target, degree first
    const Graph<float, true>     W,     // weight(p_ij - q_ij) between Y and Target, same dimension as NG, degree first
    const float                  coeff, // coefficient, enabled if MulCoeff = True
    float *                      grad_output // output(add to it)
) {
    const unsigned &n = NG.n();

    float grad[D], delta[D]; // local grad, and tmp delta, should be register

    unsigned     tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += blockDim.x * gridDim.x) { // scan points
#pragma unroll
        for (int d = 0; d < D; d++) {
            grad[d] = 0.0f;
        }

        for (int i = 0; i < NG.d(); i++) { // scan neighbor
            int nn = NG[i][idx];           // degree first
            if (nn == 0xffffffff) {        // not enough point in IVF, we will get this
                continue;
            }
            float dist = 0;
#pragma unroll
            for (int d = 0; d < D; d++) {
                delta[d] = Y[idx * D + d] - Target[nn * D + d];
                dist += delta[d] * delta[d];
            }

            dist = W[i][idx] / (1 + dist);

            if (MulCoeff) {
                dist *= coeff;
            }

#pragma unroll
            for (int d = 0; d < D; d++) {
                grad[d] += delta[d] * dist;
            }

        } // end neighbor

#pragma unroll
        for (int d = 0; d < D; d++) {
            grad_output[idx * D + d] += grad[d];
        }

    } // end scan points
}

template <int D, typename IndexType, bool MulCoeff = false>
__global__ void
    kernel_calc_negative_gradient(const float *                Y,           // low dimension points
                                  const Graph<IndexType, true> NG,          // graph between Y and Target, degree first
                                  const float                  coeff,       // coefficient, enabled if MulCoeff = True
                                  float *                      grad_output, // output(add to it)
                                  float *                      sumQ_output  // SumQ of block
    ) {
    const unsigned &n = NG.n();

    __shared__ float sumQblock;
    float            sumQthread = 0;
    float            grad[D], delta[D]; // local grad, and tmp delta, should be register

    unsigned     tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    for (; idx < n; idx += blockDim.x * gridDim.x) { // scan points
#pragma unroll
        for (int d = 0; d < D; d++) {
            grad[d] = 0.0f;
        }

        for (int i = 0; i < NG.d(); i++) { // scan neighbor
            int nn = NG[i][idx];           // degree first
            if (nn == 0xffffffff) {        // not enough point in IVF, we will get this
                continue;
            }
            float dist = 0;
#pragma unroll
            for (int d = 0; d < D; d++) {
                delta[d] = Y[idx * D + d] - Y[nn * D + d];
                dist += delta[d] * delta[d];
            }

            if (MulCoeff) {
                dist = coeff * 1 / (1 + dist);
            } else {
                dist = 1 / (1 + dist);
            }
            if (dist != 1.0) {
                sumQthread += dist; // thread sumQ
            }

#pragma unroll
            for (int d = 0; d < D; d++) {
                grad[d] += dist * dist * delta[d];
            }

        } // end neighbor

#pragma unroll
        for (int d = 0; d < D; d++) {
            grad_output[idx * D + d] += grad[d];
        }

    } // end scan points

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

template <typename T>
void __global__ kernel_fill(T *dst, T value, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < N; idx += gridDim.x * blockDim.x) {
        dst[idx] = value;
    }
}

template <typename T>
void __global__ kernel_mod(T *dst, T mod, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < N; idx += gridDim.x * blockDim.x) {
        dst[idx] = dst[idx] % mod;
    }
}

template <typename IndexType, int ClusterPerBlock, int ThreadPerBlock>
__global__ void kenel_calc_cluster_cap(unsigned N, int NC, const IndexType *cluster, unsigned *cluster_cap_out) {
    static_assert(ClusterPerBlock * sizeof(unsigned) <= 48 * 1024, "Too many center(s) per block");
    assert(ThreadPerBlock == blockDim.x);

    __shared__ unsigned cluster_cap[ClusterPerBlock];

    unsigned center_base = blockIdx.x * ClusterPerBlock;
    unsigned tid         = threadIdx.x;
    for (; center_base < NC; center_base += gridDim.x * ClusterPerBlock) {
        // clear local cluster_cap
        for (unsigned i = 0; i < ClusterPerBlock; i += ThreadPerBlock) {
            if (i + tid < ClusterPerBlock) {
                cluster_cap[i + tid] = 0.0f;
            }
        }
        __syncthreads();

        // accumulate cluster_cap
        for (unsigned i = 0; i < N; i += ThreadPerBlock) { // Scan Y
            if (i + tid < N) {
                unsigned cluster_id = cluster[i + tid];
                // cluster belong to this block
                if (cluster_id >= center_base && cluster_id < center_base + ClusterPerBlock) {
                    atomicAdd(cluster_cap + cluster_id % ClusterPerBlock, 1);
                }
            }
            __syncthreads();
        }

        // output
        for (unsigned i = 0; i < ClusterPerBlock && center_base + i < NC; i += ThreadPerBlock) {
            if (i + tid < ClusterPerBlock && center_base + i + tid < NC) {
                cluster_cap_out[center_base + i + tid] = cluster_cap[i + tid];
            }
        }

    } // end loop centerbase
}

template <int D // low dimension
          >
struct GradientCalculater {
    // Note: you can't access elements of this struct in **kernel**
    // because this struct are not in GPU memory

    unsigned  N;           // Host, number of points
    unsigned  NC;          // Host, number of clusters
    unsigned *cluster;     // GPU, point belong to which cluster
    unsigned *cluster_cap; // GPU, number of points in clusters

    float *cluster_centers; // GPU, NC * D, center of each clusters
    float *Y;               // GPU, N * D, lowdimension points
    float *grad, *grad_old, *grad_neg;
    float *gain; // GPU, N * D, gain

    float *neg_sumQ_blocks, *neg_sumQ;

    unsigned *sgd_mapping;

    Graph<unsigned, true> graph, cgraph, neg_graph; // CPU struct, GPU data,nearest graph & nearest center graph
    // Weight(p_ij - q_ij), same shape as graph, cgraph, include pitch
    // Just use the struct of Graph
    Graph<float, true> weight, cweight; // CPU struct, GPU data, nearest graph & nearest center graph

    GradientCalculater() {
        cluster         = nullptr;
        cluster_cap     = nullptr;
        Y               = nullptr;
        cluster_centers = nullptr;
        grad            = nullptr;
        grad_old        = nullptr;
        grad_neg        = nullptr;
    }

    // convert clusters to flat array
    void set_cluster(const std::vector<std::vector<int>> clusters) {
        NC = clusters.size();
        std::vector<unsigned> cluster_cap_host(clusters.size());
        size_t                N = 0;
        for (size_t i = 0; i < clusters.size(); i++) {
            N += clusters[i].size();
            cluster_cap_host[i] = clusters[i].size();
        }
        assert(N == this->N); // number of points within clusters not equl with N
        HANDLE_ERROR(cudaMalloc((void **)&cluster_cap, sizeof(unsigned) * clusters.size()));
        HANDLE_ERROR(cudaMemcpy(cluster_cap, cluster_cap_host.data(), sizeof(unsigned) * clusters.size(),
                                cudaMemcpyHostToDevice));

        std::vector<int> cluster_host(N);
        for (size_t i = 0; i < clusters.size(); i++) {
            for (size_t j = 0; j < clusters[i].size(); j++) {
                cluster_host[clusters[i][j]] = i;
            }
        }
        HANDLE_ERROR(cudaMalloc((void **)&cluster, sizeof(int) * N));
        HANDLE_ERROR(cudaMemcpy(cluster, cluster_host.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
    }

    // calculate cluster_cap by cluster
    void calc_cluster_cap() {
        HANDLE_ERROR(cudaMalloc((void **)&cluster_cap, sizeof(unsigned) * NC));
        HANDLE_ERROR(cudaMemset(cluster_cap, 0, sizeof(unsigned) * NC));
        const int ClusterPerBlock = 512;
        const int ThreadPerBlock  = 128;
        int       num_block       = (NC + ClusterPerBlock - 1) / ClusterPerBlock;
        kenel_calc_cluster_cap<unsigned, ClusterPerBlock, ThreadPerBlock>
            <<<num_block, ThreadPerBlock>>>(N, NC, cluster, cluster_cap);
    }

    // copy Y from Host to GPU
    void set_Y(const float *Y_) {
        if (Y != nullptr) {
            cudaFree(Y);
        }
        HANDLE_ERROR(cudaMalloc((void **)&Y, sizeof(float) * N * D));
        HANDLE_ERROR(cudaMemcpy(Y, Y_, sizeof(float) * N * D, cudaMemcpyHostToDevice));
    }

    //
    void set_graph_and_weight(const std::vector<std::vector<unsigned>> &graph_host,  // nearest graph
                              const std::vector<std::vector<unsigned>> &cgraph_host, // nearest center graph
                              const float *                             W // n * (graph.size()  + cgraph.size() )
    ) {
        graph.set_graph_gpu(graph_host);
        cgraph.set_graph_gpu(cgraph_host);
        weight.set_graph_gpu(W, graph.n(), graph.d(), graph.d() + cgraph.d());
        cweight.set_graph_gpu(W + graph.d(), graph.n(), graph.d(), graph.d() + cgraph.d());
    }

    void set_graph(const std::vector<std::vector<unsigned>> &graph_host, // nearest graph
                   const std::vector<std::vector<unsigned>> &cgraph_host) {
        graph.set_graph_gpu(graph_host);
        cgraph.set_graph_gpu(cgraph_host);
    }

    void set_graph(qvis::Graph<unsigned, true> &point_graph, qvis::Graph<unsigned, true> &center_graph) {
        graph.set_to_gpu(point_graph);
        cgraph.set_to_gpu(center_graph);
    }

    void set_weight(qvis::MatrixPitched<float> &W_point, qvis::MatrixPitched<float> &W_center) {
        weight.set_to_gpu(W_point);
        cweight.set_to_gpu(W_center);
    }

    // set negative sampling graph
    void set_neg_graph(const std::vector<std::vector<unsigned>> &g) { neg_graph.set_graph_gpu(g); }

    void init_gain() {
        const int num_block = 32;
        kernel_fill<float><<<num_block, 256>>>(gain, 1.0f, N * D);
    }

    void free_all() {
        if (cluster != nullptr) {
            cudaFree(cluster);
        }
        if (cluster_cap != nullptr) {
            cudaFree(cluster_cap);
        }
        if (cluster_centers != nullptr) {
            cudaFree(cluster_centers);
        }
        // We allso free GPU Y
        if (Y != nullptr) {
            cudaFree(Y);
        }
        if (grad != nullptr) {
            cudaFree(grad);
        }
        if (grad_old != nullptr) {
            cudaFree(grad_old);
        }

        graph.free();
        cgraph.free();
        weight.free();
        cweight.free();
    }
};

} // namespace qvis
namespace qvis {
namespace test {

void test_calc_centers(int num_cluster = 250, int num_point = 5 * 1000 * 1000) {
    constexpr int D = 2;
    time_point    gpu_start, cpu_start;
    double        gpu_duration, cpu_duration; // in millseconds
    srand(0);
    std::vector<float>            prior_center(num_cluster * D);
    std::vector<int>              cluster(num_point);
    std::vector<std::vector<int>> clusters(num_cluster);
    std::vector<float>            Y(num_point * D);

    // generate prior centers
    for (auto it = prior_center.begin(); it != prior_center.end(); it++) {
        *it = 10 * ((float(rand()) / RAND_MAX) - 0.5);
    }

    // generate cluster
    for (size_t i = 0; i < cluster.size(); i++) {
        cluster[i] = rand() % num_cluster;
        clusters[cluster[i]].push_back(i);
        for (int j = 0; j < D; j++) {
            Y[i * D + j] = prior_center[cluster[i] * D + j];
        }
    }

    // generate points
    for (auto it = Y.begin(); it != Y.end(); it++) {
        *it += 10 * ((float(rand()) / RAND_MAX) - 0.5);
    }

    GradientCalculater<D> grad;
    grad.N  = num_point;
    grad.NC = num_cluster;
    grad.set_cluster(clusters);
    grad.set_Y(Y.data());

    // run calc centers kernel
    const int ClusterPerBlock = 10;
    const int ThreadPerBlock  = 128;
    int       num_block       = (num_cluster + ClusterPerBlock - 1) / ClusterPerBlock;
    HANDLE_ERROR(cudaMalloc((void **)&grad.cluster_centers, sizeof(float) * num_cluster * D));
    gpu_start = now();
    if (ClusterPerBlock == 1) {
        kernel_calc_centers_register<D, ThreadPerBlock><<<num_block, ThreadPerBlock>>>(
            grad.N, grad.NC, grad.Y, grad.cluster_cap, grad.cluster, grad.cluster_centers);
    } else {
        kernel_calc_centers<D, ClusterPerBlock, ThreadPerBlock><<<num_block, ThreadPerBlock>>>(
            grad.N, grad.NC, grad.Y, grad.cluster_cap, grad.cluster, grad.cluster_centers);
    }

    // copy back
    CudaCheckError();
    HANDLE_ERROR(cudaDeviceSynchronize());
    gpu_duration = getmilliseconds(gpu_start, now());
    std::vector<float> centers_gpu(num_cluster * D);
    HANDLE_ERROR(
        cudaMemcpy(centers_gpu.data(), grad.cluster_centers, sizeof(float) * num_cluster * D, cudaMemcpyDeviceToHost));

    // CPU
    std::vector<float> centers_cpu(num_cluster * D);
    cpu_start = now();
    std::fill(centers_cpu.begin(), centers_cpu.end(), 0.0f);
    for (size_t i = 0; i < clusters.size(); i++) {
        for (size_t j = 0; j < clusters[i].size(); j++) {
            for (int d = 0; d < D; d++) {
                centers_cpu[i * D + d] += Y[clusters[i][j] * D + d];
            }
        }
        for (int d = 0; d < D; d++) {
            centers_cpu[i * D + d] /= clusters[i].size();
        }
    }
    cpu_duration = getmilliseconds(cpu_start, now());

    // compare
    for (int i = 0; i < num_cluster * D; i++) {
        printf("%8.5f %8.5f %8.5f\n", prior_center[i], centers_cpu[i], centers_gpu[i]);
    }
    printf("GPU %7.4lf ms, CPU %7.4lf ms\n", gpu_duration, cpu_duration);
    cudaFree(grad.cluster_centers);
}

void test_calc_gradient(int num_point = 1000 * 1000, int num_neighbor = 30) {
    constexpr int D = 2;
    time_point    gpu_start, cpu_start;
    double        gpu_duration, cpu_duration; // in millseconds

    srand(0);

    std::vector<std::vector<unsigned>> graph(num_point);
    std::vector<float>                 W(num_point * num_neighbor);
    std::vector<float>                 Y(num_point * D);

    // generate graph
    for (int i = 0; i < num_point; i++) {
        for (int j = 0; j < num_neighbor; j++) { // neighbor
            graph[i].push_back(rand() % num_point);
        }
    }

    // generate weight
    for (int i = 0; i < num_point * num_neighbor; i++) {
        W[i] = (float(rand()) / RAND_MAX) - 0.5;
    }

    // generate Y
    for (auto it = Y.begin(); it != Y.end(); it++) {
        *it += 10 * ((float(rand()) / RAND_MAX) - 0.5);
    }

    GradientCalculater<D> grad;
    grad.N = num_point;

    std::vector<std::vector<unsigned>> dummy_empty_graph;
    grad.set_graph_and_weight(graph, dummy_empty_graph, W.data());
    grad.set_Y(Y.data());

    // run kernel
    const int num_block      = 50;
    const int ThreadPerBlock = 128;
    HANDLE_ERROR(cudaMalloc((void **)&grad.grad, sizeof(float) * num_point * D));
    gpu_start = now();
    kernel_calc_gradient<D, unsigned>
        <<<num_block, ThreadPerBlock>>>(grad.Y, grad.Y, grad.graph, grad.weight, 1, grad.grad);

    // copy back
    CudaCheckError();
    HANDLE_ERROR(cudaDeviceSynchronize());
    gpu_duration = getmilliseconds(gpu_start, now());
    std::vector<float> grad_gpu(num_point * D);
    HANDLE_ERROR(cudaMemcpy(grad_gpu.data(), grad.grad, sizeof(float) * num_point * D, cudaMemcpyDeviceToHost));

    // CPU // copy from qvis.cpp
    std::vector<float> grad_cpu(num_point * D);
    std::fill(grad_cpu.begin(), grad_cpu.end(), 0.0f);
    cpu_start = now();
    for (int n = 0; n < num_point; n++) {
        unsigned KK = graph[n].size();

        for (unsigned i = 0; i < graph[n].size(); i++) {
            float    dist = 0;
            unsigned nn   = graph[n][i];

            for (unsigned j = 0; j < D; j++) {
                float ftmp = (Y[n * D + j] - Y[nn * D + j]);
                dist += ftmp * ftmp;
            }

            unsigned coeff = 1;
            dist           = coeff * W[n * KK + i] / (1 + dist);

            for (unsigned d = 0; d < D; d++) {
                grad_cpu[n * D + d] += dist * (Y[n * D + d] - Y[nn * D + d]);
            }
        }
    }
    cpu_duration = getmilliseconds(cpu_start, now());

    // compare
    for (int i = 0; i < num_point * D; i++) {
        if (grad_cpu[i] - grad_gpu[i] > 1e-5 || grad_cpu[i] - grad_gpu[i] < -1e-5) {
            printf("%8.5f %8.5f ERROR\n", grad_cpu[i], grad_gpu[i]);
        }
    }

    for (size_t i = 0; i < 20; i++) {
        printf("%8.5f %8.5f\n", grad_cpu[i], grad_gpu[i]);
    }
    printf("GPU %7.4lf ms, CPU %7.4lf ms\n", gpu_duration, cpu_duration);
    cudaFree(grad.grad);
}

void test_calc_cluster_cap(int num_cluster = 250, int num_point = 5 * 1000 * 1000) {
    srand(0);
    std::vector<int> cluster(num_point);

    for (int i = 0; i < num_point; i++) {
        cluster[i] = rand() % num_cluster;
    }

    GradientCalculater<2> grad;
    grad.N  = num_point;
    grad.NC = num_cluster;

    HANDLE_ERROR(cudaMalloc((void **)&grad.cluster, sizeof(unsigned) * num_point));
    HANDLE_ERROR(cudaMemcpy(grad.cluster, cluster.data(), sizeof(unsigned) * num_point, cudaMemcpyHostToDevice));

    // run kernel
    grad.calc_cluster_cap();

    // copyback
    std::vector<int> cluster_cap_gpu(num_cluster);
    HANDLE_ERROR(
        cudaMemcpy(cluster_cap_gpu.data(), grad.cluster_cap, sizeof(unsigned) * num_cluster, cudaMemcpyDeviceToHost));

    // CPU
    std::vector<int> cluster_cap_cpu(num_cluster);
    std::fill(cluster_cap_cpu.begin(), cluster_cap_cpu.end(), 0);
    for (size_t i = 0; i < cluster.size(); i++) {
        cluster_cap_cpu[cluster[i]]++;
    }

    // compare
    bool error = false;
    for (int i = 0; i < num_cluster; i++) {
        if (cluster_cap_cpu[i] != cluster_cap_gpu[i]) {
            printf("%6d %6d ERROR\n", cluster_cap_cpu[i], cluster_cap_gpu[i]);
            error = true;
        }
    }
    if (error == false) {
        printf("PASS: test_calc_cluster_cap\n");
    }

    cudaFree(grad.cluster);
    cudaFree(grad.cluster_cap);
}

} // namespace test
} // namespace qvis
