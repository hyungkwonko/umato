#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <curand.h>
#include <omp.h>

// faiss
#include "vendor/faiss/IndexIVFFlat.h"
#include "vendor/faiss/IndexIVFPQ.h"
#include "vendor/faiss/gpu/GpuIndexFlat.h"
#include "vendor/faiss/gpu/GpuIndexIVF.h"
#include "vendor/faiss/gpu/GpuIndexIVFFlat.h"
#include "vendor/faiss/gpu/GpuIndexIVFPQ.h"
#include "vendor/faiss/gpu/StandardGpuResources.h"
#include "vendor/faiss/gpu/impl/FlatIndex.cuh"
#include "vendor/faiss/gpu/utils/CopyUtils.cuh"
#include "vendor/faiss/index_io.h"

// cmdline
#include "vendor/cmdline/cmdline.h"

// qvis
#include "gradient.cuh"
#include "graph.hpp"
#include "handle_cuda_err.hpp"
#include "qvis_faiss_patch.cuh"
#include "qvis_io.h"
#include "testing.hpp"
#include "tsne.cuh"
#include "weight.cuh"

using namespace std;

// @data_stride data stride for vector, like dim
void build_nn_graph(const faiss::gpu::GpuIndexIVF *index, const float *data, int data_stride, int points_num, int k,
                    qvis::Graph<unsigned, true> &graph) {
    if (k == 0) {
        printf("Warning: build_cn_graph, k = 0\n");
        return;
    }
    const size_t batch_memory = 1 << 30; // 1GB
    int          batch_size   = batch_memory / sizeof(int) / (k + 1);
    float *      distances; // FIXME: we don't need distances, there should be a patch to faiss
    long *       indicates; // FIXME: we perfer int

    HANDLE_ERROR(cudaMallocHost((void **)&distances, batch_size * (k + 1) * sizeof(float)));
    HANDLE_ERROR(cudaMallocHost((void **)&indicates, batch_size * (k + 1) * sizeof(long)));
    for (int batch = 0; batch < points_num; batch += batch_size) {
        int this_batch_size = std::min(points_num, batch + batch_size) - batch;
        index->search(this_batch_size, data + batch * data_stride, k + 1, distances, indicates);
        HANDLE_ERROR(cudaDeviceSynchronize()); // FIXME: I don't know why we should wait there, but if not, we may get
                                               // some zero in result

        for (int p = 0; p < this_batch_size; p++) {
            int offset = 0;
            for (int d = 0; d < k; d++) {
                if (indicates[p * (k + 1) + offset] == batch + p) { // the same point of query should be ignored
                    offset++;
                }
                graph[d][batch + p] = indicates[p * (k + 1) + offset];
                offset++;
            }
        }
    }

    HANDLE_ERROR(cudaFreeHost(distances));
    HANDLE_ERROR(cudaFreeHost(indicates));
}

// @data_stride data stride for vector, like dim
void build_cn_graph(const faiss::gpu::GpuIndexFlat *index, const float *data, int data_stride, int points_num, int k,
                    qvis::Graph<unsigned, true> &cgraph) {
    if (k == 0) {
        printf("Warning: build_cn_graph, k = 0\n");
        return;
    }
    const size_t batch_memory = 1 << 30; // 1GB
    int          batch_size   = batch_memory / sizeof(int) / k;
    int *        indicates;

    const faiss::gpu::qvis_patch::GpuIndexFlat *index_ = (faiss::gpu::qvis_patch::GpuIndexFlat *)index; // qvis patch
    HANDLE_ERROR(cudaMallocHost((void **)&indicates, batch_size * k * sizeof(int)));
    for (int batch = 0; batch < points_num; batch += batch_size) {
        int this_batch_size = std::min(points_num, batch + batch_size) - batch;
        index_->search_int_labels(this_batch_size, data + batch * data_stride, k, nullptr, indicates);
        HANDLE_ERROR(cudaDeviceSynchronize()); // FIXME: I don't know why we should wait there, but if not, we may get
                                               // some zero in result

        for (int p = 0; p < this_batch_size; p++) {
            for (int d = 0; d < k; d++) {
                cgraph[d][batch + p] = indicates[p * k + d];
            }
        }
    }

    HANDLE_ERROR(cudaFreeHost(indicates));
}

// @data_stride data stride for vector, like dim
void build_cn_graph_long(const faiss::gpu::GpuIndexFlat *index, const float *data, int data_stride, int points_num,
                         int k, qvis::Graph<unsigned, true> &cgraph) {
    const size_t batch_memory = 1 << 30; // 1GB
    int          batch_size   = batch_memory / sizeof(int) / k;
    long *       indicates;
    float *      distances;
    HANDLE_ERROR(cudaMallocHost((void **)&indicates, batch_size * k * sizeof(long)));
    HANDLE_ERROR(cudaMallocHost((void **)&distances, batch_size * k * sizeof(float)));
    for (int batch = 0; batch < points_num; batch += batch_size) {
        int this_batch_size = std::min(points_num, batch + batch_size) - batch;
        index->search(this_batch_size, data + batch * data_stride, k, distances, indicates);
        HANDLE_ERROR(cudaDeviceSynchronize()); // FIXME: I don't know why we should wait there, but if not, we may get
                                               // some zero in result

        for (int i = 0; i < k; i++) {
            printf("> %ld %e\n", indicates[i], distances[i]);
        }

        for (int p = 0; p < this_batch_size; p++) {
            for (int d = 0; d < k; d++) {
                cgraph[d][batch + p] = indicates[p * k + d];
            }
        }
    }

    HANDLE_ERROR(cudaFreeHost(indicates));
    HANDLE_ERROR(cudaFreeHost(distances));
}

void convert_graph(vector<vector<unsigned>> &graph_v, const qvis::Graph<unsigned, true> &graph) {
    printf("convert graph\n");
    printf("graph_v %lu x %lu\n", graph_v.size(), graph_v[0].size());
    printf("graph %u x %u\n", graph.n(), graph.d());
    graph_v.resize(graph.n());
    for (unsigned i = 0; i < graph.n(); i++) {
        graph_v[i].resize(graph.d());
        for (unsigned j = 0; j < graph.d(); j++) {
            graph_v[i][j] = graph[j][i];
        }
    }
}

void build_evalation_graph(const float *points, unsigned points_num, unsigned dim, int K, int sample,
                           std::vector<unsigned> &sample_indicates, long *graph) {
    sample_indicates.resize(points_num);
    for (unsigned i = 0; i < points_num; i++) {
        sample_indicates[i] = i;
    }
    std::shuffle(sample_indicates.begin(), sample_indicates.end(), std::default_random_engine(0));
    sample_indicates.resize(sample);

    // build index
    faiss::gpu::StandardGpuResources gpuresource;
    faiss::gpu::GpuIndexFlat *       flat_index = new faiss::gpu::GpuIndexFlat(&gpuresource, dim, faiss::METRIC_L2);
    flat_index->add(points_num, points);

    // generate sample data
    float *sample_points;
    HANDLE_ERROR(cudaMallocHost((void **)&sample_points, sample * dim * sizeof(float)));
    for (int i = 0; i < sample; i++) {
        std::copy_n(points + sample_indicates[i] * dim, dim, sample_points + i * dim);
    }

    // search
    float *dummy_distance;
    long * indicates;
    HANDLE_ERROR(cudaMallocHost((void **)&dummy_distance, sample * (K + 1) * sizeof(float)));
    HANDLE_ERROR(cudaMallocHost((void **)&indicates, sample * (K + 1) * sizeof(long)));

    flat_index->search(sample, sample_points, K + 1, dummy_distance, indicates);
    HANDLE_ERROR(cudaDeviceSynchronize());

    for (int i = 0; i < sample; i++) {
        std::copy_n(indicates + i * (K + 1) + 1, K, graph + i * K);
    }

    // clean
    cudaFreeHost(dummy_distance);
    cudaFreeHost(indicates);
    cudaFreeHost(sample_points);
    delete flat_index;
}

double evalue_graph(qvis::Graph<unsigned, true> &point_graph, const std::vector<unsigned> &sample_indicates,
                    const long *ground_truth_graph) {
    const int sample = sample_indicates.size();
    const int K      = point_graph.d();

    vector<unsigned> graph(sample * point_graph.d());
#pragma omp parallel for
    for (int i = 0; i < sample; i++) {
        for (int j = 0; j < K; j++) {
            graph[i * K + j] = point_graph[j][sample_indicates[i]];
        }
        sort(graph.begin() + i * K, graph.begin() + (i + 1) * K);
    }

    vector<unsigned> gt_graph(sample * point_graph.d());
#pragma omp parallel for
    for (int i = 0; i < sample * K; i++) {
        gt_graph[i] = ground_truth_graph[i];
    }
#pragma omp parallel for
    for (int i = 0; i < sample; i++) {
        sort(gt_graph.begin() + i * K, gt_graph.begin() + (i + 1) * K);
    }

    // calc overlap
    vector<int> overlap(sample, 0);
#pragma omp parallel for
    for (int i = 0; i < sample; i++) {
        int ig = 0;
        for (int j = 0; j < K && ig < K; j++) {
            while (ig < K && graph[i * K + ig] < gt_graph[i * K + j]) {
                ig++;
            }
            overlap[i] += graph[i * K + ig] == gt_graph[i * K + j];
        }
    }

    // get result
    double result = 0;
    for (int i = 0; i < sample; i++) {
        result += overlap[i];
    }
    result /= sample * K;
    return result;
}

template <int D, int MaxPointPerBlock>
__global__ void kernel_gen_low_dim(unsigned N, int outdim, float *Y, float *centers,
                                   qvis::Graph<unsigned, true> cn_ggraph, int first_number, float *coeff) {
    unsigned         tid = threadIdx.x;
    unsigned int     idx = blockIdx.x * blockDim.x + tid;
    __shared__ float Y_local[MaxPointPerBlock * D];
    assert(blockDim.x <= MaxPointPerBlock);

    for (; idx < N; idx += gridDim.x * blockDim.x) { // iter point
#pragma unroll
        for (int i = 0; i < D; i++) {
            Y_local[tid * D + i] = 0.0f;
        }

        for (int f = 0; f < first_number; f++) {
#pragma unroll
            for (int d = 0; d < D; d++) {
                Y_local[tid * D + d] += centers[cn_ggraph[f][idx] * D + d];
            }
        }
#pragma unroll
        for (int i = 0; i < D; i++) {
            Y[idx * D + i] += Y_local[tid * D + i];
        }
    }
}

template <int D>
__global__ void kernel_gen_low_dim_simple(unsigned N, int outdim, float *Y, float *centers,
                                          qvis::Graph<unsigned, true> cn_ggraph) {
    unsigned     tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    for (; idx < N; idx += gridDim.x * blockDim.x) { // iter point
#pragma unroll

        for (int d = 0; d < D; d++) {
            Y[idx * D + d] += centers[cn_ggraph[0][idx] * D + d];
        }
    }
}

void gen_low_dim(unsigned N, int outdim, float *Y, int clusters_num, float *Y_centers,
                 qvis::Graph<unsigned, true> cn_graph) {
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_CALL(curandGenerateNormal(gen, Y, N * outdim, 0.0, 0.05));

    // float *coeff = new float[cn_graph.d()];
    const int ThreadPerBlock = 256;
    int       num_block      = min((N + ThreadPerBlock - 1) / ThreadPerBlock, 50);
    assert(outdim == 2);

    float *centers_device;
    HANDLE_ERROR(cudaMalloc((void **)&centers_device, sizeof(float) * clusters_num * outdim));
    HANDLE_ERROR(cudaMemcpy(centers_device, Y_centers, sizeof(float) * clusters_num * outdim, cudaMemcpyHostToDevice));

    kernel_gen_low_dim_simple<2><<<num_block, ThreadPerBlock>>>(N, outdim, Y, centers_device, cn_graph);
    HANDLE_ERROR(cudaDeviceSynchronize());
    cudaFree(centers_device);
}

void gen_low_dim(unsigned N, int outdim, float *Y, float scale) {
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_CALL(curandGenerateNormal(gen, Y, N * outdim, 0.0, scale));
    HANDLE_ERROR(cudaDeviceSynchronize());
}

void normalize_scale(unsigned N, int dim, float *data, float scale = 4.0) {
    std::vector<double> e(dim), es(dim), sd(dim); // E(X) , E(X ^ 2), standard dev(X)
    for (unsigned i = 0; i < N; i++) {
        for (int d = 0; d < dim; d++) {
            e[d] += data[i * dim + d];
            es[d] += data[i * dim + d] * data[i * dim + d];
        }
    }
    for (int d = 0; d < dim; d++) {
        e[d] /= N;
        es[d] /= N;
    }
    for (int d = 0; d < dim; d++) {
        sd[d] = sqrt(es[d] - e[d] * e[d]);
    }
    for (unsigned i = 0; i < N; i++) {
        for (int d = 0; d < dim; d++) {
            data[i * dim + d] = (data[i * dim + d] - e[d]) / sd[d] * scale;
        }
    }
}

float get_neg_sumQ(int negative_smapling_blocks, float *neg_sumQ_blocks) {
    std::vector<float> neg_sumQ_blocks_host(negative_smapling_blocks);
    HANDLE_ERROR(cudaMemcpy(neg_sumQ_blocks_host.data(), neg_sumQ_blocks, sizeof(float) * negative_smapling_blocks,
                            cudaMemcpyDeviceToHost));
    float neg_sumQ = 0;
    for (auto it = neg_sumQ_blocks_host.begin(); it != neg_sumQ_blocks_host.end(); it++) {
        neg_sumQ += *it;
    }
    return neg_sumQ;
}

__global__ void kernel_generate_permuation(unsigned n, unsigned *data) {
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    for (; idx < n; idx += gridDim.x * blockDim.x) { // iter point
        data[idx] = idx;
    }
}
void generate_permutation(unsigned n, unsigned *data) {
    assert(data != nullptr);
    // kernel_generate_permuation <<< 1, 512>>>(n, data);
    // HANDLE_ERROR( cudaDeviceSynchronize() );
    std::vector<unsigned> batchs(n);
    for (unsigned i = 0; i < n; i++) {
        batchs[i] = i;
    }
    auto rng = std::default_random_engine{};
    std::shuffle(batchs.begin(), batchs.end(), rng);
    HANDLE_ERROR(cudaMemcpy(data, batchs.data(), sizeof(unsigned) * n, cudaMemcpyHostToDevice));
}

int main(int argc, char **argv) {
    cmdline::parser parser;
    parser.add<string>("datafile", 'b', "base vectors file path", true);
    parser.add<string>("save_path", 'o', "output filename", true);
    parser.add<string>("labelfile", 'l', "label file path, not required", false, "");
    parser.add<int>("outdim", 'd', "output dimension", false, 2);
    parser.add<float>("lr", '\0', "learning_rate", false, 0.05);
    parser.add<int>("vis_iter", '\0', "number of iterations of gradient dencent", false, 2000);
    parser.add<float>("perplexity", 'p', "perplexity", false, 50);
    parser.add<int>("save_interval", '\0', "interval of iterations to save intermedit result", false, 0);
    parser.add<int>("clusters", '\0', "number of culsters(<1024)", false, 1000);
    parser.add<int>("k", 'k', "k neighbor point", false, 100);
    parser.add<int>("n_negative", 'n', "number of negative sampling point", false, 100);
    parser.add<int>("center_number", '\0', "number of neighbor center", false, 5);
    parser.add<float>("center_grad_coeff", '\0', "learning rate coefficient for center gradient, float", false, 1.0);
    parser.add<int>("center_pull_iter", '\0', "number of iterations for center pull", false, 500);
    parser.add<float>("early_pull_rate", '\0', "early pull rate", false, 20);
    parser.add<int>("early_pull_iter", '\0', "number of iterations for early_pull", false, 1000);
    parser.add<int>("late_pull_iter", '\0', "number of iterations for late_pull", false, 0);
    parser.add<float>("knn_negative_rate", '\0', "coefficient of negative gradient of neighbor point", false, 0.0);
    parser.add<float>("scale", '\0', "standard dev of initialize mapping", false, 10);
    parser.add<float>("center_perplexity", '\0', "perplexity of initialize mapping", false, 70);
    parser.add<int>("nprobe", '\0', "number of probe for IVFIndex search", false, 50);
    parser.add<bool>("ivfpq", '\0', "use ifvpq index", false, false);
    parser.add<int>("subQuantizers", '\0', "subQuantizers of ivfpq index", false, 16);
    parser.add<int>("bitsPerCode", '\0', "bitsPerCode of ivfpq index", false, 8);
    parser.add("evalue_graph", '\0', "evalue accuracy of graph");
    parser.add("no_pre_projection", '\0', "disable pre pre-projection");
    parser.add("sgd", '\0', "use minibatch SGD optimization (experimental and undocumented feature)");
    parser.add("debug_grad", '\0', "output debugging gradient");
    parser.add("use_cache", '\0', "using cached Graph and Weight");
    parser.add("save_cache", '\0', "save cache for Graph and Weight");
    parser.add<int>("verbose", 'v', "verbose level[0,1]", false, 0);
    parser.add<int>("seed", '\0', "random seed", false, 0);

    // parser.add<float>("c2clr", '\0', "center to center learning_rate, 0 for disable", false, 10);

    parser.parse_check(argc, argv);

    int verbose = parser.get<int>("verbose");

    save_args((parser.get<string>("save_path") + ".args").c_str(), argc, argv);

    const int clusters_num = parser.get<int>("clusters");

    if (clusters_num > 1024 || clusters_num < 1) {
        printf("Incorrect cluster num\n");
        return 0;
    }

    qvis::test::time_point last_timepoint = qvis::test::now();

    srand(parser.get<int>("seed"));

    float *  data = NULL;
    unsigned points_num, dim;
    load_data(parser.get<string>("datafile").c_str(), data, points_num, dim);
    unsigned *data_label = nullptr;

    if (!parser.get<string>("labelfile").empty()) {
        data_label = new unsigned[points_num];
        load_label(parser.get<string>("labelfile").c_str(), data_label);
    }

    bool using_sgd = parser.exist("sgd");

    int    outdim   = parser.get<int>("outdim");
    float *vis_data = new float[points_num * outdim];

    unsigned K = parser.get<int>("k");
    ;

    int negative_num = parser.get<int>("n_negative");

    int Cnum = parser.get<int>("center_number");

    float perplexity = parser.get<float>("perplexity");
    if (perplexity == 0.0) {
        perplexity = K;
    }

    float early_pull_rate   = parser.get<float>("early_pull_rate");
    float knn_negative_rate = parser.get<float>("knn_negative_rate");
    float learning_rate     = parser.get<float>("lr");
    // float *Grad = new float[points_num * outdim];
    // float *Grad_old = new float[points_num * outdim]();
    int vis_iter      = parser.get<int>("vis_iter");
    int save_interval = parser.get<int>("save_interval");

    float *Weight = new float[points_num * (K + Cnum)];

    // Build evalue_graph
    long *                evalue_sample_graph;
    std::vector<unsigned> evalue_sample_indicates;
    if (parser.exist("evalue_graph")) {
        int sample = std::max(10000, int(points_num / 100));
        sample     = std::min(int(points_num), sample);
        HANDLE_ERROR(cudaMallocHost((void **)&evalue_sample_graph, sample * K * sizeof(long)));
        build_evalation_graph(data, points_num, dim, K, sample, evalue_sample_indicates, evalue_sample_graph);
    }

    // Build GpuIndexIVFPQ
    faiss::gpu::StandardGpuResources gpuresource;
    faiss::IndexIVF *                cpu_ivf_index = nullptr;
    faiss::gpu::GpuIndexIVF *        ivf_index     = nullptr;
    // GpuIndexIVFPQ (GpuResources *resources, int dims, int nlist, int subQuantizers, int bitsPerCode,
    // faiss::MetricType metric, GpuIndexIVFPQConfig config=GpuIndexIVFPQConfig())
    if (parser.get<bool>("ivfpq")) {
        int subQuantizers = parser.get<int>("subQuantizers");
        int bitsPerCode   = parser.get<int>("bitsPerCode");
        ivf_index         = new faiss::gpu::GpuIndexIVFPQ(&gpuresource, dim, clusters_num, subQuantizers, bitsPerCode,
                                                  faiss::METRIC_L2);
        cpu_ivf_index     = new faiss::IndexIVFPQ();
    } else {
        ivf_index     = new faiss::gpu::GpuIndexIVFFlat(&gpuresource, dim, clusters_num, faiss::METRIC_L2);
        cpu_ivf_index = new faiss::IndexIVFFlat();
    }
    ivf_index->verbose     = true;
    cpu_ivf_index->verbose = true;

    FILE * coarse_quantizer_cache_file      = nullptr;
    string coarse_quantizer_cache_file_path = parser.get<string>("datafile") + ".quantizer.cache";

    if (parser.exist("use_cache")) {
        coarse_quantizer_cache_file = fopen(coarse_quantizer_cache_file_path.c_str(), "r");
    }
    if (coarse_quantizer_cache_file) {
        delete cpu_ivf_index;
        cpu_ivf_index = (faiss::IndexIVF *)faiss::read_index(coarse_quantizer_cache_file);
        printf("coarse_quantizer_cache loaded.\n");
        fclose(coarse_quantizer_cache_file);
        if (parser.get<bool>("ivfpq")) {
            ((faiss::gpu::GpuIndexIVFPQ *)ivf_index)->copyFrom((faiss::IndexIVFPQ *)cpu_ivf_index);
        } else {
            ((faiss::gpu::GpuIndexIVFFlat *)ivf_index)->copyFrom((faiss::IndexIVFFlat *)cpu_ivf_index);
        }
    } else {
        printf("train ivf_index\n");
        ivf_index->train(points_num, data);
        printf("ivf_index trained\n");
        if (parser.get<bool>("ivfpq")) {
            ((faiss::gpu::GpuIndexIVFPQ *)ivf_index)->copyTo((faiss::IndexIVFPQ *)cpu_ivf_index);
        } else {
            ((faiss::gpu::GpuIndexIVFFlat *)ivf_index)->copyTo((faiss::IndexIVFFlat *)cpu_ivf_index);
        }
        if (parser.exist("save_cache")) {
            faiss::write_index(cpu_ivf_index, coarse_quantizer_cache_file_path.c_str());
            printf("coarse_quantizer_cache saved.\n");
        }
    }
    delete cpu_ivf_index;
    cpu_ivf_index = nullptr;

    // add data to index
    ivf_index->add(points_num, data);
    printf("data added to index\n");

    const int aligned_points_num = (points_num + (256 / 4) - 1) / (256 / 4) * (256 / 4); // aligned to 256 bytes
    qvis::Graph<unsigned, true> nn_ggraph(K, points_num, aligned_points_num * sizeof(unsigned), nullptr);
    qvis::Graph<unsigned, true> cn_ggraph(Cnum, points_num, aligned_points_num * sizeof(unsigned), nullptr);
    qvis::MatrixPitched<float>  W_point(K, points_num, aligned_points_num * sizeof(float), nullptr);
    qvis::MatrixPitched<float>  W_center(Cnum, points_num, aligned_points_num * sizeof(float), nullptr);
    HANDLE_ERROR(cudaMallocManaged((void **)&nn_ggraph.data(), K * aligned_points_num * sizeof(unsigned)));
    HANDLE_ERROR(cudaMallocManaged((void **)&cn_ggraph.data(), Cnum * aligned_points_num * sizeof(unsigned)));
    HANDLE_ERROR(cudaMallocManaged((void **)&W_point.data(), K * aligned_points_num * sizeof(float)));
    HANDLE_ERROR(cudaMallocManaged((void **)&W_center.data(), Cnum * aligned_points_num * sizeof(float)));
    ivf_index->setNumProbes(parser.get<int>("nprobe"));

    float *centers = new float[clusters_num * dim];

    cudaStream_t              main_stream      = gpuresource.getDefaultStream(ivf_index->getDevice());
    faiss::gpu::GpuIndexFlat *coarse_quantizer = ivf_index->getQuantizer();

    // get kmeans centers from ivf_index
    faiss::gpu::fromDevice<float, 2>(coarse_quantizer->getGpuData()->getVectorsFloat32Ref(), centers, main_stream);
    HANDLE_ERROR(cudaDeviceSynchronize());

    float center_perplexity = parser.get<float>("center_perplexity");
    if (center_perplexity == 0.0) {
        center_perplexity = perplexity + 10;
    }
    qvis::tsne::tSNE<2> tsne_grad(clusters_num, dim, centers);
    tsne_grad.calc_W(dim, center_perplexity);

    // pre_projection
    float *centers_tsne_result = nullptr;
    if (!parser.exist("no_pre_projection")) {
        // Give lables

        unsigned *center_label = nullptr;

        if (data_label != nullptr) { // if data_label is provided,
                                     // save pre_projection lables based on centers' neighbor neighbor
            center_label            = new unsigned[clusters_num];
            float *center_distances = new float[clusters_num];
            long * center_indicates = new long[clusters_num];
            ivf_index->search(clusters_num, centers, 1, center_distances, center_indicates);
            for (int i = 0; i < clusters_num; i++) {
                center_label[i] = data_label[center_indicates[i]];
            }
            delete[] center_distances;
            delete[] center_indicates;
        }

        centers_tsne_result                             = new float[clusters_num * outdim];
        std::function<void(int, float *)> save_function = [&](int iter, float *data) -> void {
            if (iter % 500 == 0) {
                printf("pre_projection iter %d\n", iter);
            }
        };
        tsne_grad.tsne(centers_tsne_result, learning_rate, vis_iter, 0.5, save_function);

        // qvis::tsne::tsne<2>(clusters_num, dim, centers, centers_tsne_result, perplexity, learning_rate, vis_iter,
        // 0.1, save_function);

        save_result((parser.get<string>("save_path") + ".coarse").c_str(), clusters_num, outdim, centers_tsne_result,
                    center_label);
        delete[] center_label;
        tsne_grad.free_intermediate();
        normalize_scale(clusters_num, outdim, centers_tsne_result, parser.get<float>("scale"));
    }

    printf("calc_gauss_perplexity\n");

    string W_point_cache_path   = parser.get<string>("datafile") + ".W_point.cache";
    string W_center_cache_path  = parser.get<string>("datafile") + ".W_center.cache";
    string nn_ggraph_cache_path = parser.get<string>("datafile") + ".nn_ggraph.cache";
    string cn_ggraph_cache_path = parser.get<string>("datafile") + ".cn_ggraph.cache";
    if (parser.exist("use_cache")) {
        W_point.load_data_cpu(W_point_cache_path.c_str());
        W_center.load_data_cpu(W_center_cache_path.c_str());
        nn_ggraph.load_data_cpu(nn_ggraph_cache_path.c_str());
        cn_ggraph.load_data_cpu(cn_ggraph_cache_path.c_str());
    } else {
        last_timepoint = qvis::test::now();
        calc_gauss_perplexity_and_build_graph_gpu(data, points_num, dim, centers, ivf_index, coarse_quantizer,
                                                  perplexity,
                                                  Cnum, // number of neighborhood center
                                                  K,    // number of neighborhood points
                                                  W_point, W_center, nn_ggraph, cn_ggraph);
        printf("%s %7.4lf ms\n", "calc_gauss_perplexity",
               qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));
    }
    if (parser.exist("save_cache")) {
        W_point.save_data_cpu(W_point_cache_path.c_str());
        W_center.save_data_cpu(W_center_cache_path.c_str());
        nn_ggraph.save_data_cpu(nn_ggraph_cache_path.c_str());
        cn_ggraph.save_data_cpu(cn_ggraph_cache_path.c_str());
    }

    if (parser.exist("evalue_graph")) {
        // evalue
        printf("evalue %d simple point\n", int(evalue_sample_indicates.size()));
        last_timepoint        = qvis::test::now();
        double graph_accuracy = evalue_graph(nn_ggraph, evalue_sample_indicates, evalue_sample_graph);
        printf("%s %7.4lf ms\n", "evalue_graph", qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));
        printf("graph accuracy = %f\n", graph_accuracy);
        // clear
        cudaFreeHost(evalue_sample_graph);
        evalue_sample_indicates.clear();
        evalue_sample_indicates.shrink_to_fit();
    }

    // momentum
    float *gains = new float[points_num * outdim];
    for (unsigned i = 0; i < points_num * outdim; i++) {
        gains[i] = 1;
    }

    // initialize GPU grad
    const int ThreadPerBlock           = 256;
    int       num_block                = min((points_num + ThreadPerBlock - 1) / ThreadPerBlock, 5280);
    int       negative_smapling_blocks = num_block;

    constexpr int               D = 2;
    qvis::GradientCalculater<D> gradC;
    gradC.N  = points_num;
    gradC.NC = clusters_num;

    gradC.set_graph(nn_ggraph, cn_ggraph);
    gradC.set_weight(W_point, W_center);

    HANDLE_ERROR(cudaMalloc((void **)&gradC.Y, sizeof(float) * points_num * D));
    HANDLE_ERROR(cudaMalloc((void **)&gradC.grad, sizeof(float) * points_num * D));
    HANDLE_ERROR(cudaMalloc((void **)&gradC.grad_old, sizeof(float) * points_num * D));
    HANDLE_ERROR(cudaMalloc((void **)&gradC.grad_neg, sizeof(float) * points_num * D));
    HANDLE_ERROR(cudaMalloc((void **)&gradC.gain, sizeof(float) * points_num * D));
    HANDLE_ERROR(cudaMalloc((void **)&gradC.neg_sumQ_blocks, sizeof(float) * negative_smapling_blocks));
    HANDLE_ERROR(cudaMalloc((void **)&gradC.cluster_centers, sizeof(float) * gradC.NC * D));
    HANDLE_ERROR(cudaMalloc((void **)&gradC.cluster, sizeof(int) * points_num));

    HANDLE_ERROR(cudaMemset(gradC.grad_old, 0, sizeof(float) * points_num * D));

    // search point belong to which cluster
    last_timepoint = qvis::test::now();
    ((faiss::gpu::qvis_patch::GpuIndexFlat *)coarse_quantizer)
        ->search_int_labels(points_num, data, 1, nullptr, (int *)gradC.cluster);
    HANDLE_ERROR(cudaDeviceSynchronize());

    printf("%s %7.4lf ms\n", "searching for cluster centers",
           qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));

    // initialize center to center graditent

    tsne_grad.Y = gradC.cluster_centers;
    tsne_grad.clear_grad();

    // data is nolonger available from there
    delete[] data;
    data = nullptr;

    last_timepoint = qvis::test::now();
    gradC.calc_cluster_cap();
    printf("%s %7.4lf ms\n", "calc_cluster_cap", qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));

    gradC.init_gain();
    HANDLE_ERROR(cudaDeviceSynchronize());

    // negative sampling init
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)parser.get<int>("seed")));
    gradC.neg_graph.allocate_memory_managed(points_num, negative_num);

    if (!parser.exist("no_pre_projection")) {
        printf("do pre_projection\n");
        gen_low_dim(points_num, outdim, gradC.Y, clusters_num, centers_tsne_result, gradC.cgraph);
    } else {
        gen_low_dim(points_num, outdim, gradC.Y, parser.get<float>("scale"));
    }

    int early_pull_iter  = parser.get<int>("early_pull_iter");
    int center_pull_iter = parser.get<int>("center_pull_iter");
    int late_pull_iter   = parser.get<int>("late_pull_iter");

    // init moving_neg_sumQ
    float    moving_neg_sumQ = 0;
    unsigned sgd_batchs      = 0;
    if (using_sgd) {
        // malloc sgd_mapping
        sgd_batchs = (points_num + 31) / 32;
        HANDLE_ERROR(cudaMalloc((void **)&gradC.sgd_mapping, sizeof(unsigned) * sgd_batchs));

        CURAND_CALL(curandGenerate(gen, gradC.neg_graph.data_,
                                   gradC.neg_graph.d() * gradC.neg_graph.pitch / sizeof(unsigned)));
        // printf("%s %7.4lf ms\n", "generate neg graph", qvis::test::getmilliseconds(iter_start, qvis::test::now()));
        qvis::kernel_mod<unsigned><<<num_block, 128>>>(gradC.neg_graph.data_, points_num,
                                                       gradC.neg_graph.d() * gradC.neg_graph.pitch / sizeof(unsigned));
        HANDLE_ERROR(cudaDeviceSynchronize());

        qvis::kernel_calc_negative_gradient<D, unsigned, true>
            <<<num_block, ThreadPerBlock>>>(gradC.Y, gradC.graph, 1.0, gradC.grad_neg, gradC.neg_sumQ_blocks);
        HANDLE_ERROR(cudaDeviceSynchronize());
        moving_neg_sumQ = get_neg_sumQ(negative_smapling_blocks, gradC.neg_sumQ_blocks);
    }

    for (int iter = 0; iter < vis_iter; iter++) {
        HANDLE_ERROR(cudaDeviceSynchronize());

        if (save_interval != 0 && iter % save_interval == 0) {
            HANDLE_ERROR(cudaMemcpy(vis_data, gradC.Y, sizeof(float) * points_num * D, cudaMemcpyDeviceToHost));
            char num_buffer[10];
            sprintf(num_buffer, "%08d", iter);
            string save_intermedit_path = parser.get<string>("save_path") + "." + num_buffer;
            save_result(save_intermedit_path.c_str(), points_num, outdim, vis_data, data_label);
        }

        int                    early_pull_flag = iter > early_pull_iter ? 0 : 1;
        qvis::test::time_point iter_start      = qvis::test::now();

        const int ClusterPerBlock = 32;

        // Grad for neighbor points
        float coeff    = early_pull_flag ? early_pull_rate : 1;
        float p2pcoeff = coeff;
        if (vis_iter - iter < late_pull_iter) {
            p2pcoeff *= 2;
        }

        // Grad for neighbor center
        float center_grad_coeff = 0.0;
        if (center_pull_iter == 0 || iter < center_pull_iter) {
            // Centers
            int num_block_centering = (gradC.NC + ClusterPerBlock - 1) / ClusterPerBlock;
            if (num_block_centering > 10) {
                num_block_centering = 10;
            }
            qvis::kernel_calc_centers<D, ClusterPerBlock, ThreadPerBlock><<<num_block_centering, ThreadPerBlock>>>(
                gradC.N, gradC.NC, gradC.Y, gradC.cluster_cap, gradC.cluster, gradC.cluster_centers);
            HANDLE_ERROR(cudaDeviceSynchronize());
            if (verbose >= 2) {
                printf("%s %7.4lf ms\n", "kernel_calc_centers",
                       qvis::test::getmilliseconds(iter_start, qvis::test::now()));
            }

            center_grad_coeff = coeff * parser.get<float>("center_grad_coeff");
            // run tsne on centers
            tsne_grad.do_iter(learning_rate, 0.8);
            HANDLE_ERROR(cudaDeviceSynchronize());
            if (!using_sgd) {
                qvis::kernel_calc_gradient<D, unsigned, true><<<num_block, ThreadPerBlock>>>(
                    gradC.Y, gradC.cluster_centers, gradC.cgraph, gradC.cweight, center_grad_coeff, gradC.grad);
                HANDLE_ERROR(cudaDeviceSynchronize());
                if (verbose >= 2) {
                    printf("%s %7.4lf ms\n", "kernel_calc_gradient NC",
                           qvis::test::getmilliseconds(iter_start, qvis::test::now()));
                }
            }
        }

        // use CURAND to generate random graph
        CURAND_CALL(curandGenerate(gen, gradC.neg_graph.data_,
                                   gradC.neg_graph.d() * gradC.neg_graph.pitch / sizeof(unsigned)));
        if (verbose >= 2) {
            HANDLE_ERROR(cudaDeviceSynchronize());
            printf("%s %7.4lf ms\n", "generate neg graph", qvis::test::getmilliseconds(iter_start, qvis::test::now()));
        }
        qvis::kernel_mod<unsigned><<<num_block, 128>>>(gradC.neg_graph.data_, points_num,
                                                       gradC.neg_graph.d() * gradC.neg_graph.pitch / sizeof(unsigned));
        HANDLE_ERROR(cudaDeviceSynchronize());
        if (verbose >= 2) {
            printf("%s %7.4lf ms\n", "kernel_mod", qvis::test::getmilliseconds(iter_start, qvis::test::now()));
        }

        // clear neg_sumQ_blocks
        HANDLE_ERROR(cudaMemset(gradC.neg_sumQ_blocks, 0, sizeof(float) * negative_smapling_blocks)); // Clear flag

        float neg_sumQ = 0;
        if (using_sgd) { // DO SGD
            // generate random mapping
            generate_permutation(sgd_batchs, gradC.sgd_mapping);
            printf("sumQ = %f, learning_rate = %f\n", moving_neg_sumQ, learning_rate * (1 - 0.9 * iter / vis_iter));
            // call SGD keanel
            if (parser.exist("debug_grad")) {
                HANDLE_ERROR(cudaMemset(gradC.grad, 0, sizeof(float) * points_num * D));     // Clear flag
                HANDLE_ERROR(cudaMemset(gradC.grad_neg, 0, sizeof(float) * points_num * D)); // Clear flag

                qvis::kernel_update_sgd<D, unsigned, true><<<num_block, ThreadPerBlock>>>(
                    gradC.Y,               // low dimension points,
                    gradC.sgd_mapping,     // random index of batchs
                    gradC.cluster_centers, // centers
                    gradC.graph,           // graph between Y and Y, degree first
                    gradC.weight,          // weight(p_ij - q_ij) between Y and Y, same dimension as NG, degree first
                    p2pcoeff,              // coefficient of neighbor positive
                    knn_negative_rate,     // coefficient of neighbor negative
                    gradC.cgraph,          // graph between Y and center, degree first
                    center_grad_coeff,     // coefficient of center positive
                    gradC.cweight,   // weight(p_ij - q_ij) between Y and center, same dimension as NG, degree first
                    gradC.neg_graph, // negitive graph between Y and Y, degree first
                    1.0,             // coefficient of negative sampling negative
                    moving_neg_sumQ / points_num,                // sum[(1 + (x - y)^2)^-1]
                    learning_rate * (1 - 0.9 * iter / vis_iter), // learning rate
                    gradC.neg_sumQ_blocks,                       // SumQ of block
                    gradC.grad, gradC.grad_neg);
            } else {
                qvis::kernel_update_sgd<D, unsigned><<<num_block, ThreadPerBlock>>>(
                    gradC.Y,               // low dimension points,
                    gradC.sgd_mapping,     // random index of batchs
                    gradC.cluster_centers, // centers
                    gradC.graph,           // graph between Y and Y, degree first
                    gradC.weight,          // weight(p_ij - q_ij) between Y and Y, same dimension as NG, degree first
                    p2pcoeff,              // coefficient of neighbor positive
                    knn_negative_rate,     // coefficient of neighbor negative
                    gradC.cgraph,          // graph between Y and center, degree first
                    center_grad_coeff,     // coefficient of center positive
                    gradC.cweight,   // weight(p_ij - q_ij) between Y and center, same dimension as NG, degree first
                    gradC.neg_graph, // negitive graph between Y and Y, degree first
                    1.0,             // coefficient of negative sampling negative
                    moving_neg_sumQ / points_num,                // sum[(1 + (x - y)^2)^-1]
                    learning_rate * (1 - 0.9 * iter / vis_iter), // learning rate
                    gradC.neg_sumQ_blocks,                       // SumQ of block
                    nullptr, nullptr);
            }

            HANDLE_ERROR(cudaDeviceSynchronize());
            neg_sumQ += get_neg_sumQ(negative_smapling_blocks, gradC.neg_sumQ_blocks);
        } else { // full gradient decent

            // Run negative sampling
            // Cause we use CUDA unified memory, negative sampling should be near graph generating
            HANDLE_ERROR(cudaMemset(gradC.grad_neg, 0, sizeof(float) * points_num * D)); // Clear flag

            // negative gradient of random sampling
            qvis::kernel_calc_negative_gradient<D, unsigned, false>
                <<<num_block, ThreadPerBlock>>>(gradC.Y, gradC.neg_graph, 1.0, gradC.grad_neg, gradC.neg_sumQ_blocks);
            HANDLE_ERROR(cudaDeviceSynchronize());
            if (verbose >= 2) {
                printf("%s %7.4lf ms\n", "kernel_calc_negative_gradient",
                       qvis::test::getmilliseconds(iter_start, qvis::test::now()));
            }
            neg_sumQ += get_neg_sumQ(
                negative_smapling_blocks,
                gradC.neg_sumQ_blocks); // cause neg_sumQ_blocks is accumulated, we do not need to calc it twice
            // negative gradient of nn-graph
            if (knn_negative_rate > 0) {
                qvis::kernel_calc_negative_gradient<D, unsigned, true><<<num_block, ThreadPerBlock>>>(
                    gradC.Y, gradC.graph, knn_negative_rate, gradC.grad_neg, gradC.neg_sumQ_blocks);
                HANDLE_ERROR(cudaDeviceSynchronize());
                neg_sumQ += get_neg_sumQ(
                    negative_smapling_blocks,
                    gradC.neg_sumQ_blocks); // cause neg_sumQ_blocks is accumulated, we do not need to calc it twice
            }

            HANDLE_ERROR(cudaMemset(gradC.grad, 0, sizeof(float) * points_num * D)); // Clear flag

            qvis::kernel_calc_gradient<D, unsigned, true>
                <<<num_block, ThreadPerBlock>>>(gradC.Y, gradC.Y, gradC.graph, gradC.weight, p2pcoeff, gradC.grad);
            HANDLE_ERROR(cudaDeviceSynchronize());
            if (verbose >= 2) {
                printf("%s %7.4lf ms\n", "kernel_calc_gradient NN",
                       qvis::test::getmilliseconds(iter_start, qvis::test::now()));
            }
        }

        if (using_sgd) {
            moving_neg_sumQ = 0.9 * moving_neg_sumQ + 0.1 * neg_sumQ;
        } else {
            if ((negative_num == 0 && knn_negative_rate == 0) || neg_sumQ == 0) {
                neg_sumQ = 1.0;
            }
            // update Y
            float momentum = iter > early_pull_iter ? 0.8 : 0.5;
            qvis::kernel_update_Y<D><<<num_block, ThreadPerBlock>>>(points_num, gradC.Y, gradC.grad, gradC.grad_old,
                                                                    gradC.grad_neg, gradC.gain, neg_sumQ / points_num,
                                                                    momentum, learning_rate);
            HANDLE_ERROR(cudaDeviceSynchronize());
        }
        if (save_interval != 0 && iter % save_interval == 0 && parser.exist("debug_grad")) {
            char num_buffer[10];
            sprintf(num_buffer, "%08d", iter);
            string save_intermedit_grad_path = parser.get<string>("save_path") + "." + num_buffer + ".grad";
            save_gradient(save_intermedit_grad_path.c_str(), points_num, outdim, gradC.grad, gradC.grad_neg);
        }
        if (verbose >= 2) {
            printf("%s %7.4lf ms\n", "kernel_update_Y", qvis::test::getmilliseconds(iter_start, qvis::test::now()));
        }

        printf("finish iter %4d, %7.4lf\n", iter, qvis::test::getmilliseconds(iter_start, qvis::test::now()));
    }

    // CopyBack
    HANDLE_ERROR(cudaMemcpy(vis_data, gradC.Y, sizeof(float) * points_num * D, cudaMemcpyDeviceToHost));

    save_result(parser.get<string>("save_path").c_str(), points_num, outdim, vis_data, data_label);
    save_data((parser.get<string>("save_path") + ".fvecs").c_str(), vis_data, points_num, outdim);

    return 0;
}
