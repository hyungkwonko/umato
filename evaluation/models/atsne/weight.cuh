#pragma once
#include <iostream>

#include "vendor/faiss/gpu/GpuIndexFlat.h"
#include "vendor/faiss/gpu/GpuIndexIVF.h"

#include "testing.hpp"

const float float_max = std::numeric_limits<float>::max();

__global__ void kernel_calc_W(int N, int K, float perplexity, const float *distances,
                              int    distances_align,      // K * distance_pitch
                              float *W_output, int W_align // K * distance_pitch
) {
    const float  logp = log(perplexity);
    unsigned     tid  = threadIdx.x;
    unsigned int idx  = blockIdx.x * blockDim.x + tid;

    for (; idx < N; idx += gridDim.x * blockDim.x) { // iter point
        float beta = 1e-5;
        float minf = -1;
        float maxf = 1;
        float sumf;
        for (int iter = 0; iter < 200; iter++) {
            float H = 0;
            sumf    = 0;
            for (int i = 0; i < K; i++) {
                float d = distances[i * distances_align + idx];
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
        for (int i = 0; i < K; i++) {
            W_output[i * W_align + idx] = exp(-beta * distances[i * distances_align + idx]) / sumf;
        }
    }
}
// @param X: data
// @param Centers:
// @param C: number of centers
// @param K: number of neighbors
void calc_gauss_perplexity_and_build_graph_gpu(const float *points, unsigned points_num, unsigned dim,
                                               const float *Centers, const faiss::gpu::GpuIndexIVF *point_index,
                                               const faiss::gpu::GpuIndexFlat *center_index, float perplexity,
                                               int                          K_center, // number of neighborhood center
                                               int                          K_point,  // number of neighborhood points
                                               qvis::MatrixPitched<float> & W_point,
                                               qvis::MatrixPitched<float> & W_center,
                                               qvis::Graph<unsigned, true> &point_graph,
                                               qvis::Graph<unsigned, true> &center_graph,
                                               size_t                       memory_limit = 2lu << 30) {

    if (K_center == 0 && K_point == 0) {
        return;
    }
    qvis::test::time_point last_timepoint = qvis::test::now();

    // allocate memory for one batch of points
    // GPU memory consumption:
    // (K_point + K_center) * batch_size distances
    // (K_point + K_center) * batch_size W
    const size_t batch_line_memory = (K_point + K_center) * 2 * sizeof(float);

    // so pitch should be  batch_size * sizeof(float)
    unsigned batch_size = (memory_limit / batch_line_memory) & (~((1 << 6) - 1));
    if (batch_size > points_num) {
        batch_size = ((points_num + 63) / 64) * 64;
    }
    printf("calc_gauss_perplexity_and_build_graph_gpu batch_size = %d\n", batch_size);

    // following 6 array is batch sized
    float *point_distances;
    long * point_indicates; // FIXME: we perfer int

    float *center_distances;
    int *  center_indicates;

    float *distances_host, *distances_device;
    float *W_device;

    if (K_point > 0) {

        HANDLE_ERROR(cudaMallocHost((void **)&point_indicates, batch_size * (K_point + 1) * sizeof(long)));
        HANDLE_ERROR(cudaMallocHost((void **)&point_distances, batch_size * (K_point + 1) * sizeof(float)));
    }

    if (K_center > 0) {
        HANDLE_ERROR(cudaMallocHost((void **)&center_indicates, batch_size * K_center * sizeof(int)));
        HANDLE_ERROR(cudaMallocHost((void **)&center_distances, batch_size * K_center * sizeof(float)));
    }

    HANDLE_ERROR(cudaMallocHost((void **)&distances_host, batch_size * (K_point + K_center) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&distances_device, batch_size * (K_point + K_center) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&W_device, batch_size * (K_point + K_center) * sizeof(float)));

    // use our patch
    const faiss::gpu::qvis_patch::GpuIndexFlat *center_index_ =
        (faiss::gpu::qvis_patch::GpuIndexFlat *)center_index; // qvis patch

    for (unsigned batch = 0; batch < points_num; batch += batch_size) { // loop for batch
        int this_batch_size = std::min(points_num, batch + batch_size) - batch;

        last_timepoint = qvis::test::now();

        // build point graph
        if (K_point > 0) {
            point_index->search(this_batch_size, points + batch * dim, K_point + 1, point_distances, point_indicates);
            HANDLE_ERROR(
                cudaDeviceSynchronize()); // FIXME: 2018-4-20, if we don't sync here, next search call may crash
        }
        printf("%s %7.4lf ms\n", "point_index->search",
               qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));

        // build center graph
        if (K_center > 0) {
            center_index_->search_int_labels(this_batch_size, points + batch * dim, K_center, center_distances,
                                             center_indicates);
        }
        HANDLE_ERROR(cudaDeviceSynchronize()); // FIXME: I don't know why we should wait there, but if not, we may get
                                               // some zero in result
        printf("%s %7.4lf ms\n", "center_index_->search_int_labels",
               qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));

// for each point in batch
#pragma omp parallel for
        for (int p = 0; p < this_batch_size; p++) { // suffix '_p' means point
            float *point_distances_p  = point_distances + p * (K_point + 1);
            long * point_indicates_p  = point_indicates + p * (K_point + 1);
            float *center_distances_p = center_distances + p * K_center;
            int *  center_indicates_p = center_indicates + p * K_center;

            // remove the same point from point_graph
            {
                int offset = 0;
                for (int i = 0; i < K_point; i++) {
                    if (point_indicates_p[offset] == batch + p) {
                        offset++; // skip the same point
                    }
                    point_distances_p[i] = point_distances_p[offset];
                    point_indicates_p[i] = point_indicates_p[offset];
                    offset++;
                }
            }

            // fill distances
            for (int i = 0; i < K_point; i++) {
                distances_host[i * batch_size + p] = point_distances_p[i];
            }
            for (int i = 0; i < K_center; i++) {
                distances_host[(i + K_point) * batch_size + p] = center_distances_p[i];
            }

            // output graph
            for (int i = 0; i < K_point; i++) {
                point_graph[i][batch + p] = point_indicates_p[i];
            }
            for (int i = 0; i < K_center; i++) {
                center_graph[i][batch + p] = center_indicates_p[i];
            }

        } // end loop point in batch

        printf("%s %7.4lf ms\n", "finish dedupe", qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));
        // run binary search kernel
        HANDLE_ERROR(cudaMemcpy(distances_device, distances_host, sizeof(float) * batch_size * (K_point + K_center),
                                cudaMemcpyHostToDevice));

        const int ThreadPerBlock = 128;
        int       num_block      = min((batch_size + ThreadPerBlock - 1) / ThreadPerBlock, 50);
        last_timepoint           = qvis::test::now();
        kernel_calc_W<<<num_block, ThreadPerBlock>>>(this_batch_size, K_point + K_center,
                                                     perplexity, // Note, N = this_batch_size
                                                     distances_device, batch_size, W_device, batch_size);
        HANDLE_ERROR(cudaDeviceSynchronize());
        printf("%s %7.4lf ms\n", "kernel_calc_W", qvis::test::getmilliseconds(last_timepoint, qvis::test::now()));
        // Copy back W
        for (int i = 0; i < K_point; i++) {
            HANDLE_ERROR(cudaMemcpy(W_point.data() + i * W_point.pitch / sizeof(float) + batch,
                                    W_device + i * batch_size, sizeof(float) * this_batch_size,
                                    cudaMemcpyDeviceToHost));
        }
        for (int i = 0; i < K_center; i++) {
            HANDLE_ERROR(cudaMemcpy(W_center.data() + i * W_center.pitch / sizeof(float) + batch,
                                    W_device + (i + K_point) * batch_size, sizeof(float) * this_batch_size,
                                    cudaMemcpyDeviceToHost));
        }
    } // end loop of batch

    printf("finish perplexity build\n");

    // clear buffer for batch
    if (K_point > 0) {
        HANDLE_ERROR(cudaFreeHost(point_indicates));
        HANDLE_ERROR(cudaFreeHost(point_distances));
    }
    if (K_center > 0) {
        HANDLE_ERROR(cudaFreeHost(center_indicates));
        HANDLE_ERROR(cudaFreeHost(center_distances));
    }

    HANDLE_ERROR(cudaFreeHost(distances_host));

    HANDLE_ERROR(cudaFree(distances_device));
    HANDLE_ERROR(cudaFree(W_device));

    // calc sumf
    float *sumf_degree = new float[K_point + K_center]; // sumf for per degree

#pragma omp parallel for
    for (int i = 0; i < K_point + K_center; i++) {
        sumf_degree[i] = 0.0;
        if (i < K_point) { // point
            for (unsigned n = 0; n < points_num; n++) {
                sumf_degree[i] += W_point[i][n];
            }
        } else { // center
            for (unsigned n = 0; n < points_num; n++) {
                sumf_degree[i] += W_center[i - K_point][n];
            }
        }
    }

    float sumf = 0;
    for (int i = 0; i < K_point + K_center; i++) {
        sumf += sumf_degree[i];
        // printf("sumf_degree[%d] = %f\n", i, sumf_degree[i]);
    }
    delete[] sumf_degree;

    printf("sumf = %f\n", sumf);

    // symmetric
    // FIXME: scan neighbor of point in degree first graph is extreme cache unfriendly, maybe transpose is need

    printf("building symmetric weight\n");
#pragma omp parallel for
    for (unsigned n = 0; n < points_num; n++) { // loop for points
        for (int i = 0; i < K_point; i++) {     // loop for neighbors
            unsigned id = point_graph[i][n];
            if (id == 0xffffffff) { // not enough point in IVF, we will get this
                break;
            }
            int found_id = -1;
            for (int j = 0; j < K_point; j++) {
                if (point_graph[j][id] == n) {
                    found_id = j;
                    break;
                }
            }
            if (found_id > 0) {
                if (id > n) {
                    float avg             = (W_point[i][n] + W_point[found_id][id]) / 2;
                    W_point[i][n]         = avg;
                    W_point[found_id][id] = avg;
                }
            } else {
                W_point[i][n] /= 2;
            }
        }
        for (int i = 0; i < K_center; i++) {
            W_center[i][n] /= 2;
        }
    }
    printf("finish symmetric weight build\n");

    // divide each weight by the sum of weights

    // #pragma omp parallel for
    // for (int i = 0; i < K_point + K_center; i++) {
    //   if (i < K_point) { // point
    //     for (unsigned n = 0; n < points_num; n++) {
    //       W_point[i][n] /= sumf;
    //     }
    //   } else { // center
    //     for (unsigned n = 0; n < points_num; n++) {
    //       W_center[i - K_point][n] /= sumf;
    //     }
    //   }
    // }
}

// @param X: data
// @param Centers:
// @param C: number of centers
// @param K: number of neighbors
void calc_gauss_perplexity_and_build_graph(const float *points, unsigned points_num, unsigned dim,
                                           const float *Centers, const faiss::gpu::GpuIndexIVF *point_index,
                                           const faiss::gpu::GpuIndexFlat *center_index, float perplexity,
                                           int                         K_center, // number of neighborhood center
                                           int                         K_point,  // number of neighborhood points
                                           qvis::MatrixPitched<float> &W_point, qvis::MatrixPitched<float> &W_center,
                                           qvis::Graph<unsigned, true> &point_graph,
                                           qvis::Graph<unsigned, true> &center_graph, size_t memory_limit = 1 << 30) {

    if (K_center == 0 && K_point == 0) {
        return;
    }

    // allocate memory for one batch of points
    const size_t batch_line_memory =
        std::max((K_point + 1) * (sizeof(float) + sizeof(long)), K_center * (sizeof(float) + sizeof(int)));

    int batch_size = memory_limit / batch_line_memory;

    // following 6 array is batch sized
    float *point_distances;
    long * point_indicates; // FIXME: we perfer int

    float *center_distances;
    int *  center_indicates;

    float *point_w  = new float[batch_size * K_point];
    float *center_w = new float[batch_size * K_center];

    HANDLE_ERROR(cudaMallocHost((void **)&point_indicates, batch_size * (K_point + 1) * sizeof(long)));
    HANDLE_ERROR(cudaMallocHost((void **)&point_distances, batch_size * (K_point + 1) * sizeof(float)));

    HANDLE_ERROR(cudaMallocHost((void **)&center_indicates, batch_size * K_center * sizeof(int)));
    HANDLE_ERROR(cudaMallocHost((void **)&center_distances, batch_size * K_center * sizeof(float)));

    // use our patch
    const faiss::gpu::qvis_patch::GpuIndexFlat *center_index_ =
        (faiss::gpu::qvis_patch::GpuIndexFlat *)center_index; // qvis patch

    for (unsigned batch = 0; batch < points_num; batch += batch_size) { // loop for batch
        int this_batch_size = std::min(points_num, batch + batch_size) - batch;

        // build point graph
        if (K_point > 0) {
            point_index->search(this_batch_size, points + batch * dim, K_point + 1, point_distances, point_indicates);
            HANDLE_ERROR(
                cudaDeviceSynchronize()); // FIXME: 2018-4-20, if we don't sync here, next search call may crash
        }

        // build center graph
        if (K_center > 0) {
            center_index_->search_int_labels(this_batch_size, points + batch * dim, K_center, center_distances,
                                             center_indicates);
            HANDLE_ERROR(cudaDeviceSynchronize()); // FIXME: I don't know why we should wait there, but if not, we may
                                                   // get some zero in result
        }

// for each point in batch
#pragma omp parallel for
        for (int p = 0; p < this_batch_size; p++) { // suffix '_p' means point
            float *point_distances_p  = point_distances + p * (K_point + 1);
            long * point_indicates_p  = point_indicates + p * (K_point + 1);
            float *center_distances_p = center_distances + p * K_center;
            int *  center_indicates_p = center_indicates + p * K_center;
            float *w_point_p          = point_w + p * K_point;
            float *w_center_p         = center_w + p * K_center;

            // remove the same point from point_graph
            {
                int offset = 0;
                for (int i = 0; i < K_point; i++) {
                    if (point_indicates_p[offset] == batch + p) {
                        offset++; // skip the same point
                    }
                    point_distances_p[i] = point_distances_p[offset];
                    point_indicates_p[i] = point_indicates_p[offset];
                    offset++;
                }
            }

            // distance form faiss is not squared
            for (int i = 0; i < K_point; i++) {
                // point_distances_p[i] *= point_distances_p[i];
            }
            for (int i = 0; i < K_center; i++) {
                // center_distances_p[i] *= center_distances_p[i];
            }

            // calculate W
            float    beta = 1e-5;
            float    minf = -1;
            float    maxf = 1;
            unsigned iter = 0;
            float    sumf = 0;

            // do binary search
            while (iter++ < 200) {
                // calculate Shannon entropy h
                sumf = 0;
                for (int i = 0; i < K_point; i++) {
                    w_point_p[i] = exp(-beta * point_distances_p[i]);
                    sumf += w_point_p[i];
                }
                for (int i = 0; i < K_center; i++) {
                    w_center_p[i] = exp(-beta * center_distances_p[i]);
                    sumf += w_center_p[i];
                }

                float H = 0;

                for (int i = 0; i < K_point; i++) {
                    H += beta * (point_distances_p[i] * w_point_p[i]);
                }
                for (int i = 0; i < K_center; i++) {
                    H += beta * (center_distances_p[i] * w_center_p[i]);
                }

                H = (H / sumf) + log(sumf);

                // update beta
                float Hdiff = H - log(perplexity);

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
                if (beta > std::numeric_limits<float>::max()) beta = std::numeric_limits<float>::max();
            }

            // FIXME: we need a cache friendly implementation
            // output weight
            for (int i = 0; i < K_point; i++) {
                W_point[i][batch + p] = w_point_p[i] / sumf;
            }
            for (int i = 0; i < K_center; i++) {
                W_center[i][batch + p] = w_center_p[i] / sumf;
            }
            // output graph
            for (int i = 0; i < K_point; i++) {
                point_graph[i][batch + p] = point_indicates_p[i];
            }
            for (int i = 0; i < K_center; i++) {
                center_graph[i][batch + p] = center_indicates_p[i];
            }

        } // end loop point in batch
    }     // end loop of batch

    // clear buffer for batch
    if (K_point > 0) {
        HANDLE_ERROR(cudaFreeHost(point_indicates));
        HANDLE_ERROR(cudaFreeHost(point_distances));
    }
    if (K_center > 0) {
        HANDLE_ERROR(cudaFreeHost(center_indicates));
        HANDLE_ERROR(cudaFreeHost(center_distances));
    }

    // symmetric
    // FIXME: scan neighbor of point in degree first graph is extreme cache unfriendly, maybe transpose is need

#pragma omp parallel for
    for (unsigned n = 0; n < points_num; n++) { // loop for points
        for (int i = 0; i < K_point; i++) {     // loop for neighbors
            unsigned id = point_graph[i][n];
            if (id == 0xffffffff) { // not enough point in IVF, we will get this
                break;
            }
            int found_id = -1;
            for (int j = 0; j < K_point; j++) {
                if (point_graph[j][id] == n) {
                    found_id = j;
                    break;
                }
            }
            if (found_id > 0) {
                if (id > n) {
                    float avg             = (W_point[i][n] + W_point[found_id][id]) / 2;
                    W_point[i][n]         = avg;
                    W_point[found_id][id] = avg;
                }
            } else {
                W_point[i][n] /= 2;
            }
        }
        for (int i = 0; i < K_center; i++) {
            W_center[i][n] /= 2;
        }
    }
    printf("finish symmetric perplexity build\n");

    // divide each weight by the sum of weights

    float *sumf_degree = new float[K_point + K_center]; // sumf for per degree

#pragma omp parallel for
    for (int i = 0; i < K_point + K_center; i++) {
        sumf_degree[i] = 0.0;
        if (i < K_point) { // point
            for (unsigned n = 0; n < points_num; n++) {
                sumf_degree[i] += W_point[i][n];
            }
        } else { // center
            for (unsigned n = 0; n < points_num; n++) {
                sumf_degree[i] += W_center[i - K_point][n];
            }
        }
    }

    float sumf = 0;
    for (int i = 0; i < K_point + K_center; i++) {
        sumf += sumf_degree[i];
    }
    delete[] sumf_degree;

#pragma omp parallel for
    for (int i = 0; i < K_point + K_center; i++) {
        if (i < K_point) { // point
            for (unsigned n = 0; n < points_num; n++) {
                W_point[i][n] /= sumf;
            }
        } else { // center
            for (unsigned n = 0; n < points_num; n++) {
                W_center[i - K_point][n] /= sumf;
            }
        }
    }
}
