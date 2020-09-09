#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <utility>
// faiss
#include "vendor/faiss/gpu/GpuIndexFlat.h"
#include "vendor/faiss/gpu/StandardGpuResources.h"
// cmdline
#include "vendor/cmdline/cmdline.h"
// qvis
#include "handle_cuda_err.hpp"
#include "qvis_io.h"

using namespace std;

__global__ void kernel_fill_labels(unsigned points_num, unsigned K, unsigned aligned_num, long *knn_indicates_device,
                                   unsigned *labels, unsigned *knn_labels) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < points_num; idx += gridDim.x * blockDim.x) {
        for (int i = 0; i < K + 1; i++) {
            knn_labels[i * aligned_num + idx] = labels[knn_indicates_device[idx * (K + 1) + i]];
        }
    }
}

__global__ void kernel_knn_label(unsigned points_num, unsigned K, unsigned aligned_num, unsigned *knn_labels) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned     t;
    for (; idx < points_num; idx += gridDim.x * blockDim.x) {
        // bubble sort label [1..K]
        for (int i = 1; i < K; i++) {
            for (int j = 1; j < K; j++) {
                unsigned &a = knn_labels[j * aligned_num + idx];
                unsigned &b = knn_labels[(j + 1) * aligned_num + idx];
                if (a > b) {
                    t = a;
                    a = b;
                    b = t;
                }
            }
        }
        // find common number
        int      result_times = 0, current_times = 0;
        unsigned result_label, current_label     = 0xffffffff;
        for (int i = 1; i <= K; i++) {
            if (knn_labels[i * aligned_num + idx] == current_label) {
                current_times++;

            } else {
                current_times = 1;
                current_label = knn_labels[i * aligned_num + idx];
            }
            if (current_times > result_times) {
                result_label = current_label;
                result_times = current_times;
            }
        }

        // save result
        // knn_labels[aligned_num + idx] = result_label;
        knn_labels[idx] = result_label;
    }
}
set<int> parse_ks(const string &str) {
    set<int> ks;
    int      num = 0;
    for (size_t i = 0; i < str.size(); i++) {
        if (str[i] == ' ' || str[i] == '\t') {
            continue;
        }
        if (str[i] == ',') {
            ks.insert(num);
            num = 0;
            continue;
        }
        if (str[i] >= '0' && str[i] <= '9') {
            num *= 10;
            num += str[i] - '0';
        }
    }
    if (num) {
        ks.insert(num);
    }
    return ks;
}
int main(int argc, char **argv) {
    cmdline::parser parser;
    parser.add<string>("datafile", 'd', "lowdim vector file path", true, "");
    parser.add<string>("labelfile", 'l', "label file path", true, "");
    parser.add<int>("k", 'k', "number neighborhood", false, 5);
    parser.add<string>("ks", '\0', "test multiple K at same time, Ks must seperated by comma", false, "");
    parser.parse_check(argc, argv);

    // parse Ks
    set<int> ks = parse_ks(parser.get<string>("ks"));
    if (parser.exist("k") || ks.size() == 0) {
        ks.insert(parser.get<int>("k"));
    }
    int K = *ks.rbegin();
    printf("Ks: ");
    for (auto it = ks.begin(); it != ks.end(); it++) {
        printf("%d ", *it);
    }
    printf("\n");

    //
    unsigned  points_num, labels_num, dim;
    float *   data, *data_device;
    unsigned *labels = nullptr;

    // load data
    load_data(parser.get<string>("datafile").c_str(), data, points_num, dim);
    printf("Data load successful, N = %u, dim = %u\n", points_num, dim);

    // load label
    load_label(parser.get<string>("labelfile").c_str(), labels, &labels_num);
    printf("Labels laod successful, N = %u\n", labels_num);
    assert(points_num == labels_num);

    // build knn graph
    HANDLE_ERROR(cudaMallocManaged((void **)&data_device, sizeof(float) * points_num * dim));
    HANDLE_ERROR(cudaMemcpy(data_device, data, sizeof(float) * points_num * dim, cudaMemcpyHostToDevice));
    faiss::gpu::StandardGpuResources gpuresource;
    faiss::gpu::GpuIndexFlat *       data_index = nullptr;
    data_index                                  = new faiss::gpu::GpuIndexFlat(&gpuresource, dim, faiss::METRIC_L2);
    data_index->add(points_num, data_device);

    // search for knn
    float *knn_distances_device;
    long * knn_indicates_device;

    HANDLE_ERROR(cudaMallocManaged((void **)&knn_distances_device, sizeof(float) * points_num * (K + 1)));
    HANDLE_ERROR(cudaMallocManaged((void **)&knn_indicates_device, sizeof(long) * points_num * (K + 1)));
    data_index->search(points_num, data_device, K + 1, knn_distances_device, knn_indicates_device);
    HANDLE_ERROR(cudaFree(knn_distances_device));
    HANDLE_ERROR(cudaFree(data_device));

    // get labels
    unsigned *knn_labels, *labels_device;
    unsigned  aligned_num = (points_num + 63) / 64 * 64; // aligned to 256 bytes;

    HANDLE_ERROR(cudaMallocManaged((void **)&knn_labels, sizeof(unsigned) * aligned_num * (K + 1)));
    HANDLE_ERROR(cudaMallocManaged((void **)&labels_device, sizeof(unsigned) * points_num));
    HANDLE_ERROR(cudaMemcpy(labels_device, labels, sizeof(float) * points_num, cudaMemcpyHostToDevice));

    const int ThreadPerBlock = 256;
    kernel_fill_labels<<<50, ThreadPerBlock>>>(points_num, K, aligned_num, knn_indicates_device, labels_device,
                                               knn_labels);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaFree(labels_device));
    HANDLE_ERROR(cudaFree(knn_indicates_device));

    // print result
    printf("K\tratio\tcorrect\tsample_number\n");
    for (auto k = ks.begin(); k != ks.end(); k++) {
        // sort labels and get most common

        kernel_knn_label<<<50, ThreadPerBlock>>>(points_num, *k, aligned_num, knn_labels);
        HANDLE_ERROR(cudaDeviceSynchronize());
        int equal_num = 0;
        for (unsigned i = 0; i < points_num; i++) {
            equal_num += knn_labels[i] == labels[i];
        }
        // sum up
        printf("%d\t%f\t%d\t%u\n", *k, double(equal_num) / points_num, equal_num, points_num);
    }

    HANDLE_ERROR(cudaFree(knn_labels));
    return 0;
}
