#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>

// faiss
#include "vendor/faiss/IndexFlat.h"
#include "vendor/faiss/gpu/GpuIndexFlat.h"
#include "vendor/faiss/gpu/StandardGpuResources.h"
// cmdline
#include "vendor/cmdline/cmdline.h"

// qvis
#include "qvis_io.h"
using namespace std;

bool check_file_exists(const char *path) {
    ifstream f(path);
    return f.good();
}
int main(int argc, char **argv) {
    cmdline::parser parser;
    parser.add<string>("datafile", 'b', "base vectors file path", false, "");
    parser.add<string>("lowdim_datafile", 'l', "lowdim vector file path", false, "");
    parser.add<string>("top1_gt", 'g', "top 1 neighbor ground truth file path", true);
    parser.add<int>("top_num", 't', "number of lowdim neighbor", false, 1);
    parser.add("cpuindex", '\0', "use cpu index");

    parser.parse_check(argc, argv);
    faiss::gpu::StandardGpuResources gpuresource;

    unsigned  points_num;
    unsigned *top1_gt = nullptr;

    if (!check_file_exists(parser.get<string>("top1_gt").c_str())) {
        string datafile_path = parser.get<string>("datafile");
        assert(check_file_exists(datafile_path.c_str()));
        // load data
        unsigned dim;
        float *  data;
        load_data(datafile_path.c_str(), data, points_num, dim);
        printf("Data load successful. N = %u, dim = %u\n", points_num, dim);
        faiss::Index *data_index = nullptr;
        if (parser.exist("cpuindex")) {
            printf("CPUINDEX\n");
            data_index = new faiss::IndexFlat(dim, faiss::METRIC_L2);
        } else {
            printf("GPUINDEX\n");
            data_index = new faiss::gpu::GpuIndexFlat(&gpuresource, dim, faiss::METRIC_L2);
        }
        // add data to index
        printf("Adding data to index\n");
        data_index->add(points_num, data);
        printf("Data loaded into index\n");
        // search ground truth
        top1_gt              = new unsigned[points_num];
        long * top1_gt_long  = new long[2 * points_num];
        float *top1_distance = new float[2 * points_num];
        data_index->search(points_num, data, 2, top1_distance, top1_gt_long);
        for (unsigned i = 0; i < points_num; i++) {
            top1_gt[i] = top1_gt_long[i * 2 + 1];
        }
        delete data_index;
        delete[] top1_gt_long;
        delete[] top1_distance;
        save_label(parser.get<string>("top1_gt").c_str(), top1_gt, 1, points_num);
    } else {
        load_label(parser.get<string>("top1_gt").c_str(), top1_gt, &points_num);
        printf("%u points' ground truth loaded\n", points_num);
    }
    printf("got ground_truth\n");

    if (parser.get<string>("lowdim_datafile") == "") {
        return 0;
    }

    // lowdim top1
    float *  low_data;
    unsigned low_points_num, low_dim;
    load_data(parser.get<string>("lowdim_datafile").c_str(), low_data, low_points_num, low_dim);
    printf("%u points' lowdim points loades\n", low_points_num);
    assert(points_num == low_points_num);

    faiss::Index *low_index = new faiss::gpu::GpuIndexFlat(&gpuresource, low_dim, faiss::METRIC_L2);

    printf("Adding low dimension data to index\n");
    low_index->add(points_num, low_data);

    int top_num = parser.get<int>("top_num");
    printf("searching for lowdim top1\n");
    long * top1_compare = new long[(top_num + 1) * points_num];
    float *low_distance = new float[(top_num + 1) * points_num];
    low_index->search(points_num, low_data, (top_num + 1), low_distance, top1_compare);
    // delete[] low_distance;

    // calc difference
    unsigned diff = 0;
    for (unsigned i = 0; i < points_num; i++) {
        unsigned this_diff = 1;
        for (int j = 1; j <= top_num; j++) {
            this_diff &= top1_gt[i] != top1_compare[(top_num + 1) * i + j];
        }
        diff += this_diff;
    }

    printf("points_num = %u, diff = %u\n", points_num, diff);
    printf("Top %d error %.8f\n", top_num, double(diff) / points_num);
    return 0;
}
