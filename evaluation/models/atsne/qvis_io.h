#pragma once
#include "handle_cuda_err.hpp"
#include <fstream>
#include <iostream>

void Load_nn_graph(const char *filename, std::vector<std::vector<unsigned>> &graph) { // Useless
    std::ifstream in(filename, std::ios::binary);
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize          = (size_t)ss;
    size_t num            = (unsigned)(fsize / (k + 1) / 4);
    in.seekg(0, std::ios::beg);

    graph.resize(num);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        graph[i].resize(k);
        in.read((char *)graph[i].data(), k * sizeof(unsigned));
    }
    in.close();
}

void load_data(const char *filename, float *&data, unsigned &num, unsigned &dim) { // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize          = (size_t)ss;
    num                   = (unsigned)(fsize / (dim + 1) / 4);
    data                  = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
}

void load_label(const char *filename, unsigned *&labels, unsigned *num = nullptr) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    unsigned N;
    in.read((char *)&N, sizeof(unsigned));
    if (num != nullptr) {
        *num = N;
    }
    if (labels == nullptr) {
        labels = new unsigned[N];
    }
    std::cout << "label num: " << N << std::endl;
    for (unsigned i = 0; i < N; i++) {
        in.read((char *)(labels + i), sizeof(unsigned));
    }
    in.close();
}

void save_data(const char *filename, float *data, unsigned num, unsigned dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cout << "can not open data output file" << std::endl;
        return;
    }
    for (unsigned i = 0; i < num; i++) {
        out.write(reinterpret_cast<const char *>(&dim), sizeof(unsigned));
        out.write(reinterpret_cast<const char *>(data + i * dim), dim * sizeof(float));
    }
    out.close();
}
void save_label(const char *filename, unsigned *data, unsigned num, unsigned dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cout << "can not open data output file" << std::endl;
        return;
    }
    for (unsigned i = 0; i < num; i++) {
        out.write(reinterpret_cast<const char *>(&dim), sizeof(unsigned));
        out.write(reinterpret_cast<const char *>(data + i * dim), dim * sizeof(unsigned));
    }
    out.close();
}

void save_result(const char *filename, unsigned num, unsigned D, float *data, unsigned *label) {
    auto out = fopen(filename, "w");
    if (out == nullptr) {
        std::cout << "can not open data output file" << std::endl;
        return;
    }
    for (unsigned i = 0; i < num; i++) {
        for (unsigned j = 0; j < D; j++) {
            fprintf(out, "%.6f\t", data[i * D + j]);
        }
        if (label != nullptr) {
            fprintf(out, "%u", label[i]);
        }
        fprintf(out, "\n");
    }
    fclose(out);
}

template <bool binary = false>
void save_gradient(const char *filename, unsigned num_points, int dim, float *grad, float *grad_neg) {
    static float *grad_host = nullptr, *grad_neg_host;
    if (grad_host == nullptr) {
        HANDLE_ERROR(cudaMallocHost((void **)&grad_host, num_points * dim * sizeof(float)));
    }
    if (grad_neg_host == nullptr) {
        HANDLE_ERROR(cudaMallocHost((void **)&grad_neg_host, num_points * dim * sizeof(float)));
    }

    HANDLE_ERROR(cudaMemcpy(grad_host, grad, num_points * dim * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(grad_neg_host, grad_neg, num_points * dim * sizeof(float), cudaMemcpyDeviceToHost));

    FILE *out;
    if (binary) {
        out = fopen(filename, "wb");
    } else {
        out = fopen(filename, "w");
    }
    if (out == nullptr) {
        std::cout << "can not open gradient data output file" << std::endl;
        return;
    }
    if (binary) {
        unsigned dim2 = 2 * dim;
        for (unsigned i = 0; i < num_points; i++) {
            fwrite(reinterpret_cast<const char *>(&dim2), sizeof(unsigned), 1, out);
            fwrite(reinterpret_cast<const char *>(grad_host + i * dim), dim * sizeof(float), 1, out);
            fwrite(reinterpret_cast<const char *>(grad_neg_host + i * dim), dim * sizeof(float), 1, out);
        }
    } else {
        for (unsigned i = 0; i < num_points; i++) {
            fprintf(out, "%.6f\t%.6f\t%.6f\t%.6f\t\n", grad_host[i * dim], grad_host[i * dim + 1],
                    grad_neg_host[i * dim], grad_neg_host[i * dim + 1]);
        }
        fclose(out);
    }
}

void save_args(const char *path, int argc, char **argv) {
    std::ofstream out(path);
    for (int i = 0; i < argc; i++) {
        out << argv[i] << '\t';
    }
}
