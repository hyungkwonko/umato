#pragma once

#include <cassert>
#include <cstdio>
#include <vector>

#include "handle_cuda_err.hpp"
#include "matrix.cuh"

namespace qvis {
template <typename ElementType, bool ConstantDegree>
struct Graph {};

// Graph with constant out degree
// This is A GPU Graph
// Note: data in this struct is **degree first**
template <typename ElementType>
struct Graph<ElementType, true> : public MatrixPitched<ElementType> {
    using MatrixPitched<ElementType>::MatrixPitched;

    __host__ __device__ inline unsigned d() const { return this->row; }

    __host__ __device__ inline unsigned n() const { return this->col; }

    __host__ __device__ Graph() {
        this->col   = 0;
        this->row   = 0;
        this->data_ = nullptr;
    }

    __host__ __device__ Graph(unsigned row, unsigned col, size_t pitch, ElementType *data) {
        this->row   = row;
        this->col   = col;
        this->pitch = pitch;
        this->data_ = data;

        if (pitch < col * sizeof(ElementType)) {
            printf("Warning: col is %u, but pitch is %lu\n", col, pitch);
        }
    }

    // __host__ __device__ Graph(const std::vector<std::vector<unsigned> > &g) {
    //     set_graph_gpu(g);
    // }

    // using MatrixPitched<ElementType>::MatrixPitched<ElementType>;
    __host__ void set_graph_gpu(const std::vector<std::vector<unsigned>> &g) {
        this->col = g.size();
        if (this->col == 0) {
            this->row = 0;
            return;
        }
        this->row = g[0].size();
        assert(this->data_ == nullptr);
        HANDLE_ERROR(cudaMallocPitch((void **)&this->data_, &this->pitch, sizeof(ElementType) * this->col, this->row));
        uint8_t *data_host;
        HANDLE_ERROR(cudaMallocHost((void **)&data_host, this->row * this->pitch)); // Unpaged Memory

        for (size_t j = 0; j < this->row; j++) {
            ElementType *data_row = (ElementType *)(data_host + j * this->pitch);
            for (size_t i = 0; i < this->col; i++) {
                data_row[i] = g[i][j];
            }
        }
        HANDLE_ERROR(cudaMemcpy(this->data_, data_host, this->row * this->pitch, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaFreeHost(data_host));
    }

    // set graph
    // @param data n * d
    __host__ void set_graph_gpu(const ElementType *data, unsigned n, unsigned d, unsigned data_row_stride) {
        this->set_data_transpose_gpu(data, n, d, data_row_stride);
    }
};

} // namespace qvis

namespace qvis {
namespace test {} // namespace test
} // namespace qvis
