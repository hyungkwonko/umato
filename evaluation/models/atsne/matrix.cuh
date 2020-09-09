#pragma once
#include "handle_cuda_err.hpp"
#include <cassert>

namespace qvis {

bool isManaged(cudaPointerAttributes &attr) {
#if CUDART_VERSION < 10000 // isManaged deprecated in CUDA 10.
    return attr.isManaged != 0;
#else // attr.type doesn't exist before CUDA 10
    return attr.type == cudaMemoryTypeManaged;
#endif
}
template <typename ElementType> struct MatrixPitched {
    // data is optimized for aligned (e.g. GPU) reading, you should not access it directly,
    // using [] operater instead
    ElementType *data_; // Host or GPU, row * col, matrix data
    size_t       pitch; // Host & GPU, pitch bytes, for alligment
    unsigned     col;   // Host & GPU
    unsigned     row;   // Host & GPU

    __host__ __device__ MatrixPitched() {
        col   = 0;
        row   = 0;
        data_ = nullptr;
    }

    __host__ __device__ MatrixPitched(unsigned row, unsigned col, size_t pitch, ElementType *data) {
        this->row   = row;
        this->col   = col;
        this->pitch = pitch;
        this->data_ = data;
    }

    __host__ void free() {
        if (data_ != nullptr) {
            HANDLE_ERROR(cudaFree(data_));
            data_ = nullptr;
        }
    }

    __host__ __device__ ElementType *&data() { return data_; }

    __host__ __device__ const ElementType *data() const { return data_; }

    __host__ void allocate_memory_managed(int col, int row) {
        this->col = col;
        this->row = row;
        if (col * row == 0) {
            return;
        }
        pitch = (sizeof(ElementType) * col + 128 - 1) / 128 * 128;
        HANDLE_ERROR(cudaMallocManaged((void **)&data_, pitch * row));
    }

    // transpose data and set
    // @param data row * data_row_stride,
    // @param row data row
    // @param col data colom
    __host__ void set_data_transpose_gpu(const ElementType *data, unsigned row, unsigned col,
                                         unsigned data_row_stride) {
        this->col = row;
        this->row = col;
        if (row == 0 || col == 0) {
            return;
        }
        assert(data_ == nullptr);
        HANDLE_ERROR(cudaMallocPitch((void **)&data_, &pitch, sizeof(ElementType) * this->col, this->row));
        uint8_t *data_host;
        HANDLE_ERROR(cudaMallocHost((void **)&data_host, this->row * pitch)); // Unpaged Memory

        for (size_t j = 0; j < col; j++) {
            ElementType *data_row = (ElementType *)(data_host + j * pitch);
            for (size_t i = 0; i < row; i++) {
                data_row[i] = data[i * data_row_stride + j];
            }
        }
        HANDLE_ERROR(cudaMemcpy(data_, data_host, this->row * pitch, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaFreeHost(data_host));
    }

    __host__ void set_to_gpu(const MatrixPitched<ElementType> &matrix) {
        if (matrix.data() == nullptr) {
            return;
        }
        cudaPointerAttributes att;
        HANDLE_ERROR(cudaPointerGetAttributes(&att, matrix.data()));
        assert(att.memoryType == cudaMemoryTypeHost ||
               isManaged(att)); // it has been deprecated in favour of cudaPointerAttributes::type.

        *this = matrix;
        if (isManaged(att)) { // use CUDA unified memory
            return;
        }
        HANDLE_ERROR(cudaMalloc((void **)&data_, matrix.row * matrix.pitch));
        HANDLE_ERROR(cudaMemcpy(data_, matrix.data(), this->row * pitch, cudaMemcpyHostToDevice));
    }

    __device__ __host__ const ElementType *operator[](int index) const {
        return (ElementType *)(((uint8_t *)data_) + index * pitch);
    }

    __device__ __host__ ElementType *operator[](int index) {
        return (ElementType *)(((uint8_t *)data_) + index * pitch);
    }

    __host__ void save_data_cpu(const char *filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            std::cout << "can not open data output file" << std::endl;
            return;
        }
        for (unsigned i = 0; i < row; i++) {
            out.write(reinterpret_cast<const char *>(&col), sizeof(unsigned));
            out.write(reinterpret_cast<const char *>(data_ + i * (pitch / sizeof(ElementType))),
                      col * sizeof(ElementType));
        }
        out.close();
    }

    __host__ void load_data_cpu(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cout << "can not open data input file" << std::endl;
            return;
        }
        for (unsigned i = 0; i < row; i++) {
            in.seekg(sizeof(unsigned), std::ios::cur);
            in.read(reinterpret_cast<char *>(data_ + i * (pitch / sizeof(ElementType))), col * sizeof(ElementType));
        }
        in.close();
    }

    __host__ void save_data_gpu(const char *filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            std::cout << "can not open data output file" << std::endl;
            return;
        }
        ElementType *data_host;
        HANDLE_ERROR(cudaMallocHost((void **)&data_host, this->row * pitch)); // Unpaged Memory
        HANDLE_ERROR(cudaMemcpy(data_host, data, this->row * pitch, cudaMemcpyDeviceToHost));

        for (unsigned i = 0; i < row; i++) {
            out.write(reinterpret_cast<const char *>(&col), sizeof(unsigned));
            out.write(reinterpret_cast<const char *>(data_host + i * (pitch / sizeof(ElementType))),
                      col * sizeof(ElementType));
        }
        out.close();
        HANDLE_ERROR(cudaFreeHost(data_host));
    }
};

} // namespace qvis
