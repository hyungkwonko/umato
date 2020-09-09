#pragma once

#include "vendor/faiss/gpu/GpuIndexFlat.h"
#include "vendor/faiss/gpu/utils/CopyUtils.cuh"
#include "vendor/faiss/gpu/utils/DeviceUtils.h"

// #include <limits>

namespace faiss {
namespace gpu {
namespace qvis_patch {
class GpuIndexFlat : public faiss::gpu::GpuIndexFlat {
  public:
    void search_int_labels(faiss::Index::idx_t n, const float *x, faiss::Index::idx_t k, float *distances,
                           int *labels) const;
};

void GpuIndexFlat::search_int_labels(faiss::Index::idx_t n, const float *x, faiss::Index::idx_t k, float *distances,
                                     int *labels) const {
    if (n == 0) {
        return;
    }

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(n <= (faiss::Index::idx_t)std::numeric_limits<int>::max(),
                           "GPU index only supports up to %zu indices", (size_t)std::numeric_limits<int>::max());
    FAISS_THROW_IF_NOT_FMT(k <= 1024, "GPU only supports k <= 1024 (requested %d)",
                           (int)k); // select limitation

    DeviceScope scope(device_);
    auto        stream = resources_->getDefaultStream(device_);

    // The input vectors may be too large for the GPU, but we still
    // assume that the output distances and labels are not.
    // Go ahead and make space for output distances and labels on the
    // GPU.
    // If we reach a point where all inputs are too big, we can add
    // another level of tiling.
    DeviceTensor<float, 2, true> outDistances;
    if (distances == nullptr) {
        outDistances =
            DeviceTensor<float, 2, true>(resources_->getMemoryManagerCurrentDevice(), {(int)n, (int)k}, stream);
    } else {
        outDistances = toDevice<float, 2>(resources_, device_, distances, stream, {(int)n, (int)k});
    }

    // FlatIndex only supports an interface returning int indices
    // search_int_labels of use int indices
    auto outIntIndices = toDevice<int, 2>(resources_, device_, labels, stream, {(int)n, (int)k});

    bool usePaged = false;

    if (getDeviceForAddress(x) == -1) {
        // It is possible that the user is querying for a vector set size
        // `x` that won't fit on the GPU.
        // In this case, we will have to handle paging of the data from CPU
        // -> GPU.
        // Currently, we don't handle the case where the output data won't
        // fit on the GPU (e.g., n * k is too large for the GPU memory).
        size_t dataSize = (size_t)n * this->d * sizeof(float);

        if (dataSize >= minPagedSize_) {
            searchFromCpuPaged_(n, x, k, outDistances.data(), outIntIndices.data());
            usePaged = true;
        }
    }

    if (!usePaged) {
        searchNonPaged_(n, x, k, outDistances.data(), outIntIndices.data());
    }

    // Copy back if necessary
    if (distances != nullptr) {
        fromDevice<float, 2>(outDistances, distances, stream);
    }
    fromDevice<int, 2>(outIntIndices, labels, stream);
}
} // namespace qvis_patch
} // namespace gpu
} // namespace faiss
