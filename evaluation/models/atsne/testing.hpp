#pragma once
#include <chrono>
#include <iostream>
namespace qvis {
namespace test {
typedef std::chrono::system_clock::time_point time_point;
time_point                                    now() { return std::chrono::system_clock::now(); }

double getmilliseconds(time_point start, time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

template <typename T, int buffer_size = 4096>
void print_gpu(T *array, int count, int linebreak = 0, T mul = T(1.0)) {
    static T host[buffer_size];
    if (count > buffer_size) {
        count = buffer_size;
    }
    cudaMemcpy(host, array, sizeof(T) * count, cudaMemcpyDeviceToHost);
    for (int i = 0; i < count; i++) {
        std::cout << host[i] * mul << '\t';
        if (linebreak && (i + 1) % linebreak == 0) {
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
}

} // namespace test
} // namespace qvis
