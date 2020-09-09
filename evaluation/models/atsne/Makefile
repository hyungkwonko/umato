NVCC = nvcc
CXX = g++
CPPFLAGS = -march=native -fopenmp -Wall

NVCCFLAGS = -std=c++11 -lcurand -lcublas -lopenblas \
	-Lvendor/faiss/gpu \
	-Lvendor/faiss \
	-l:libfaiss.a \
	-l:libgpufaiss.a \
	-DQVIS_GPU \
	-ccbin $(CXX) \
    -gencode arch=compute_35,code="compute_35" \
    -gencode arch=compute_52,code="compute_52" \
    -gencode arch=compute_60,code="compute_60" \
    -Xcompiler "$(CPPFLAGS)"

all = qvis_gpu test_knn_accuracy test_top1_error

all: $(all)

qvis_gpu: qvis_gpu.cu handle_cuda_err.hpp qvis_io.h qvis_faiss_patch.cuh gradient.cuh graph.hpp testing.hpp tsne.cuh weight.cuh
	$(NVCC) $(NVCCFLAGS) qvis_gpu.cu -o qvis_gpu

test_knn_accuracy: test_knn_accuracy.cu qvis_io.h handle_cuda_err.hpp
	$(NVCC) $(NVCCFLAGS) test_knn_accuracy.cu -o test_knn_accuracy

test_top1_error: test_top1_error.cu qvis_io.h
	$(NVCC) $(NVCCFLAGS) test_top1_error.cu -o test_top1_error

.PHONY : clean
clean:
	rm $(all)
