#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <cudnn.h>
#include <cudnn_graph.h>

#define CHECK_CUDA(func)                                    \
{                                                           \
  cudaError_t status = (func);                              \
  if (status != cudaSuccess) {                              \
    fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
    fprintf(stderr, "code: %d, reason: %s\n", status, cudaGetErrorString(status)); \
    exit(-1);                                               \
  }                                                         \
}

#define CHECK_CUDNN(func)                                   \
{                                                           \
  cudnnStatus_t status = (func);                            \
  if (status != CUDNN_STATUS_SUCCESS) {                     \
    fprintf(stderr, "CUDNN Error: %s:%d, ", __FILE__, __LINE__);  \
    fprintf(stderr, "reason: %s\n", cudnnGetErrorString(status)); \
    exit(-1);                                               \
  }                                                         \
}

cudnnBackendDescriptor_t tensorDescriptorCreate(int64_t numDim, 
                                                int64_t *dim, 
                                                int64_t *stride, 
                                                int64_t byteAlignment, 
                                                cudnnDataType_t dataType, 
                                                int64_t name);

#ifdef __cplusplus
}
#endif
