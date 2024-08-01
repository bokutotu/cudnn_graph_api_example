#include <stdio.h>
#include <stdlib.h>

#include <cudnn.h>
// #include <cudnn_graph.h>
#include <string>

#include "utils.h"

void assertDescriptorIsNull(cudnnBackendDescriptor_t desc) {
  if (desc == NULL) {
    fprintf(stderr, "Error: descriptor is not NULL\n");
    exit(-1);
  }
}

cudnnBackendDescriptor_t tensorDescriptorCreate(
  int64_t numDim, 
  int64_t *dim, 
  int64_t *stride, 
  int64_t byteAlignment, 
  cudnnDataType_t dataType, 
  std::string name
) {
  const char *name_ptr = name.c_str();
  cudnnBackendDescriptor_t tensorDesc;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensorDesc));
  assertDescriptorIsNull(tensorDesc);

  CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, 
                           CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dataType));
  assertDescriptorIsNull(tensorDesc);

  CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, 
                          CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, numDim, dim));
  assertDescriptorIsNull(tensorDesc);

  CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, 
                           CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, numDim, stride));
  assertDescriptorIsNull(tensorDesc);

  CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, 
                           CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &byteAlignment));
  assertDescriptorIsNull(tensorDesc);

  CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, 
                           CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, name_ptr));
  assertDescriptorIsNull(tensorDesc);

  CHECK_CUDNN(cudnnBackendFinalize(tensorDesc));
  assertDescriptorIsNull(tensorDesc);

  return tensorDesc;
}
