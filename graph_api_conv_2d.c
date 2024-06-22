#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_graph.h>

#include "utils.h"

cudnnBackendDescriptor_t convolutionDescriptorCreate(
  int64_t numDim, 
  int64_t *pad, 
  int64_t *filterStride, 
  int64_t *dilation, 
  int64_t *upscale, 
  cudnnDataType_t dataType
) {
  cudnnBackendDescriptor_t cDesc;
  cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &cDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                          CUDNN_TYPE_INT64, 1, &numDim));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                          CUDNN_TYPE_DATA_TYPE, 1, &dataType));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                          CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                          CUDNN_TYPE_INT64, numDim, pad));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                          CUDNN_TYPE_INT64, numDim, pad));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS,
                          CUDNN_TYPE_INT64, numDim, dilation));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                          CUDNN_TYPE_INT64, numDim, filterStride));
  CHECK_CUDNN(cudnnBackendFinalize(cDesc));
  return cDesc;
}

cudnnBackendDescriptor_t convolutionForwardDescriptorCreate(
  cudnnBackendDescriptor_t cDesc, 
  cudnnBackendDescriptor_t xDesc, 
  cudnnBackendDescriptor_t wDesc, 
  cudnnBackendDescriptor_t yDesc, 
  cudnnDataType_t dataType
) {
  cudnnBackendDescriptor_t fprop;
  float alpha = 1.0;
  float beta = 0.5;

  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                  &fprop));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop,
  CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                          CUDNN_TYPE_FLOAT, 1, &alpha));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                          CUDNN_TYPE_FLOAT, 1, &beta));
  CHECK_CUDNN(cudnnBackendFinalize(fprop));
  return fprop;
}

cudnnBackendDescriptor_t graphDescriptorCreate(cudnnBackendDescriptor_t fprop, cudnnHandle_t handle) {
  cudnnBackendDescriptor_t op_graph;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));
  CHECK_CUDNN(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &fprop));
  printf("handle: %p\n", handle);
  CHECK_CUDNN(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                          CUDNN_TYPE_HANDLE, 1, &handle));

  printf("op_graph: %p\n", op_graph);
  CHECK_CUDNN(cudnnBackendFinalize(op_graph));
  return op_graph;
}

cudnnBackendDescriptor_t engineDescriptorCreate(cudnnBackendDescriptor_t op_graph) {
  cudnnBackendDescriptor_t engine;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
  CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
  int64_t gidx = 0;
  CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                          CUDNN_TYPE_INT64, 1, &gidx));
  CHECK_CUDNN(cudnnBackendFinalize(engine));
  return engine;
}

struct EngineConfig {
  cudnnBackendDescriptor_t engcfg;
  int64_t workspaceSize;
};

struct EngineConfig engineConfigDescriptorCreate(cudnnBackendDescriptor_t engine) {
  cudnnBackendDescriptor_t engcfg;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engcfg));
  CHECK_CUDNN(cudnnBackendSetAttribute(engcfg, CUDNN_ATTR_ENGINECFG_ENGINE,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine));
  CHECK_CUDNN(cudnnBackendFinalize(engcfg));

  int64_t workspaceSize;
  CHECK_CUDNN(cudnnBackendGetAttribute(engcfg, CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                          CUDNN_TYPE_INT64, 1, NULL, &workspaceSize));
  struct EngineConfig config = {engcfg, workspaceSize};
  return config;
}

cudnnBackendDescriptor_t planDescriptorCreate(struct EngineConfig config, cudnnHandle_t handle) {
  cudnnBackendDescriptor_t plan;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
  CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
  CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config.engcfg));
  CHECK_CUDNN(cudnnBackendFinalize(plan));

  int64_t workspaceSize;
  CHECK_CUDNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                          CUDNN_TYPE_INT64, 1, NULL, &config.workspaceSize));
  return plan;
}

cudnnBackendDescriptor_t varPackDescriptorCreate(cudnnBackendDescriptor_t plan, void *xData, void *wData, void *yData) {
  void *dev_ptrs[3] = {xData, wData, yData}; // device pointer
  int64_t uids[3] = {'x', 'w', 'y'};
  void *workspace = NULL;

  cudnnBackendDescriptor_t varpack;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack));
  CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                          CUDNN_TYPE_VOID_PTR, 3, dev_ptrs));
  CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                          CUDNN_TYPE_INT64, 3, uids));
  CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                          CUDNN_TYPE_VOID_PTR, 1, &workspace));
  CHECK_CUDNN(cudnnBackendFinalize(varpack));
  return varpack;
}

int main() {
  cudnnHandle_t handle;
  CHECK_CUDNN(cudnnCreate(&handle));

  printf("CUDNN version: %ld\n", cudnnGetVersion());

  // input tensor
  int64_t n = 2;
  int64_t c = 3;
  int64_t h = 5;
  int64_t w = 5;

  // filter tensor
  int64_t k = 4;
  int64_t r = 3;
  int64_t kh = 3;
  int64_t kw = 3;

  // output tensor
  int64_t oh = 5;
  int64_t ow = 5;

  int64_t pad[2] = {1, 1};
  int64_t filterStride[2] = {1, 1};
  int64_t dilation[2] = {1, 1};
  int64_t upscale[2] = {1, 1};

  int64_t xDim[4] = {n, c, h, w};
  int64_t xStride[4] = {c*h*w, h*w, w, 1};
  cudnnBackendDescriptor_t xDesc = tensorDescriptorCreate(4, xDim, xStride, 4, CUDNN_DATA_FLOAT, 'x');

  printf("xDesc: created\n");

  int64_t wDim[4] = {k, c, kh, kw};
  int64_t wStride[4] = {c*kh*kw, kh*kw, kw, 1};
  // cudnnBackendDescriptor_t wDesc = tensorDescriptorCreate(4, wDim, wStride, 4, CUDNN_DATA_FLOAT, 'w');
  cudnnBackendDescriptor_t wDesc = tensorDescriptorCreate(4, wDim, wStride, 4, CUDNN_DATA_FLOAT, 'x');

  printf("wDesc: created\n");

  int64_t yDim[4] = {n, k, oh, ow};
  int64_t yStride[4] = {k*oh*ow, oh*ow, ow, 1};
  cudnnBackendDescriptor_t yDesc = tensorDescriptorCreate(4, yDim, yStride, 4, CUDNN_DATA_FLOAT, 'y');

  printf("yDesc: created\n");

  cudnnBackendDescriptor_t cDesc = convolutionDescriptorCreate(2, pad, filterStride, dilation, upscale, CUDNN_DATA_FLOAT);

  printf("cDesc: created\n");

  cudnnBackendDescriptor_t fprop = convolutionForwardDescriptorCreate(cDesc, xDesc, wDesc, yDesc, CUDNN_DATA_FLOAT);

  printf("fprop: created\n");

  cudnnBackendDescriptor_t op_graph = graphDescriptorCreate(fprop, handle);

  printf("op_graph: created\n");

  cudnnBackendDescriptor_t engine = engineDescriptorCreate(op_graph);

  printf("engine: created\n");

  struct EngineConfig config = engineConfigDescriptorCreate(engine);

  printf("config: created\n");

  cudnnBackendDescriptor_t plan = planDescriptorCreate(config, handle);

  printf("Workspace size: %ld\n", config.workspaceSize);

  // Allocate memory for input, filter and output tensors
  float *xData, *wData, *yData;
  CHECK_CUDA(cudaMalloc((void**)&xData, n*c*h*w*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&wData, k*c*kh*kw*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&yData, n*k*oh*ow*sizeof(float)));

  cudnnBackendDescriptor_t varpack = varPackDescriptorCreate(plan, xData, wData, yData);

  // Execute the plan
  CHECK_CUDNN(cudnnBackendExecute(handle, plan, varpack));

  // Free resources
  CHECK_CUDA(cudaFree(xData));
  CHECK_CUDA(cudaFree(wData));
  CHECK_CUDA(cudaFree(yData));
  CHECK_CUDNN(cudnnDestroy(handle));

  // Free descriptors
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(xDesc));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(wDesc));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(yDesc));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(cDesc));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(fprop));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(op_graph));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(engine));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(config.engcfg));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(plan));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(varpack));

  return 0;
}
