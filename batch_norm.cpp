#include "utils.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cudnn.h>

cudnnBackendDescriptor_t init_graph(cudnnHandle_t cudnn) {
  cudnnBackendDescriptor_t graph;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph));
  CHECK_CUDNN(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnn));
  return graph;
}

void finalize_graph(cudnnBackendDescriptor_t graph) {
  CHECK_CUDNN(cudnnBackendFinalize(graph));
}

cudnnBackendDescriptor_t create_engine_by_graph(cudnnBackendDescriptor_t graph) {
  cudnnBackendDescriptor_t engine;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
  CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph));
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

class NormConfig {
private:
  cudnnHandle_t cudnn;

  cudnnBackendDescriptor_t norm_desc;
  cudnnBackendDescriptor_t mode;
  cudnnBackendDescriptor_t phase;
  cudnnBackendDescriptor_t x_desc;
  cudnnBackendDescriptor_t y_desc;
  cudnnBackendDescriptor_t mean_desc;
  cudnnBackendDescriptor_t inv_var_desc;
  cudnnBackendDescriptor_t scale_desc;
  cudnnBackendDescriptor_t bias_desc;
  cudnnBackendDescriptor_t epsilon_desc;
  cudnnBackendDescriptor_t input_running_mean_desc;
  cudnnBackendDescriptor_t input_running_var_desc;
  cudnnBackendDescriptor_t output_running_mean_desc;
  cudnnBackendDescriptor_t output_running_var_desc;

  cudnnBackendDescriptor_t op_graph;

  void setAttribute(cudnnBackendDescriptor_t desc, cudnnBackendAttributeName_t attr, cudnnBackendAttributeType_t type, int64_t num, void *value) {
    CHECK_CUDNN(cudnnBackendSetAttribute(desc, attr, type, num, value));
  }

  void setTensorAttribute(cudnnBackendDescriptor_t desc, cudnnBackendAttributeName_t attr, cudnnBackendDescriptor_t tensor_desc) {
    this->setAttribute(desc, attr, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor_desc);
  }

public:
  NormConfig(cudnnHandle_t cudnn_, cudnnBackendDescriptor_t graph) : cudnn(cudnn_), op_graph(graph) {}

  void CreateNormDesc(
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    cudnnBackendNormMode_t mode,
    cudnnBackendNormFwdPhase_t phase
  ) {
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, &this->norm_desc));
    setAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, CUDNN_TYPE_NORM_MODE, 1, &mode);
    setAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase);

    int64_t dims[4] = {batch_size, channels, height, width};
    int64_t strides[4] = {channels * height * width, height * width, width, 1};
    int64_t scalar[4] = {1, 1, 1, 1};
    int64_t dim2d[4] = {1, channels, 1, 1};
    int64_t dim2d_stride[4] = {channels, 1, channels, channels};

    this->x_desc = tensorDescriptorCreate(4, dims, strides, 4, CUDNN_DATA_FLOAT, std::string("x"));
    this->y_desc = tensorDescriptorCreate(4, dims, strides, 4, CUDNN_DATA_FLOAT, std::string("y"));
    this->mean_desc = tensorDescriptorCreate(4, dim2d, dim2d_stride, 4, CUDNN_DATA_FLOAT, std::string("mean"));
    this->inv_var_desc = tensorDescriptorCreate(4, dim2d, dim2d_stride, 4, CUDNN_DATA_FLOAT, std::string("inv_var"));
    this->scale_desc = tensorDescriptorCreate(4, dim2d, dim2d_stride, 4, CUDNN_DATA_FLOAT, std::string("scale"));
    this->bias_desc = tensorDescriptorCreate(4, dim2d, dim2d_stride, 4, CUDNN_DATA_FLOAT, std::string("bias"));
    this->epsilon_desc = tensorDescriptorCreate(4, scalar, scalar, 4, CUDNN_DATA_FLOAT, std::string("epsilon"));
    this->input_running_mean_desc = tensorDescriptorCreate(4, dim2d, dim2d_stride, 4, CUDNN_DATA_FLOAT, std::string("input_running_mean"));
    this->input_running_var_desc = tensorDescriptorCreate(4, dim2d, dim2d_stride, 4, CUDNN_DATA_FLOAT, std::string("input_running_var"));
    this->output_running_mean_desc = tensorDescriptorCreate(4, dim2d, dim2d_stride, 4, CUDNN_DATA_FLOAT, std::string("output_running_mean"));
    this->output_running_var_desc = tensorDescriptorCreate(4, dim2d, dim2d_stride, 4, CUDNN_DATA_FLOAT, std::string("output_running_var"));

    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, this->x_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, this->mean_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, this->inv_var_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, this->scale_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, this->bias_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, this->epsilon_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC, this->input_running_mean_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC, this->input_running_var_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC, this->output_running_mean_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC, this->output_running_var_desc);
    setTensorAttribute(this->norm_desc, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, this->y_desc);

    CHECK_CUDNN(cudnnBackendFinalize(this->norm_desc));
  };

  void register_graph() {
    CHECK_CUDNN(cudnnBackendSetAttribute(this->op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &this->norm_desc));
  }
};

int main() {
  cudnnHandle_t cudnn;
  CHECK_CUDNN(cudnnCreate(&cudnn));
  cudnnBackendDescriptor_t graph = init_graph(cudnn);

  NormConfig norm_config = NormConfig(cudnn, graph);
  norm_config.CreateNormDesc(4, 3, 224, 224, CUDNN_BATCH_NORM, CUDNN_NORM_FWD_TRAINING);
  norm_config.register_graph();

  finalize_graph(graph);

  cudnnBackendDescriptor_t engine = create_engine_by_graph(graph);

  struct EngineConfig config = engineConfigDescriptorCreate(engine);

  return 0;
}
