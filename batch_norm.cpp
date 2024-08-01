#include "utils.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cudnn.h>

class NormConfig {
private:
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

  void setAttribute(cudnnBackendDescriptor_t desc, cudnnBackendAttributeName_t attr, cudnnBackendAttributeType_t type, int64_t num, void *value) {
    CHECK_CUDNN(cudnnBackendSetAttribute(desc, attr, type, num, value));
  }

  void setTensorAttribute(cudnnBackendDescriptor_t desc, cudnnBackendAttributeName_t attr, cudnnBackendDescriptor_t tensor_desc) {
    this->setAttribute(desc, attr, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor_desc);
  }

public:
  NormConfig() {};
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
    this->mean_desc = tensorDescriptorCreate(4, dim2d, dim2d, 4, CUDNN_DATA_FLOAT, std::string("mean"));
    this->inv_var_desc = tensorDescriptorCreate(4, dim2d, dim2d, 4, CUDNN_DATA_FLOAT, std::string("inv_var"));
    this->scale_desc = tensorDescriptorCreate(4, dim2d, dim2d, 4, CUDNN_DATA_FLOAT, std::string("scale"));
    this->bias_desc = tensorDescriptorCreate(4, dim2d, dim2d, 4, CUDNN_DATA_FLOAT, std::string("bias"));
    this->epsilon_desc = tensorDescriptorCreate(4, scalar, scalar, 4, CUDNN_DATA_FLOAT, std::string("epsilon"));
    this->input_running_mean_desc = tensorDescriptorCreate(4, dim2d, dim2d, 4, CUDNN_DATA_FLOAT, std::string("input_running_mean"));
    this->input_running_var_desc = tensorDescriptorCreate(4, dim2d, dim2d, 4, CUDNN_DATA_FLOAT, std::string("input_running_var"));
    this->output_running_mean_desc = tensorDescriptorCreate(4, dim2d, dim2d, 4, CUDNN_DATA_FLOAT, std::string("output_running_mean"));
    this->output_running_var_desc = tensorDescriptorCreate(4, dim2d, dim2d, 4, CUDNN_DATA_FLOAT, std::string("output_running_var"));

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
};

int main() {
  NormConfig norm_config;
  norm_config.CreateNormDesc(4, 3, 224, 224, CUDNN_BATCH_NORM, CUDNN_NORM_FWD_TRAINING);
  return 0;
}
