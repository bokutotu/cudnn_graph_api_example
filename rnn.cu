#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { cudaError_t status = call; if (status != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status)); exit(EXIT_FAILURE); } }
#define CHECK_CUDNN(call) { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN Error at line %d: %s\n", __LINE__, cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}

const int input_size = 64;
const int hidden_size = 128;
const int output_size = 128;
const int seq_length = 20;
const int batch_size = 64;
const int num_layers = 2;

int main() {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnRNNDescriptor_t rnn_desc;
    CHECK_CUDNN(cudnnCreateRNNDescriptor(&rnn_desc));
    CHECK_CUDNN(cudnnSetRNNDescriptor_v8(rnn_desc, CUDNN_RNN_ALGO_STANDARD, CUDNN_RNN_RELU,
        CUDNN_RNN_NO_BIAS, CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT,
        CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH,
        input_size, hidden_size, hidden_size, num_layers, NULL, CUDNN_RNN_PADDED_IO_DISABLED));

    cudnnRNNDataDescriptor_t x_desc, y_desc;
    CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&y_desc));

    int seq_length_array[batch_size];
    for (int i = 0; i < batch_size; i++) seq_length_array[i] = seq_length;

    CHECK_CUDNN(cudnnSetRNNDataDescriptor(x_desc, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
        seq_length, batch_size, input_size, seq_length_array, NULL));

    CHECK_CUDNN(cudnnSetRNNDataDescriptor(y_desc, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
        seq_length, batch_size, output_size, seq_length_array, NULL));

    cudnnTensorDescriptor_t h_desc, c_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&h_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&c_desc));
    
    int dims[3] = {num_layers, batch_size, hidden_size};
    int strides[3] = {hidden_size * batch_size, hidden_size, 1};
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(h_desc, CUDNN_DATA_FLOAT, 3, dims, strides));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(c_desc, CUDNN_DATA_FLOAT, 3, dims, strides));

    void *x, *y, *hx, *cx, *hy, *cy, *dx, *dy, *dhx, *dcx;
    CHECK_CUDA(cudaMalloc(&x, seq_length * batch_size * input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&y, seq_length * batch_size * output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&hx, num_layers * batch_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cx, num_layers * batch_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&hy, num_layers * batch_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cy, num_layers * batch_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dx, seq_length * batch_size * input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, seq_length * batch_size * output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dhx, num_layers * batch_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dcx, num_layers * batch_size * hidden_size * sizeof(float)));

    size_t weight_size, workspace_size, reserve_size;
    CHECK_CUDNN(cudnnGetRNNWeightSpaceSize(cudnn, rnn_desc, &weight_size));
    CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(cudnn, rnn_desc, CUDNN_FWD_MODE_TRAINING, x_desc, &workspace_size, &reserve_size));

    printf("Weight size: %ld\n", weight_size);
    printf("Workspace size: %ld\n", workspace_size);

    void *weights, *dweights, *workspace, *reserve_space;
    CHECK_CUDA(cudaMalloc(&weights, weight_size));
    CHECK_CUDA(cudaMalloc(&dweights, weight_size));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    CHECK_CUDA(cudaMalloc(&reserve_space, reserve_size));

    CHECK_CUDNN(cudnnRNNForward(cudnn, rnn_desc, CUDNN_FWD_MODE_TRAINING,
        NULL, x_desc, x, y_desc, y,
        h_desc, hx, hy, c_desc, cx, cy,
        weight_size, weights, workspace_size, workspace,
        reserve_size, reserve_space));

    CHECK_CUDNN(cudnnRNNBackwardData_v8(cudnn, rnn_desc,
        NULL, y_desc, y, dy, x_desc, dx,
        h_desc, hx, dhx, dhx, c_desc, cx, dcx, dcx,
        weight_size, weights, workspace_size, workspace,
        reserve_size, reserve_space));

    CHECK_CUDNN(cudnnRNNBackwardWeights_v8(cudnn, rnn_desc,
        CUDNN_WGRAD_MODE_ADD, NULL,
        x_desc, x, h_desc, hx, y_desc, y,
        weight_size, dweights, workspace_size, workspace,
        reserve_size, reserve_space));

    cudnnDestroyRNNDataDescriptor(x_desc);
    cudnnDestroyRNNDataDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(h_desc);
    cudnnDestroyTensorDescriptor(c_desc);
    cudnnDestroyRNNDescriptor(rnn_desc);
    cudnnDestroy(cudnn);

    cudaFree(x); cudaFree(y); cudaFree(hx); cudaFree(cx);
    cudaFree(hy); cudaFree(cy); cudaFree(dx); cudaFree(dy);
    cudaFree(dhx); cudaFree(dcx); cudaFree(weights);
    cudaFree(dweights); cudaFree(workspace);
    cudaFree(reserve_space);

    return 0;
}
