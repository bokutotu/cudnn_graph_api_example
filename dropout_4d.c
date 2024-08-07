#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDNN(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error: %s\n", cudnnGetErrorString(status)); \
            exit(1); \
        } \
    } while(0)

int main() {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 4D tensor dimensions
    int n = 1, c = 3, h = 224, w = 224;
    int elements = n * c * h * w;

    // Create tensor descriptor
    cudnnTensorDescriptor_t xDesc, yDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    // Allocate memory
    float *x, *y, *reserveSpace;
    cudaMalloc((void**)&x, elements * sizeof(float));
    cudaMalloc((void**)&y, elements * sizeof(float));

    // Create dropout descriptor
    cudnnDropoutDescriptor_t dropoutDesc;
    CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));

    size_t stateSize;
    CHECK_CUDNN(cudnnDropoutGetStatesSize(cudnn, &stateSize));
    void *states;
    cudaMalloc(&states, stateSize);

    float dropout_probability = 0.5;
    CHECK_CUDNN(cudnnSetDropoutDescriptor(dropoutDesc, cudnn, dropout_probability, states, stateSize, 0));

    // Get workspace size and allocate
    size_t reserveSpaceSize;
    CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(xDesc, &reserveSpaceSize));
    cudaMalloc((void**)&reserveSpace, reserveSpaceSize);

    // Perform dropout
    CHECK_CUDNN(cudnnDropoutForward(cudnn, dropoutDesc, xDesc, x, yDesc, y, reserveSpace, reserveSpaceSize));

    // Clean up
    cudaFree(x);
    cudaFree(y);
    cudaFree(states);
    cudaFree(reserveSpace);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyDropoutDescriptor(dropoutDesc);
    cudnnDestroy(cudnn);

    return 0;
}
