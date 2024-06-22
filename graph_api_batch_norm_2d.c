// #include "utils.h"
//
// #include <cudnn_graph_v9.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <cudnn.h>
// #include <cudnn_graph.h>
//
// cudnnBackendDescriptor_t normalizationDescriptorCreate(cudnnBackendNormMode_t mode,
//                                                        cudnnBackendNormFwdPhase_t phase,
//                                                        cudnnBackendDescriptor_t xDesc,
//                                                        cudnnBackendDescriptor_t yDesc,
//                                                        cudnnBackendDescriptor_t runningMeanDesc,
//                                                        cudnnBackendDescriptor_t runningVarianceDesc,
//                                                        cudnnBackendDescriptor_t saveMeanDesc,
//                                                        cudnnBackendDescriptor_t saveInvVarianceDesc,
//                                                        cudnnBackendDescriptor_t scaleDesc,
//                                                        cudnnBackendDescriptor_t biasDesc,
//                                                        cudnnBackendDescriptor_t epsilonDesc,
//                                                        cudnnBackendDescriptor_t momentumDesc ) {
//   cudnnBackendDescriptor_t normDesc;
//   CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, &normDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, 
//                                        CUDNN_TYPE_NORM_MODE, 1, &mode));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, 
//                                        CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, 
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, 
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &saveMeanDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, 
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &saveInvVarianceDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &epsilonDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &momentumDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningMeanDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningVarianceDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningMeanDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningVarianceDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scaleDesc));
//
//   CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
//                                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &biasDesc));
//
//   CHECK_CUDNN(cudnnBackendFinalize(normDesc));
//   return normDesc;
// }
//
// cudnnBackendDescriptor_t createOperationGraph(cudnnHandle_t handle, cudnnBackendDescriptor_t bnDesc) {
//     cudnnBackendDescriptor_t opGraph;
//     CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));
//     CHECK_CUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
//                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bnDesc));
//     CHECK_CUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
//                                         CUDNN_TYPE_HANDLE, 1, &handle));
//     CHECK_CUDNN(cudnnBackendFinalize(opGraph));
//     return opGraph;
// }
//
// cudnnBackendDescriptor_t createEngine(cudnnBackendDescriptor_t opGraph) {
//     cudnnBackendDescriptor_t engine;
//     CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
//     CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
//                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph));
//     int64_t gidx = 0;
//     CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
//                                         CUDNN_TYPE_INT64, 1, &gidx));
//     CHECK_CUDNN(cudnnBackendFinalize(engine));
//     return engine;
// }
//
// cudnnBackendDescriptor_t createEngineConfig(cudnnBackendDescriptor_t engine) {
//     cudnnBackendDescriptor_t engCfg;
//     CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engCfg));
//     CHECK_CUDNN(cudnnBackendSetAttribute(engCfg, CUDNN_ATTR_ENGINECFG_ENGINE,
//                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine));
//     CHECK_CUDNN(cudnnBackendFinalize(engCfg));
//     return engCfg;
// }
//
// cudnnBackendDescriptor_t createExecutionPlan(cudnnHandle_t handle, cudnnBackendDescriptor_t engCfg) {
//     cudnnBackendDescriptor_t plan;
//     CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
//     CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, 
//                                         CUDNN_TYPE_HANDLE, 1, &handle));
//     CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
//                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engCfg));
//     CHECK_CUDNN(cudnnBackendFinalize(plan));
//     return plan;
// }
//
// cudnnBackendDescriptor_t createVariantPack(void** devPtrs, int64_t* uids, int numPtrs, void* workspace) {
//     cudnnBackendDescriptor_t varPack;
//     CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varPack));
//     CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
//                                         CUDNN_TYPE_VOID_PTR, numPtrs, devPtrs));
//     CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
//                                         CUDNN_TYPE_INT64, numPtrs, uids));
//     CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
//                                         CUDNN_TYPE_VOID_PTR, 1, &workspace));
//     CHECK_CUDNN(cudnnBackendFinalize(varPack));
//     return varPack;
// }
//
// int main() {
//   cudnnHandle_t handle;
//   CHECK_CUDNN(cudnnCreate(&handle));
//
//   int64_t n = 2;
//   int64_t c = 3;
//   int64_t h = 5;
//   int64_t w = 5;
//
//   cudnnBackendDescriptor_t xDesc = tensorDescriptorCreate(4, 
//                                                           (int64_t[]){n, c, h, w}, 
//                                                           (int64_t[]){c*h*w, h*w, w, 1}, 
//                                                           4, 
//                                                           CUDNN_DATA_FLOAT, 
//                                                           'x');
//
//   cudnnBackendDescriptor_t yDesc = tensorDescriptorCreate(4, 
//                                                           (int64_t[]){n, c, h, w}, 
//                                                           (int64_t[]){c*h*w, h*w, w, 1}, 
//                                                           4, 
//                                                           CUDNN_DATA_FLOAT, 
//                                                           'y');
//
//   cudnnBackendDescriptor_t runningMeanDesc = tensorDescriptorCreate(4, 
//                                                                     (int64_t[]){1, c, 1, 1}, 
//                                                                     (int64_t[]){c, 1, c, c}, 
//                                                                     4, 
//                                                                     CUDNN_DATA_FLOAT, 
//                                                                     0);
//
//   cudnnBackendDescriptor_t runningVarianceDesc = tensorDescriptorCreate(4, 
//                                                                         (int64_t[]){1, c, 1, 1}, 
//                                                                         (int64_t[]){c, 1, c, c}, 
//                                                                         4, 
//                                                                         CUDNN_DATA_FLOAT, 
//                                                                         1);
//
//   cudnnBackendDescriptor_t scaleDesc = tensorDescriptorCreate(4, 
//                                                               (int64_t[]){1, c, 1, 1}, 
//                                                               (int64_t[]){c, 1, c, c}, 
//                                                               4, 
//                                                               CUDNN_DATA_FLOAT, 
//                                                               2);
//
//   cudnnBackendDescriptor_t biasDesc = tensorDescriptorCreate(4, 
//                                                             (int64_t[]){1, c, 1, 1}, 
//                                                             (int64_t[]){c, 1, c, c}, 
//                                                             4, 
//                                                             CUDNN_DATA_FLOAT, 
//                                                             3);
//
//   cudnnBackendDescriptor_t epsilonDesc = tensorDescriptorCreate(4, 
//                                                                 (int64_t[]){1, 1, 1, 1}, 
//                                                                 (int64_t[]){1, 1, 1, 1}, 
//                                                                 4, 
//                                                                 CUDNN_DATA_FLOAT, 
//                                                                 4);
//
//   cudnnBackendDescriptor_t momentumDesc = tensorDescriptorCreate(4, 
//                                                                 (int64_t[]){1, 1, 1, 1}, 
//                                                                 (int64_t[]){1, 1, 1, 1}, 
//                                                                 4, 
//                                                                 CUDNN_DATA_FLOAT, 
//                                                                 5);
//
//   cudnnBackendDescriptor_t savingMeanDesc = tensorDescriptorCreate(4, 
//                                                                     (int64_t[]){1, c, 1, 1}, 
//                                                                     (int64_t[]){c, 1, c, c}, 
//                                                                     4, 
//                                                                     CUDNN_DATA_FLOAT, 
//                                                                     6);
//
//   cudnnBackendDescriptor_t savingInvVarDesc = tensorDescriptorCreate(4, 
//                                                                         (int64_t[]){1, c, 1, 1}, 
//                                                                         (int64_t[]){c, 1, c, c}, 
//                                                                         4, 
//                                                                         CUDNN_DATA_FLOAT, 
//                                                                         7);
//
//   cudnnBackendDescriptor_t normDesc = normalizationDescriptorCreate(CUDNN_BATCH_NORM, 
//                                                                     CUDNN_NORM_FWD_INFERENCE,
//                                                                     xDesc, 
//                                                                     yDesc, 
//                                                                     runningMeanDesc, 
//                                                                     runningVarianceDesc, 
//                                                                     savingMeanDesc, 
//                                                                     savingInvVarDesc, 
//                                                                     scaleDesc, 
//                                                                     biasDesc, 
//                                                                     epsilonDesc, 
//                                                                     momentumDesc);
//
//   cudnnBackendDescriptor_t opGraph = createOperationGraph(handle, normDesc);
//
//   // エンジン設定
//   cudnnBackendDescriptor_t engine = createEngine(opGraph);
//
//   // エンジン設定の作成
//   cudnnBackendDescriptor_t engCfg = createEngineConfig(engine);
//
//   // 実行プランの作成
//   cudnnBackendDescriptor_t plan = createExecutionPlan(handle, engCfg);
//
//
//   int64_t workspaceSize;
//   CHECK_CUDNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
//                                         CUDNN_TYPE_INT64, 1, NULL, &workspaceSize));
//
//
//   return 0;
// }
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDNN(expression)                                      \
{                                                                    \
    cudnnStatus_t status = (expression);                             \
    if (status != CUDNN_STATUS_SUCCESS) {                            \
        fprintf(stderr, "Error on line %d: %s\n", __LINE__,          \
                cudnnGetErrorString(status));                        \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

#define CHECK_CUDA(expression)                                       \
{                                                                    \
    cudaError_t status = (expression);                               \
    if (status != cudaSuccess) {                                     \
        fprintf(stderr, "Error on line %d: %s\n", __LINE__,          \
                cudaGetErrorString(status));                         \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

int main() {
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    // Define tensor dimensions
    const int n = 32;    // batch size
    const int c = 64;    // number of channels
    const int h = 224;   // height
    const int w = 224;   // width

    // Create tensor descriptors
    cudnnBackendDescriptor_t xDesc, yDesc, scaleBiasDesc, runningMeanDesc, runningVarDesc;

    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc));
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc));
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &scaleBiasDesc));
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &runningMeanDesc));
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &runningVarDesc));

    // Set tensor attributes
    int64_t xDim[] = {n, c, h, w};
    int64_t xStr[] = {c * h * w, h * w, w, 1};
    int64_t bnDim[] = {1, c, 1, 1};
    int64_t bnStr[] = {c, 1, 1, 1};

    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    int64_t alignment = 4; // assuming float alignment
    int64_t id = 1;

    // Helper function to set tensor attributes
    void setTensorAttributes(cudnnBackendDescriptor_t desc, int64_t* dims, int64_t* strides, int dim_size) {
        CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
        CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dim_size, dims));
        CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, dim_size, strides));
        CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id));
        CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
        CHECK_CUDNN(cudnnBackendFinalize(desc));
        id++;
    }

    setTensorAttributes(xDesc, xDim, xStr, 4);
    setTensorAttributes(yDesc, xDim, xStr, 4);
    setTensorAttributes(scaleBiasDesc, bnDim, bnStr, 4);
    setTensorAttributes(runningMeanDesc, bnDim, bnStr, 4);
    setTensorAttributes(runningVarDesc, bnDim, bnStr, 4);

    // Create normalization operation descriptor
    cudnnBackendDescriptor_t normDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, &normDesc));

    // Set normalization attributes
    cudnnBackendNormMode_t mode = CUDNN_BATCH_NORM;
    cudnnBackendNormFwdPhase_t phase = CUDNN_NORM_FWD_TRAINING;
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, CUDNN_TYPE_NORM_MODE, 1, &mode));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scaleBiasDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scaleBiasDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningMeanDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningVarDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningMeanDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningVarDesc));

    // Set epsilon and exponential average factor
    cudnnBackendDescriptor_t epsilonDesc, expAvgFactorDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &epsilonDesc));
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &expAvgFactorDesc));

    float epsilon = 1e-5f;
    float expAvgFactor = 0.1f;
    int64_t scalarDim[] = {1, 1, 1, 1};
    int64_t scalarStr[] = {1, 1, 1, 1};

    setTensorAttributes(epsilonDesc, scalarDim, scalarStr, 4);
    setTensorAttributes(expAvgFactorDesc, scalarDim, scalarStr, 4);

    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &epsilonDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &expAvgFactorDesc));

    CHECK_CUDNN(cudnnBackendFinalize(normDesc));

    // Create operation graph
    cudnnBackendDescriptor_t opGraph;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));
    CHECK_CUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &normDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
    CHECK_CUDNN(cudnnBackendFinalize(opGraph));

    // Create engine configuration
    cudnnBackendDescriptor_t engine, engineConfig;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engineConfig));

    int64_t globalIdx = 0;
    CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph));
    CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &globalIdx));
    CHECK_CUDNN(cudnnBackendFinalize(engine));

    CHECK_CUDNN(cudnnBackendSetAttribute(engineConfig, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine));
    CHECK_CUDNN(cudnnBackendFinalize(engineConfig));

    // Create execution plan
    cudnnBackendDescriptor_t plan;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
    CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engineConfig));
    CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
    CHECK_CUDNN(cudnnBackendFinalize(plan));

    // Allocate memory and set up variant pack
    float *x, *y, *scale, *bias, *runningMean, *runningVar;
    CHECK_CUDA(cudaMalloc((void**)&x, n * c * h * w * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&y, n * c * h * w * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&scale, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&bias, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&runningMean, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&runningVar, c * sizeof(float)));

    void* devPtrs[] = {x, y, scale, bias, runningMean, runningVar, &epsilon, &expAvgFactor};
    int64_t uids[] = {1, 2, 3, 4, 5, 6, 7, 8};  // Matching the ids set earlier

    cudnnBackendDescriptor_t variantPack;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variantPack));
    CHECK_CUDNN(cudnnBackendSetAttribute(variantPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 8, devPtrs));
    CHECK_CUDNN(cudnnBackendSetAttribute(variantPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 8, uids));
    CHECK_CUDNN(cudnnBackendFinalize(variantPack));

    // Execute the batch normalization
    CHECK_CUDNN(cudnnBackendExecute(handle, plan, variantPack));

    // Clean up
    cudaFree(x); cudaFree(y); cudaFree(scale); cudaFree(bias); cudaFree(runningMean); cudaFree(runningVar);
    cudnnBackendDestroyDescriptor(xDesc);
    cudnnBackendDestroyDescriptor(yDesc);
    cudnnBackendDestroyDescriptor(scaleBiasDesc);
    cudnnBackendDestroyDescriptor(runningMeanDesc);
    cudnnBackendDestroyDescriptor(runningVarDesc);
    cudnnBackendDestroyDescriptor(epsilonDesc);
    cudnnBackendDestroyDescriptor(expAvgFactorDesc);
    cudnnBackendDestroyDescriptor(normDesc);
    cudnnBackendDestroyDescriptor(opGraph);
    cudnnBackendDestroyDescriptor(engine);
    cudnnBackendDestroyDescriptor(engineConfig);
    cudnnBackendDestroyDescriptor(plan);
    cudnnBackendDestroyDescriptor(variantPack);
    cudnnDestroy(handle);

    printf("Batch Normalization completed successfully.\n");

    return 0;
}
