#include "utils.h"

#include <cudnn_graph_v9.h>
#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cudnn_graph.h>

cudnnBackendDescriptor_t normalizationDescriptorCreate(cudnnBackendNormMode_t mode,
                                                       cudnnBackendNormFwdPhase_t phase,
                                                       cudnnBackendDescriptor_t xDesc,
                                                       cudnnBackendDescriptor_t yDesc,
                                                       cudnnBackendDescriptor_t runningMeanDesc,
                                                       cudnnBackendDescriptor_t runningVarianceDesc,
                                                       cudnnBackendDescriptor_t saveMeanDesc,
                                                       cudnnBackendDescriptor_t saveInvVarianceDesc,
                                                       cudnnBackendDescriptor_t scaleDesc,
                                                       cudnnBackendDescriptor_t biasDesc,
                                                       cudnnBackendDescriptor_t epsilonDesc,
                                                       cudnnBackendDescriptor_t momentumDesc ) {
  cudnnBackendDescriptor_t normDesc;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, &normDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, 
                                       CUDNN_TYPE_NORM_MODE, 1, &mode));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, 
                                       CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, 
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, 
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &saveMeanDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, 
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &saveInvVarianceDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &epsilonDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &momentumDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningMeanDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningVarianceDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningMeanDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &runningVarianceDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scaleDesc));

  CHECK_CUDNN(cudnnBackendSetAttribute(normDesc, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &biasDesc));

  CHECK_CUDNN(cudnnBackendFinalize(normDesc));
  return normDesc;
}

cudnnBackendDescriptor_t createOperationGraph(cudnnHandle_t handle, cudnnBackendDescriptor_t bnDesc) {
  cudnnBackendDescriptor_t opGraph;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));
  CHECK_CUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bnDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                      CUDNN_TYPE_HANDLE, 1, &handle));
  CHECK_CUDNN(cudnnBackendFinalize(opGraph));
  return opGraph;
}

cudnnBackendDescriptor_t createEngine(cudnnBackendDescriptor_t opGraph) {
  cudnnBackendDescriptor_t engine;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
  CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph));
  int64_t gidx = 0;
  CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                      CUDNN_TYPE_INT64, 1, &gidx));
  CHECK_CUDNN(cudnnBackendFinalize(engine));
  return engine;
}

cudnnBackendDescriptor_t createEngineConfig(cudnnBackendDescriptor_t engine) {
  cudnnBackendDescriptor_t engCfg;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engCfg));
  CHECK_CUDNN(cudnnBackendSetAttribute(engCfg, CUDNN_ATTR_ENGINECFG_ENGINE,
                                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine));
  CHECK_CUDNN(cudnnBackendFinalize(engCfg));
  return engCfg;
}

cudnnBackendDescriptor_t createExecutionPlan(cudnnHandle_t handle, cudnnBackendDescriptor_t engCfg) {
  cudnnBackendDescriptor_t plan;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
  CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, 
                                      CUDNN_TYPE_HANDLE, 1, &handle));
  CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engCfg));
  CHECK_CUDNN(cudnnBackendFinalize(plan));
  return plan;
}

cudnnBackendDescriptor_t createVariantPack(void** devPtrs, int64_t* uids, int numPtrs, void* workspace) {
  cudnnBackendDescriptor_t varPack;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varPack));
  CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                      CUDNN_TYPE_VOID_PTR, numPtrs, devPtrs));
  CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                      CUDNN_TYPE_INT64, numPtrs, uids));
  CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                      CUDNN_TYPE_VOID_PTR, 1, &workspace));
  CHECK_CUDNN(cudnnBackendFinalize(varPack));
  return varPack;
}

int main() {
  cudnnHandle_t handle;
  CHECK_CUDNN(cudnnCreate(&handle));

  int64_t n = 2;
  int64_t c = 3;
  int64_t h = 5;
  int64_t w = 5;

  cudnnBackendDescriptor_t xDesc = tensorDescriptorCreate(4, 
                                                          (int64_t[]){n, c, h, w}, 
                                                          (int64_t[]){c*h*w, h*w, w, 1}, 
                                                          4, 
                                                          CUDNN_DATA_FLOAT, 
                                                          'x');

  cudnnBackendDescriptor_t yDesc = tensorDescriptorCreate(4, 
                                                          (int64_t[]){n, c, h, w}, 
                                                          (int64_t[]){c*h*w, h*w, w, 1}, 
                                                          4, 
                                                          CUDNN_DATA_FLOAT, 
                                                          'y');

  cudnnBackendDescriptor_t runningMeanDesc = tensorDescriptorCreate(4, 
                                                                    (int64_t[]){1, c, 1, 1}, 
                                                                    (int64_t[]){c, 1, c, c}, 
                                                                    4, 
                                                                    CUDNN_DATA_FLOAT, 
                                                                    0);

  cudnnBackendDescriptor_t runningVarianceDesc = tensorDescriptorCreate(4, 
                                                                        (int64_t[]){1, c, 1, 1}, 
                                                                        (int64_t[]){c, 1, c, c}, 
                                                                        4, 
                                                                        CUDNN_DATA_FLOAT, 
                                                                        1);

  cudnnBackendDescriptor_t scaleDesc = tensorDescriptorCreate(4, 
                                                              (int64_t[]){1, c, 1, 1}, 
                                                              (int64_t[]){c, 1, c, c}, 
                                                              4, 
                                                              CUDNN_DATA_FLOAT, 
                                                              2);

  cudnnBackendDescriptor_t biasDesc = tensorDescriptorCreate(4, 
                                                            (int64_t[]){1, c, 1, 1}, 
                                                            (int64_t[]){c, 1, c, c}, 
                                                            4, 
                                                            CUDNN_DATA_FLOAT, 
                                                            3);

  cudnnBackendDescriptor_t epsilonDesc = tensorDescriptorCreate(4, 
                                                                (int64_t[]){1, 1, 1, 1}, 
                                                                (int64_t[]){1, 1, 1, 1}, 
                                                                4, 
                                                                CUDNN_DATA_FLOAT, 
                                                                4);

  cudnnBackendDescriptor_t momentumDesc = tensorDescriptorCreate(4, 
                                                                (int64_t[]){1, 1, 1, 1}, 
                                                                (int64_t[]){1, 1, 1, 1}, 
                                                                4, 
                                                                CUDNN_DATA_FLOAT, 
                                                                5);

  cudnnBackendDescriptor_t savingMeanDesc = tensorDescriptorCreate(4, 
                                                                    (int64_t[]){1, c, 1, 1}, 
                                                                    (int64_t[]){c, 1, c, c}, 
                                                                    4, 
                                                                    CUDNN_DATA_FLOAT, 
                                                                    6);

  cudnnBackendDescriptor_t savingInvVarDesc = tensorDescriptorCreate(4, 
                                                                        (int64_t[]){1, c, 1, 1}, 
                                                                        (int64_t[]){c, 1, c, c}, 
                                                                        4, 
                                                                        CUDNN_DATA_FLOAT, 
                                                                        7);

  cudnnBackendDescriptor_t normDesc = normalizationDescriptorCreate(CUDNN_BATCH_NORM, 
                                                                    CUDNN_NORM_FWD_INFERENCE,
                                                                    xDesc, 
                                                                    yDesc, 
                                                                    runningMeanDesc, 
                                                                    runningVarianceDesc, 
                                                                    savingMeanDesc, 
                                                                    savingInvVarDesc, 
                                                                    scaleDesc, 
                                                                    biasDesc, 
                                                                    epsilonDesc, 
                                                                    momentumDesc);

  cudnnBackendDescriptor_t opGraph = createOperationGraph(handle, normDesc);

  // エンジン設定
  cudnnBackendDescriptor_t engine = createEngine(opGraph);

  // エンジン設定の作成
  cudnnBackendDescriptor_t engCfg = createEngineConfig(engine);

  // 実行プランの作成
  cudnnBackendDescriptor_t plan = createExecutionPlan(handle, engCfg);


  int64_t workspaceSize;
  CHECK_CUDNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                        CUDNN_TYPE_INT64, 1, NULL, &workspaceSize));

  return 0;
}
