cmake_minimum_required(VERSION 3.10)
project(MyProject CUDA C CXX)

# -Wall -Wextra -Werror
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(CUDA REQUIRED)
find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include)
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    message(STATUS "Found cuDNN: include path: ${CUDNN_INCLUDE_DIR}, library: ${CUDNN_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find cuDNN")
endif()

include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
add_library(utils STATIC utils.cpp)

# add_executable(graph_api_conv_2d graph_api_conv_2d.c)
# target_link_libraries(graph_api_conv_2d PRIVATE utils)
# target_link_libraries(graph_api_conv_2d PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})
#
# # graph_api_batch_norm_2d.c
# include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
# add_executable(graph_api_batch_norm_2d graph_api_batch_norm_2d.c)
# target_link_libraries(graph_api_batch_norm_2d PRIVATE utils)
# target_link_libraries(graph_api_batch_norm_2d PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})

include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
add_executable(batch_norm batch_norm.cpp)
target_link_libraries(batch_norm PRIVATE utils)
target_link_libraries(batch_norm PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})

include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
add_executable(layer_norm layer_norm.cpp)
target_link_libraries(layer_norm PRIVATE utils)
target_link_libraries(layer_norm PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})

include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
add_executable(dropout_4d dropout_4d.c)
target_link_libraries(dropout_4d PRIVATE utils)
target_link_libraries(dropout_4d PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})

include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
add_executable(dropout_2d dropout_2d.c)
target_link_libraries(dropout_2d PRIVATE utils)
target_link_libraries(dropout_2d PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})

include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
add_executable(rnn rnn.cu)
target_link_libraries(rnn PRIVATE utils)
target_link_libraries(rnn PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})
