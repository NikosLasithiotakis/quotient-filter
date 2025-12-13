cmake_minimum_required(VERSION 3.11)
project(quotient-filter VERSION 1.0 LANGUAGES C CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SOURCE_FILES
    qf.c
)

add_library(quotient-filter STATIC ${SOURCE_FILES})
target_include_directories(quotient-filter PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
set_property(TARGET quotient-filter PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(quotient-filter-gpu STATIC qf.c qf_gpu.cu)
target_include_directories(quotient-filter-gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
set_property(TARGET quotient-filter-gpu PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(quotient-filter-gpu PUBLIC CUDA::cudart)
