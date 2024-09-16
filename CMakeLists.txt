cmake_minimum_required(VERSION 3.11)
project(quotient-filter VERSION 1.0 LANGUAGES C)

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
