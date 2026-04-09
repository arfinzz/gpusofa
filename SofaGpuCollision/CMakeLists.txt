cmake_minimum_required(VERSION 3.22)

project(SofaGpuCollision VERSION 0.1.0 LANGUAGES CXX)

option(SOFAGPUCOLLISION_ENABLE_CUDA "Build the experimental CUDA backend." OFF)

find_package(Sofa.Framework REQUIRED)
find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
find_package(Sofa.Component.Collision.Geometry REQUIRED)

set(HEADER_FILES
    src/SofaGpuCollision/config.h
    src/SofaGpuCollision/GpuCollisionBackend.h
    src/SofaGpuCollision/GpuCollisionBroadPhase.h
    src/SofaGpuCollision/GpuCollisionNarrowPhase.h
)

set(SOURCE_FILES
    src/SofaGpuCollision/init.cpp
    src/SofaGpuCollision/GpuCollisionBackendStub.cpp
    src/SofaGpuCollision/GpuCollisionBroadPhase.cpp
    src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp
)

if(SOFAGPUCOLLISION_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(REMOVE_ITEM SOURCE_FILES src/SofaGpuCollision/GpuCollisionBackendStub.cpp)
    list(APPEND SOURCE_FILES src/SofaGpuCollision/cuda/GpuCollisionBackend.cu)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PRIVATE SOFA_BUILD_SOFAGPUCOLLISION)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Sofa.Core
        Sofa.Component.Collision.Detection.Algorithm
        Sofa.Component.Collision.Detection.Intersection
        Sofa.Component.Collision.Geometry
)

if(SOFAGPUCOLLISION_ENABLE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SOFAGPUCOLLISION_WITH_CUDA=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif()

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/SofaGpuCollision/
    DESTINATION include/SofaGpuCollision
    FILES_MATCHING
    PATTERN "*.h"
)
