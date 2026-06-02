cmake_minimum_required(VERSION 3.22)

project(SofaGpuCollision VERSION 0.1.0 LANGUAGES CXX)

option(SOFAGPUCOLLISION_ENABLE_CUDA "Build the experimental CUDA backend." OFF)

find_package(Sofa.Framework REQUIRED)
find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
find_package(Sofa.Component.Collision.Geometry REQUIRED)
find_package(Sofa.Component.Controller REQUIRED)
find_package(Sofa.Component.StateContainer REQUIRED)

set(HEADER_FILES
    src/SofaGpuCollision/config.h
    src/SofaGpuCollision/GpuCollisionBackend.h
    src/SofaGpuCollision/GpuCollisionBroadPhase.h
    src/SofaGpuCollision/GpuCollisionNarrowPhase.h
    src/SofaGpuCollision/GpuKinematicRigidController.h
    src/SofaGpuCollision/GpuPipelineBenchmarkController.h
    src/SofaGpuCollision/GpuPipelineProfiling.h
)

set(SOURCE_FILES
    src/SofaGpuCollision/init.cpp
    src/SofaGpuCollision/GpuCollisionBackendStub.cpp
    src/SofaGpuCollision/GpuCollisionBroadPhase.cpp
    src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp
    src/SofaGpuCollision/GpuKinematicRigidController.cpp
    src/SofaGpuCollision/GpuPipelineBenchmarkController.cpp
    src/SofaGpuCollision/GpuPipelineProfiling.cpp
)

if(SOFAGPUCOLLISION_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_path(SOFACUDA_INCLUDE_DIR
        NAMES SofaCUDA/component/collision/geometry/CudaTriangleModel.h
        HINTS
            ${CMAKE_PREFIX_PATH}/plugins/SofaCUDA/include
            /opt/sofa/install/v25.12/plugins/SofaCUDA/include
            /home/arfin/sofa/install/v25.12/plugins/SofaCUDA/include
    )
    find_path(SOFAGL_INCLUDE_DIR
        NAMES sofa/gl/gl.h
        HINTS
            ${CMAKE_PREFIX_PATH}/include/Sofa.GL
            /opt/sofa/install/v25.12/include/Sofa.GL
            /home/arfin/sofa/install/v25.12/include/Sofa.GL
    )
    find_path(SOFA_COMPONENT_MASS_INCLUDE_DIR
        NAMES sofa/component/mass/MassType.h
        HINTS
            ${CMAKE_PREFIX_PATH}/include/Sofa.Component.Mass
            /opt/sofa/install/v25.12/include/Sofa.Component.Mass
            /home/arfin/sofa/install/v25.12/include/Sofa.Component.Mass
    )
    find_library(SOFACUDA_LIBRARY
        NAMES SofaCUDA libSofaCUDA.so
        HINTS
            ${CMAKE_PREFIX_PATH}/plugins/SofaCUDA/lib
            /opt/sofa/install/v25.12/plugins/SofaCUDA/lib
            /home/arfin/sofa/install/v25.12/plugins/SofaCUDA/lib
    )
    find_library(NVTOOLSEXT_LIBRARY
        NAMES nvToolsExt
    )
    if(NOT SOFACUDA_INCLUDE_DIR OR NOT SOFACUDA_LIBRARY OR NOT SOFAGL_INCLUDE_DIR OR NOT SOFA_COMPONENT_MASS_INCLUDE_DIR)
        message(FATAL_ERROR "SOFAGPUCOLLISION_ENABLE_CUDA requires the installed SofaCUDA headers and library.")
    endif()
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
        Sofa.Component.Controller
        Sofa.Component.StateContainer
)

if(SOFAGPUCOLLISION_ENABLE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SOFAGPUCOLLISION_WITH_CUDA=1)
    target_include_directories(${PROJECT_NAME} PRIVATE ${SOFACUDA_INCLUDE_DIR} ${SOFAGL_INCLUDE_DIR} ${SOFA_COMPONENT_MASS_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart ${SOFACUDA_LIBRARY})
    if(NVTOOLSEXT_LIBRARY)
        target_compile_definitions(${PROJECT_NAME} PRIVATE SOFAGPUCOLLISION_WITH_NVTX=1)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${NVTOOLSEXT_LIBRARY})
    endif()

    add_executable(SofaGpuCollisionDenseGridBackendBench
        src/tools/DenseGridBackendBench.cpp
    )
    target_compile_features(SofaGpuCollisionDenseGridBackendBench PRIVATE cxx_std_17)
    target_include_directories(SofaGpuCollisionDenseGridBackendBench
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(SofaGpuCollisionDenseGridBackendBench
        PRIVATE
            ${PROJECT_NAME}
    )
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
