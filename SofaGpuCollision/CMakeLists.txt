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

# Tier 1 contact-consumer components. They include SofaCUDA headers directly
# (CudaTypes / CudaTriangleModel), so they only build in the CUDA configuration;
# the CPU-only build keeps the stub backend and simply omits them.
set(CUDA_COMPONENT_HEADER_FILES
    src/SofaGpuCollision/CudaContactPenaltyForceField.h
    src/SofaGpuCollision/GpuCollisionPipeline.h
    src/SofaGpuCollision/GpuContactConstraintSolver.h
    src/SofaGpuCollision/GpuResidencyChecker.h
    src/SofaGpuCollision/GpuTissueSolver.h
)
set(CUDA_COMPONENT_SOURCE_FILES
    src/SofaGpuCollision/CudaContactPenaltyForceField.cpp
    src/SofaGpuCollision/GpuCollisionPipeline.cpp
    src/SofaGpuCollision/GpuContactConstraintSolver.cpp
    src/SofaGpuCollision/GpuResidencyChecker.cpp
    src/SofaGpuCollision/GpuTissueSolver.cpp
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
    # GpuContactConstraintSolver: SOFA's Lagrangian constraint code (its CPU
    # comparison mode runs SOFA's own contact constraint and Gauss-Seidel) and
    # Eigen (a double LDL^T for the correction reference).
    find_package(Sofa.Simulation.Core REQUIRED)
    find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)
    find_package(Sofa.Component.Constraint.Lagrangian.Model REQUIRED)
    find_package(Eigen3 REQUIRED NO_MODULE)
    # Resizing a SofaCUDA vector instantiates its OpenGL-buffer code (GLEW); Sofa.GL brings GLEW.
    find_package(Sofa.GL REQUIRED)
    list(REMOVE_ITEM SOURCE_FILES src/SofaGpuCollision/GpuCollisionBackendStub.cpp)
    list(APPEND HEADER_FILES ${CUDA_COMPONENT_HEADER_FILES})
    list(APPEND SOURCE_FILES ${CUDA_COMPONENT_SOURCE_FILES})
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
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver ${SOFACUDA_LIBRARY}
        Sofa.Simulation.Core
        Sofa.Component.Constraint.Lagrangian.Solver
        Sofa.Component.Constraint.Lagrangian.Model
        Sofa.GL
        Eigen3::Eigen)
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

    # GPU contact constraints vs SOFA's own CPU constraint solver (C1) and a
    # double-precision CPU factorisation (C2), on identical inputs.
    add_executable(SofaGpuCollisionConstraintChecks
        src/tools/ConstraintChecks.cpp
    )
    target_compile_features(SofaGpuCollisionConstraintChecks PRIVATE cxx_std_17)
    target_include_directories(SofaGpuCollisionConstraintChecks
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(SofaGpuCollisionConstraintChecks
        PRIVATE
            ${PROJECT_NAME}
            Sofa.Component.Constraint.Lagrangian.Solver
            Sofa.Component.Constraint.Lagrangian.Model
            Eigen3::Eigen
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
