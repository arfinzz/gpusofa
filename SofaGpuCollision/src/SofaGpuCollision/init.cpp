#include <SofaGpuCollision/config.h>

#include <sofa/core/ObjectFactory.h>

extern "C"
{

SOFA_GPU_COLLISION_API void initExternalModule()
{
}

SOFA_GPU_COLLISION_API const char* getModuleName()
{
    return "SofaGpuCollision";
}

SOFA_GPU_COLLISION_API const char* getModuleVersion()
{
    return "0.1.0";
}

SOFA_GPU_COLLISION_API const char* getModuleLicense()
{
    return "LGPL-2.1-or-later";
}

SOFA_GPU_COLLISION_API const char* getModuleDescription()
{
    return "GPU-first collision phases with native profiling and deterministic rigid-path helpers for SOFA.";
}

SOFA_GPU_COLLISION_API const char* getModuleComponentList()
{
    return "GpuCollisionBroadPhase,GpuCollisionNarrowPhase,GpuKinematicRigidController,GpuPipelineBenchmarkController";
}

} // extern "C"
