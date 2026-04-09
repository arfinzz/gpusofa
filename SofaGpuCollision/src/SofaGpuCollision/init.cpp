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
    return "GPU-first broad phase and narrow phase wrappers for SOFA collision detection.";
}

SOFA_GPU_COLLISION_API const char* getModuleComponentList()
{
    return "GpuCollisionBroadPhase,GpuCollisionNarrowPhase";
}

} // extern "C"
