#pragma once

#if defined(_WIN32)
#    if defined(SOFA_BUILD_SOFAGPUCOLLISION)
#        define SOFA_GPU_COLLISION_API __declspec(dllexport)
#    else
#        define SOFA_GPU_COLLISION_API __declspec(dllimport)
#    endif
#else
#    define SOFA_GPU_COLLISION_API
#endif
