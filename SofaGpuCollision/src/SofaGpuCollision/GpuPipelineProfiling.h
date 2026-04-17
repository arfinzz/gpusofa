#pragma once

#include <SofaGpuCollision/config.h>

#include <cstdint>

namespace SofaGpuCollision::profiling
{

struct StageSnapshot
{
    double wallMilliseconds { 0.0 };
    double gpuKernelMilliseconds { 0.0 };
    std::uint64_t hostToDeviceBytes { 0 };
    std::uint64_t deviceToHostBytes { 0 };
    std::uint64_t deviceAllocationBytes { 0 };
    std::uint32_t kernelLaunchCount { 0 };
    std::uint32_t inputPrimitiveCount { 0 };
    std::uint32_t outputPairCount { 0 };
    std::uint32_t outputCandidateCount { 0 };
    bool gpuUsed { false };
    bool cpuFallbackUsed { false };
};

struct StepSnapshot
{
    StageSnapshot broadPhase;
    StageSnapshot narrowPhase;
};

struct AggregateSnapshot
{
    std::uint64_t completedSteps { 0 };
    double broadWallMilliseconds { 0.0 };
    double broadGpuKernelMilliseconds { 0.0 };
    double narrowWallMilliseconds { 0.0 };
    double narrowGpuKernelMilliseconds { 0.0 };
    std::uint64_t broadHostToDeviceBytes { 0 };
    std::uint64_t broadDeviceToHostBytes { 0 };
    std::uint64_t broadDeviceAllocationBytes { 0 };
    std::uint64_t narrowHostToDeviceBytes { 0 };
    std::uint64_t narrowDeviceToHostBytes { 0 };
    std::uint64_t narrowDeviceAllocationBytes { 0 };
    std::uint64_t broadGpuStepCount { 0 };
    std::uint64_t narrowGpuStepCount { 0 };
    std::uint64_t broadCpuFallbackCount { 0 };
    std::uint64_t narrowCpuFallbackCount { 0 };
    std::uint64_t broadInputPrimitiveCount { 0 };
    std::uint64_t broadOutputPairCount { 0 };
    std::uint64_t broadOutputCandidateCount { 0 };
    std::uint64_t narrowInputPrimitiveCount { 0 };
    std::uint64_t narrowOutputPairCount { 0 };
    std::uint64_t narrowOutputCandidateCount { 0 };
};

SOFA_GPU_COLLISION_API void beginStep();
SOFA_GPU_COLLISION_API StepSnapshot finishStep();
SOFA_GPU_COLLISION_API AggregateSnapshot aggregateSnapshot();
SOFA_GPU_COLLISION_API void recordBroadPhase(const StageSnapshot& stage);
SOFA_GPU_COLLISION_API void recordNarrowPhase(const StageSnapshot& stage);

} // namespace SofaGpuCollision::profiling
