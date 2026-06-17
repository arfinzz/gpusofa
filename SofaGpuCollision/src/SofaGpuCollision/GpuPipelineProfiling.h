#pragma once

#include <SofaGpuCollision/config.h>

#include <cstdint>

namespace SofaGpuCollision::profiling
{

struct StageSnapshot
{
    double wallMilliseconds { 0.0 };
    double gpuKernelMilliseconds { 0.0 };
    double hostPreparationMilliseconds { 0.0 };
    double sofaTriangleExtractionMilliseconds { 0.0 };
    double backendTrianglePackMilliseconds { 0.0 };
    double hostToDeviceMilliseconds { 0.0 };
    double deviceAllocationMilliseconds { 0.0 };
    double denseGridClearMilliseconds { 0.0 };
    double denseGridCounterClearMilliseconds { 0.0 };
    double denseGridTissueAabbMilliseconds { 0.0 };
    double denseGridToolAabbMilliseconds { 0.0 };
    double denseGridInsertTissueMilliseconds { 0.0 };
    double denseGridInsertToolMilliseconds { 0.0 };
    double denseGridGeneratePairsMilliseconds { 0.0 };
    double denseGridCandidateReadbackMilliseconds { 0.0 };
    double denseGridSortUniqueMilliseconds { 0.0 };
    double denseGridSortUniqueHostMilliseconds { 0.0 };
    double denseGridExactContactMilliseconds { 0.0 };
    double denseGridContactCountReadbackMilliseconds { 0.0 };
    double denseGridContactDownloadMilliseconds { 0.0 };
    double sofaDetectionOutputPublishMilliseconds { 0.0 };
    double hashGridResetMilliseconds { 0.0 };
    double hashGridPairHashClearMilliseconds { 0.0 };
    double hashGridInsertTissueMilliseconds { 0.0 };
    double hashGridInsertToolMilliseconds { 0.0 };
    double hashGridPairCountMilliseconds { 0.0 };
    double hashGridScanMilliseconds { 0.0 };
    double hashGridGeneratePairsMilliseconds { 0.0 };
    double hashGridProximityCounterClearMilliseconds { 0.0 };
    // FBP-specific (2026-05-24 profiling reshape)
    double featureBasedProximityKernelMilliseconds { 0.0 };
    double hostSynchronizationMilliseconds { 0.0 };  // derived: max(0, wall - kernel) when finalized
    std::uint64_t hostToDeviceBytes { 0 };
    std::uint64_t deviceToHostBytes { 0 };
    std::uint64_t deviceAllocationBytes { 0 };
    std::uint32_t kernelLaunchCount { 0 };
    std::uint32_t cudaMemsetCount { 0 };
    std::uint32_t workspaceResizeCount { 0 };
    std::uint32_t inputPrimitiveCount { 0 };
    std::uint32_t outputPairCount { 0 };
    std::uint32_t outputCandidateCount { 0 };
    std::uint32_t outputContactCount { 0 };
    std::uint32_t rawCandidateCount { 0 };
    std::uint32_t uniqueCandidateCount { 0 };
    std::uint32_t gridCellCount { 0 };
    std::uint32_t activeMixedCellCount { 0 };
    std::uint32_t tissueInsertCount { 0 };
    std::uint32_t toolInsertCount { 0 };
    std::uint32_t maxTissueCellOccupancy { 0 };
    std::uint32_t maxToolCellOccupancy { 0 };
    std::uint32_t overflowCount { 0 };
    std::uint32_t hashDedupeProbeOverflowCount { 0 };
    std::uint32_t vfContactCount { 0 };
    std::uint32_t fvContactCount { 0 };
    std::uint32_t eeContactCount { 0 };
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
    std::uint64_t narrowOutputContactCount { 0 };
};

SOFA_GPU_COLLISION_API void beginStep();
SOFA_GPU_COLLISION_API StepSnapshot finishStep();
SOFA_GPU_COLLISION_API AggregateSnapshot aggregateSnapshot();
SOFA_GPU_COLLISION_API void recordBroadPhase(const StageSnapshot& stage);
SOFA_GPU_COLLISION_API void recordNarrowPhase(const StageSnapshot& stage);

} // namespace SofaGpuCollision::profiling
