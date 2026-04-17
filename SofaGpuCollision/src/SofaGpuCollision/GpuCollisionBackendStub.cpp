#include <SofaGpuCollision/config.h>
#include <SofaGpuCollision/GpuCollisionBackend.h>

namespace SofaGpuCollision::backend
{

BackendStatus probe()
{
    return {
        false,
        "SofaGpuCollision was built without the CUDA backend enabled."
    };
}

bool computeBroadPhasePairs(
    const std::vector<AxisAlignedBoundingBox>&,
    std::vector<BroadPhaseIndexPair>&,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }
    diagnostic = "GPU broad phase is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool prefilterNarrowPhasePairs(
    const std::vector<NarrowPhaseTreePair>&,
    std::vector<std::uint32_t>&,
    std::vector<NarrowPhaseContactCandidate>*,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }
    diagnostic = "GPU narrow phase is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool computeExactTriangleContacts(
    const std::vector<TrianglePrimitive>&,
    const std::vector<TrianglePrimitive>&,
    std::vector<ExactContact>&,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }
    diagnostic = "GPU exact triangle collision is unavailable because the plugin was built without CUDA support.";
    return false;
}

} // namespace SofaGpuCollision::backend
