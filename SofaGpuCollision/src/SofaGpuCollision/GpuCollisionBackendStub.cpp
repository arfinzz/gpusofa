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

bool computeDenseGridTriangleContacts(
    const std::vector<TrianglePrimitive>&,
    const std::vector<TrianglePrimitive>&,
    const DenseGridConfig&,
    std::vector<ExactContact>&,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }
    diagnostic = "GPU dense-grid triangle collision is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool computeDenseGridIndexedTriangleContacts(
    const TriangleIndexedSurface&,
    const TriangleIndexedSurface&,
    const DenseGridConfig&,
    std::vector<ExactContact>&,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }
    diagnostic = "GPU dense-grid indexed triangle collision is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool computeFeatureBasedProximityContacts(
    const TriangleIndexedSurface&,
    const TriangleIndexedSurface&,
    const DenseGridConfig&,
    const FeatureBasedProximityConfig&,
    std::vector<ProximityContact>&,
    FeatureBasedProximityStats* proximityStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }
    if (proximityStats != nullptr)
    {
        *proximityStats = FeatureBasedProximityStats {};
    }
    diagnostic = "GPU feature-based proximity is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool computeFeatureBasedVertexTriangleContacts(
    const PointCloudSurface&,
    const TriangleIndexedSurface&,
    const DenseGridConfig&,
    const FeatureBasedProximityConfig&,
    std::vector<ProximityContact>&,
    FeatureBasedProximityStats* proximityStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }
    if (proximityStats != nullptr)
    {
        *proximityStats = FeatureBasedProximityStats {};
    }
    diagnostic = "GPU feature-based vertex-triangle proximity is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool computeHashPrefixSumProximityContacts(
    const TriangleIndexedSurface&,
    const TriangleIndexedSurface&,
    const DenseGridConfig&,
    const HashPrefixSumConfig&,
    const FeatureBasedProximityConfig&,
    std::vector<ProximityContact>&,
    FeatureBasedProximityStats* proximityStats,
    HashPrefixSumStats* hashStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }
    if (proximityStats != nullptr)
    {
        *proximityStats = FeatureBasedProximityStats {};
    }
    if (hashStats != nullptr)
    {
        *hashStats = HashPrefixSumStats {};
    }
    diagnostic = "GPU hash + prefix-sum proximity is unavailable because the plugin was built without CUDA support.";
    return false;
}

} // namespace SofaGpuCollision::backend
