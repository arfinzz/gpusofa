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

bool computeSimpleHashProximityContacts(
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
    if (executionStats != nullptr) *executionStats = BackendExecutionStats {};
    if (proximityStats != nullptr) *proximityStats = FeatureBasedProximityStats {};
    if (hashStats != nullptr) *hashStats = HashPrefixSumStats {};
    diagnostic = "GPU simple-hash proximity is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool computeSortedGridProximityContacts(
    const TriangleIndexedSurface&,
    const TriangleIndexedSurface&,
    const DenseGridConfig&,
    const SortedGridConfig&,
    const FeatureBasedProximityConfig&,
    std::vector<ProximityContact>&,
    FeatureBasedProximityStats* proximityStats,
    SortedGridStats* sortedStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr) *executionStats = BackendExecutionStats {};
    if (proximityStats != nullptr) *proximityStats = FeatureBasedProximityStats {};
    if (sortedStats != nullptr) *sortedStats = SortedGridStats {};
    diagnostic = "GPU sorted-grid proximity is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool computeBigCellFusedProximityContacts(
    const TriangleIndexedSurface&,
    const TriangleIndexedSurface&,
    const DenseGridConfig&,
    const BigCellConfig&,
    const FeatureBasedProximityConfig&,
    std::vector<ProximityContact>&,
    FeatureBasedProximityStats* proximityStats,
    BigCellStats* bigStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    if (executionStats != nullptr) *executionStats = BackendExecutionStats {};
    if (proximityStats != nullptr) *proximityStats = FeatureBasedProximityStats {};
    if (bigStats != nullptr) *bigStats = BigCellStats {};
    diagnostic = "GPU big-cell fused proximity is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool validateContactPenaltyForces(
    const ContactPenaltyConfig&,
    const TriangleIndexedSurface&,
    const TriangleIndexedSurface&,
    ContactForceValidation* validation,
    std::string& diagnostic)
{
    if (validation != nullptr) *validation = ContactForceValidation {};
    diagnostic = "Contact-force validation is unavailable because the plugin was built without CUDA support.";
    return false;
}

bool accumulateContactPenaltyForces(
    const ContactPenaltyConfig&,
    std::uint64_t,
    std::uint64_t,
    void*,
    void*,
    const void*,
    const void*,
    ContactPenaltyStats* stats,
    std::string& diagnostic)
{
    if (stats != nullptr) *stats = ContactPenaltyStats {};
    diagnostic = "GPU contact penalty forces are unavailable because the plugin was built without CUDA support.";
    return false;
}

bool accumulateContactPenaltyDForces(
    const ContactPenaltyConfig&,
    std::uint64_t,
    std::uint64_t,
    float,
    void*,
    void*,
    const void*,
    const void*,
    std::string& diagnostic)
{
    diagnostic = "GPU contact penalty dforces are unavailable because the plugin was built without CUDA support.";
    return false;
}

} // namespace SofaGpuCollision::backend
