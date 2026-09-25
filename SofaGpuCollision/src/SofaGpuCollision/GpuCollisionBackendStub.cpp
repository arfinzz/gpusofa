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

void clearRecordedContactHandles()
{
}

void beginContactFrame()
{
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

bool validateContactSideAwareness(ContactSideValidation* validation, std::string& diagnostic)
{
    if (validation != nullptr) *validation = ContactSideValidation {};
    diagnostic = "Contact side validation is unavailable because the plugin was built without CUDA support.";
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
    const void*,
    const void*,
    std::string& diagnostic)
{
    diagnostic = "GPU contact penalty dforces are unavailable because the plugin was built without CUDA support.";
    return false;
}

namespace
{
const char* const kNoCudaConstraints = "GPU contact constraints are unavailable because the plugin was built without CUDA support.";
}

struct ConstraintWorkspace
{
};

ConstraintWorkspace* createConstraintWorkspace(std::string& diagnostic)
{
    diagnostic = kNoCudaConstraints;
    return nullptr;
}

void destroyConstraintWorkspace(ConstraintWorkspace* workspace)
{
    delete workspace;
}

bool setRigidSystem(ConstraintWorkspace*, const double[36], std::string& diagnostic)
{
    diagnostic = kNoCudaConstraints;
    return false;
}

bool setAdditionalRigidSystems(ConstraintWorkspace*, const std::vector<double>&, const std::vector<double>&, std::string& diagnostic)
{
    diagnostic = kNoCudaConstraints;
    return false;
}

bool additionalRigidResults(const ConstraintWorkspace*, std::vector<double>& corrections, std::vector<double>& impulses)
{
    corrections.clear();
    impulses.clear();
    return false;
}

bool buildContactConstraints(ConstraintWorkspace*, const ConstraintBuildInput&, ConstraintBuildStats* stats,
                             ConstraintTimings*, std::string& diagnostic)
{
    if (stats != nullptr) *stats = ConstraintBuildStats {};
    diagnostic = kNoCudaConstraints;
    return false;
}

bool factorizeDeformableSystem(ConstraintWorkspace*, const HostCsrMatrix&, ConstraintTimings*, std::string& diagnostic)
{
    diagnostic = kNoCudaConstraints;
    return false;
}

bool assembleContactCompliance(ConstraintWorkspace*, double, double, ConstraintTimings*, std::string& diagnostic)
{
    diagnostic = kNoCudaConstraints;
    return false;
}

bool solveContactConstraints(ConstraintWorkspace*, const ConstraintSolveConfig&, ConstraintSolveStats* stats,
                             ConstraintTimings*, std::string& diagnostic)
{
    if (stats != nullptr) *stats = ConstraintSolveStats {};
    diagnostic = kNoCudaConstraints;
    return false;
}

bool computeContactCorrection(ConstraintWorkspace*, std::vector<float>& deformableCorrection, double rigidCorrection[6],
                              ConstraintImpulse* impulse, ConstraintTimings*, std::string& diagnostic)
{
    deformableCorrection.clear();
    for (int e = 0; e < 6; ++e) rigidCorrection[e] = 0.0;
    if (impulse != nullptr) *impulse = ConstraintImpulse {};
    diagnostic = kNoCudaConstraints;
    return false;
}

bool downloadContactProblem(ConstraintWorkspace*, bool, ConstraintProblemSnapshot& snapshot, std::string& diagnostic)
{
    snapshot = ConstraintProblemSnapshot {};
    diagnostic = kNoCudaConstraints;
    return false;
}

bool solveFrictionProblemOnGpu(int, int, const std::vector<double>&, const std::vector<double>&, double,
                               const ConstraintSolveConfig&, std::vector<double>& lambda, ConstraintSolveStats* stats,
                               std::string& diagnostic)
{
    lambda.clear();
    if (stats != nullptr) *stats = ConstraintSolveStats {};
    diagnostic = kNoCudaConstraints;
    return false;
}

bool computeDenseComplianceOnGpu(const HostCsrMatrix&, const std::vector<int>&, std::vector<double>& compliance,
                                 ConstraintTimings*, std::string& diagnostic)
{
    compliance.clear();
    diagnostic = kNoCudaConstraints;
    return false;
}

bool useTissueFactor(ConstraintWorkspace*, TissueWorkspace*, std::string& diagnostic)
{
    diagnostic = kNoCudaConstraints;
    return false;
}

bool computeContactCorrectionOnDevice(ConstraintWorkspace*, double rigidCorrection[6], ConstraintImpulse* impulse,
                                      ConstraintTimings*, std::string& diagnostic)
{
    for (int e = 0; e < 6; ++e) rigidCorrection[e] = 0.0;
    if (impulse != nullptr) *impulse = ConstraintImpulse {};
    diagnostic = kNoCudaConstraints;
    return false;
}

bool applyContactCorrectionOnDevice(ConstraintWorkspace*, const DeviceCorrectionTarget&, std::string& diagnostic)
{
    diagnostic = kNoCudaConstraints;
    return false;
}

bool downloadDeformableCorrection(ConstraintWorkspace*, std::vector<float>& correction, std::string& diagnostic)
{
    correction.clear();
    diagnostic = kNoCudaConstraints;
    return false;
}

namespace
{
const char* const kNoCudaTissue = "The GPU tissue solver is unavailable because the plugin was built without CUDA support.";
}

struct TissueWorkspace
{
};

TissueWorkspace* createTissueWorkspace(const TissueSetup&, std::string& diagnostic)
{
    diagnostic = kNoCudaTissue;
    return nullptr;
}

void destroyTissueWorkspace(TissueWorkspace* workspace)
{
    delete workspace;
}

bool tissueFreeMotion(TissueWorkspace*, const void*, const void*, void*, void*, const TissueStepConfig&,
                      TissueTimings* timings, std::string& diagnostic)
{
    if (timings != nullptr) *timings = TissueTimings {};
    diagnostic = kNoCudaTissue;
    return false;
}

bool setTissueExternalForces(TissueWorkspace*, const std::vector<double>&, std::string& diagnostic)
{
    diagnostic = kNoCudaTissue;
    return false;
}

bool updateTissueElements(TissueWorkspace*, const TissueElementUpdate&, std::string& diagnostic)
{
    diagnostic = kNoCudaTissue;
    return false;
}

int tissueLuFallbackSteps(const TissueWorkspace*)
{
    return 0;
}

int tissueDofCount(const TissueWorkspace*)
{
    return 0;
}

bool tissueFactorReady(const TissueWorkspace*)
{
    return false;
}

int tissueBandwidth(const TissueWorkspace*)
{
    return 0;
}

bool tissueSolveInPlace(TissueWorkspace*, float*, int, std::string& diagnostic)
{
    diagnostic = kNoCudaTissue;
    return false;
}

bool tissueComplianceBlock(TissueWorkspace*, const int*, int, float*, std::string& diagnostic)
{
    diagnostic = kNoCudaTissue;
    return false;
}

bool tissueMonitor(TissueWorkspace*, const void*, int, double& minVolumeRatio, double position[3], std::string& diagnostic)
{
    minVolumeRatio = 0.0;
    for (int c = 0; c < 3; ++c) position[c] = 0.0;
    diagnostic = kNoCudaTissue;
    return false;
}

bool downloadTissueStep(TissueWorkspace*, bool, TissueStepSnapshot& snapshot, std::string& diagnostic)
{
    snapshot = TissueStepSnapshot {};
    diagnostic = kNoCudaTissue;
    return false;
}

bool rigidMappingApply(int, const double[9], const double[3], const void*, void*, void*, std::string& diagnostic)
{
    diagnostic = kNoCudaTissue;
    return false;
}

bool rigidMappingApplyJ(int, const double[3], const double[3], const void*, void*, bool, std::string& diagnostic)
{
    diagnostic = kNoCudaTissue;
    return false;
}

bool rigidMappingApplyJT(int, const void*, const void*, double forceAndTorque[6], std::string& diagnostic)
{
    for (int k = 0; k < 6; ++k) forceAndTorque[k] = 0.0;
    diagnostic = kNoCudaTissue;
    return false;
}

} // namespace SofaGpuCollision::backend
