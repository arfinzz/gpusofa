#pragma once

#include <SofaGpuCollision/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace SofaGpuCollision
{

class SOFA_GPU_COLLISION_API GpuPipelineBenchmarkController
    : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(GpuPipelineBenchmarkController, sofa::core::objectmodel::BaseObject);

    GpuPipelineBenchmarkController();
    ~GpuPipelineBenchmarkController() override;

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;

private:
    using Clock = std::chrono::steady_clock;
    using DataBool = sofa::core::objectmodel::Data<bool>;
    using DataInt = sofa::core::objectmodel::Data<int>;
    using DataString = sofa::core::objectmodel::Data<std::string>;

    struct PendingRow
    {
        int step { 0 };
        double stepSeconds { 0.0 };
        int includedInStats { 0 };
        double broadWallMs { 0.0 };
        double broadKernelMs { 0.0 };
        double narrowWallMs { 0.0 };
        double narrowKernelMs { 0.0 };
        double narrowHostPreparationMs { 0.0 };
        double narrowSofaTriangleExtractionMs { 0.0 };
        double narrowBackendTrianglePackMs { 0.0 };
        double narrowHostToDeviceMs { 0.0 };
        double narrowDeviceAllocationMs { 0.0 };
        double narrowClearGridMs { 0.0 };
        double narrowCounterClearMs { 0.0 };
        double narrowTissueAabbMs { 0.0 };
        double narrowToolAabbMs { 0.0 };
        double narrowInsertTissueMs { 0.0 };
        double narrowInsertToolMs { 0.0 };
        double narrowGeneratePairsMs { 0.0 };
        double narrowCandidateReadbackMs { 0.0 };
        double narrowSortUniqueMs { 0.0 };
        double narrowSortUniqueHostMs { 0.0 };
        double narrowExactContactMs { 0.0 };
        double narrowContactCountReadbackMs { 0.0 };
        double narrowContactDownloadMs { 0.0 };
        double narrowSofaOutputPublishMs { 0.0 };
        double narrowHashResetMs { 0.0 };
        double narrowHashPairHashClearMs { 0.0 };
        double narrowHashInsertTissueMs { 0.0 };
        double narrowHashInsertToolMs { 0.0 };
        double narrowHashPairCountMs { 0.0 };
        double narrowHashScanMs { 0.0 };
        double narrowHashGeneratePairsMs { 0.0 };
        double narrowHashProximityCounterClearMs { 0.0 };
        std::uint64_t hostToDeviceBytes { 0 };
        std::uint64_t deviceToHostBytes { 0 };
        std::uint64_t deviceAllocationBytes { 0 };
        std::uint32_t kernelLaunchCount { 0 };
        std::uint32_t cudaMemsetCount { 0 };
        std::uint32_t workspaceResizeCount { 0 };
        int broadGpuUsed { 0 };
        int narrowGpuUsed { 0 };
        std::uint32_t broadInputPrimitiveCount { 0 };
        std::uint32_t broadOutputPairCount { 0 };
        std::uint32_t narrowInputPrimitiveCount { 0 };
        std::uint32_t narrowOutputPairCount { 0 };
        std::uint32_t narrowOutputCandidateCount { 0 };
        std::uint32_t narrowOutputContactCount { 0 };
        std::uint32_t narrowRawCandidateCount { 0 };
        std::uint32_t narrowUniqueCandidateCount { 0 };
        std::uint32_t narrowGridCellCount { 0 };
        std::uint32_t narrowActiveMixedCellCount { 0 };
        std::uint32_t narrowTissueInsertCount { 0 };
        std::uint32_t narrowToolInsertCount { 0 };
        std::uint32_t narrowMaxTissueCellOccupancy { 0 };
        std::uint32_t narrowMaxToolCellOccupancy { 0 };
        std::uint32_t narrowOverflowCount { 0 };
        std::uint32_t narrowHashDedupeProbeOverflowCount { 0 };
        // FBP additions (2026-05-24)
        double narrowFeatureBasedProximityKernelMs { 0.0 };
        double narrowHostSynchronizationMs { 0.0 };
        std::uint32_t narrowVfContactCount { 0 };
        std::uint32_t narrowFvContactCount { 0 };
        std::uint32_t narrowEeContactCount { 0 };
    };

    void ensureOutputReady();
    void flushRows();
    void writeSummary();
    void finalize();
    void onAnimateBegin();
    void onAnimateEnd();

    DataString d_label;
    DataString d_outputDir;
    DataString d_pipelinePhase;
    DataString d_tissueSolver;
    DataString d_tissueForceField;
    DataString d_collisionStateTemplate;
    DataString d_collisionMapping;
    DataString d_visualMapping;
    DataString d_notes;
    DataInt d_simVertexCount;
    DataInt d_simElementCount;
    DataInt d_collisionVertexCount;
    DataInt d_collisionElementCount;
    DataInt d_visualVertexCount;
    DataInt d_visualElementCount;
    DataInt d_warmupSteps;
    DataInt d_flushInterval;
    DataInt d_logInterval;
    DataBool d_printProgress;

    int m_currentStep { 0 };
    bool m_initializedOutput { false };
    bool m_finalized { false };
    Clock::time_point m_stepStartTime {};

    int m_measuredCount { 0 };
    double m_measuredTotalSeconds { 0.0 };
    double m_measuredMinSeconds { 0.0 };
    double m_measuredMaxSeconds { 0.0 };
    double m_measuredBroadWallMs { 0.0 };
    double m_measuredBroadKernelMs { 0.0 };
    double m_measuredNarrowWallMs { 0.0 };
    double m_measuredNarrowKernelMs { 0.0 };
    double m_measuredNarrowHostPreparationMs { 0.0 };
    double m_measuredNarrowSofaTriangleExtractionMs { 0.0 };
    double m_measuredNarrowBackendTrianglePackMs { 0.0 };
    double m_measuredNarrowHostToDeviceMs { 0.0 };
    double m_measuredNarrowDeviceAllocationMs { 0.0 };
    double m_measuredNarrowClearGridMs { 0.0 };
    double m_measuredNarrowCounterClearMs { 0.0 };
    double m_measuredNarrowTissueAabbMs { 0.0 };
    double m_measuredNarrowToolAabbMs { 0.0 };
    double m_measuredNarrowInsertTissueMs { 0.0 };
    double m_measuredNarrowInsertToolMs { 0.0 };
    double m_measuredNarrowGeneratePairsMs { 0.0 };
    double m_measuredNarrowCandidateReadbackMs { 0.0 };
    double m_measuredNarrowSortUniqueMs { 0.0 };
    double m_measuredNarrowSortUniqueHostMs { 0.0 };
    double m_measuredNarrowExactContactMs { 0.0 };
    double m_measuredNarrowContactCountReadbackMs { 0.0 };
    double m_measuredNarrowContactDownloadMs { 0.0 };
    double m_measuredNarrowSofaOutputPublishMs { 0.0 };
    double m_measuredNarrowHashResetMs { 0.0 };
    double m_measuredNarrowHashPairHashClearMs { 0.0 };
    double m_measuredNarrowHashInsertTissueMs { 0.0 };
    double m_measuredNarrowHashInsertToolMs { 0.0 };
    double m_measuredNarrowHashPairCountMs { 0.0 };
    double m_measuredNarrowHashScanMs { 0.0 };
    double m_measuredNarrowHashGeneratePairsMs { 0.0 };
    double m_measuredNarrowHashProximityCounterClearMs { 0.0 };
    std::uint64_t m_measuredHostToDeviceBytes { 0 };
    std::uint64_t m_measuredDeviceToHostBytes { 0 };
    std::uint64_t m_measuredDeviceAllocationBytes { 0 };
    std::uint64_t m_measuredKernelLaunchCount { 0 };
    std::uint64_t m_measuredCudaMemsetCount { 0 };
    std::uint64_t m_measuredWorkspaceResizeCount { 0 };
    std::uint64_t m_measuredBroadInputPrimitiveCount { 0 };
    std::uint64_t m_measuredBroadOutputPairCount { 0 };
    std::uint64_t m_measuredNarrowInputPrimitiveCount { 0 };
    std::uint64_t m_measuredNarrowOutputPairCount { 0 };
    std::uint64_t m_measuredNarrowOutputCandidateCount { 0 };
    std::uint64_t m_measuredNarrowOutputContactCount { 0 };
    std::uint64_t m_measuredNarrowRawCandidateCount { 0 };
    std::uint64_t m_measuredNarrowUniqueCandidateCount { 0 };
    std::uint64_t m_measuredNarrowGridCellCount { 0 };
    std::uint64_t m_measuredNarrowActiveMixedCellCount { 0 };
    std::uint64_t m_measuredNarrowTissueInsertCount { 0 };
    std::uint64_t m_measuredNarrowToolInsertCount { 0 };
    std::uint64_t m_measuredNarrowMaxTissueCellOccupancy { 0 };
    std::uint64_t m_measuredNarrowMaxToolCellOccupancy { 0 };
    std::uint64_t m_measuredNarrowOverflowCount { 0 };
    std::uint64_t m_measuredNarrowHashDedupeProbeOverflowCount { 0 };
    // FBP additions (2026-05-24)
    double m_measuredNarrowFeatureBasedProximityKernelMs { 0.0 };
    double m_measuredNarrowHostSynchronizationMs { 0.0 };
    std::uint64_t m_measuredNarrowVfContactCount { 0 };
    std::uint64_t m_measuredNarrowFvContactCount { 0 };
    std::uint64_t m_measuredNarrowEeContactCount { 0 };

    std::string m_csvPath;
    std::string m_summaryPath;
    std::vector<PendingRow> m_pendingRows;
};

} // namespace SofaGpuCollision
