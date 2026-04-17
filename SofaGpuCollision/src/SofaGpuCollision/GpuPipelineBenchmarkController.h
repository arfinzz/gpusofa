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
        std::uint64_t hostToDeviceBytes { 0 };
        std::uint64_t deviceToHostBytes { 0 };
        std::uint64_t deviceAllocationBytes { 0 };
        int broadGpuUsed { 0 };
        int narrowGpuUsed { 0 };
        std::uint32_t broadInputPrimitiveCount { 0 };
        std::uint32_t broadOutputPairCount { 0 };
        std::uint32_t narrowInputPrimitiveCount { 0 };
        std::uint32_t narrowOutputPairCount { 0 };
        std::uint32_t narrowOutputCandidateCount { 0 };
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
    std::uint64_t m_measuredHostToDeviceBytes { 0 };
    std::uint64_t m_measuredDeviceToHostBytes { 0 };
    std::uint64_t m_measuredDeviceAllocationBytes { 0 };
    std::uint64_t m_measuredBroadInputPrimitiveCount { 0 };
    std::uint64_t m_measuredBroadOutputPairCount { 0 };
    std::uint64_t m_measuredNarrowInputPrimitiveCount { 0 };
    std::uint64_t m_measuredNarrowOutputPairCount { 0 };
    std::uint64_t m_measuredNarrowOutputCandidateCount { 0 };

    std::string m_csvPath;
    std::string m_summaryPath;
    std::vector<PendingRow> m_pendingRows;
};

} // namespace SofaGpuCollision
