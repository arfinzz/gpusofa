#include <SofaGpuCollision/GpuPipelineBenchmarkController.h>
#include <SofaGpuCollision/GpuPipelineProfiling.h>

#include <algorithm>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace SofaGpuCollision
{

int GpuPipelineBenchmarkControllerClass = sofa::core::RegisterObject(
    "Writes per-step benchmark CSVs and summaries with GPU collision stage metrics.")
    .add<GpuPipelineBenchmarkController>();

GpuPipelineBenchmarkController::GpuPipelineBenchmarkController()
    : sofa::core::objectmodel::BaseObject()
    , d_label(initData(&d_label, std::string("gpu_benchmark"), "label", "Benchmark label used in output filenames."))
    , d_outputDir(initData(&d_outputDir, std::string("benchmark_logs"), "outputDir", "Directory where CSV and summary files are written."))
    , d_pipelinePhase(initData(&d_pipelinePhase, std::string("unspecified"), "pipelinePhase", "High-level pipeline phase label written to the summary file."))
    , d_tissueSolver(initData(&d_tissueSolver, std::string(""), "tissueSolver", "Optional tissue solver description written to the summary file."))
    , d_tissueForceField(initData(&d_tissueForceField, std::string(""), "tissueForceField", "Optional tissue force-field description written to the summary file."))
    , d_collisionStateTemplate(initData(&d_collisionStateTemplate, std::string(""), "collisionStateTemplate", "Optional collision-state template description written to the summary file."))
    , d_collisionMapping(initData(&d_collisionMapping, std::string(""), "collisionMapping", "Optional collision mapping description written to the summary file."))
    , d_visualMapping(initData(&d_visualMapping, std::string(""), "visualMapping", "Optional visual mapping description written to the summary file."))
    , d_notes(initData(&d_notes, std::string(""), "notes", "Optional free-form benchmark notes written to the summary file."))
    , d_simVertexCount(initData(&d_simVertexCount, 0, "simVertexCount", "Optional simulation vertex count written to the summary file."))
    , d_simElementCount(initData(&d_simElementCount, 0, "simElementCount", "Optional simulation element count written to the summary file."))
    , d_collisionVertexCount(initData(&d_collisionVertexCount, 0, "collisionVertexCount", "Optional collision vertex count written to the summary file."))
    , d_collisionElementCount(initData(&d_collisionElementCount, 0, "collisionElementCount", "Optional collision element count written to the summary file."))
    , d_visualVertexCount(initData(&d_visualVertexCount, 0, "visualVertexCount", "Optional visual vertex count written to the summary file."))
    , d_visualElementCount(initData(&d_visualElementCount, 0, "visualElementCount", "Optional visual element count written to the summary file."))
    , d_warmupSteps(initData(&d_warmupSteps, 50, "warmupSteps", "Number of initial steps excluded from statistics."))
    , d_flushInterval(initData(&d_flushInterval, 50, "flushInterval", "How many rows to buffer before writing to disk."))
    , d_logInterval(initData(&d_logInterval, 200, "logInterval", "Measured-step interval for progress logs."))
    , d_printProgress(initData(&d_printProgress, true, "printProgress", "Emit progress logs while benchmarking."))
{
    f_listening.setValue(true);
}

GpuPipelineBenchmarkController::~GpuPipelineBenchmarkController()
{
    finalize();
}

void GpuPipelineBenchmarkController::init()
{
    sofa::core::objectmodel::BaseObject::init();
    ensureOutputReady();
}

void GpuPipelineBenchmarkController::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent*>(event) != nullptr)
    {
        onAnimateBegin();
        return;
    }

    if (dynamic_cast<sofa::simulation::AnimateEndEvent*>(event) != nullptr)
    {
        onAnimateEnd();
    }
}

void GpuPipelineBenchmarkController::ensureOutputReady()
{
    if (m_initializedOutput)
    {
        return;
    }

    std::filesystem::create_directories(std::filesystem::path(d_outputDir.getValue()));
    m_csvPath = (std::filesystem::path(d_outputDir.getValue()) / (d_label.getValue() + "_timings.csv")).string();
    m_summaryPath = (std::filesystem::path(d_outputDir.getValue()) / (d_label.getValue() + "_summary.txt")).string();

    if (!std::filesystem::exists(m_csvPath) || std::filesystem::file_size(m_csvPath) == 0)
    {
        std::ofstream csv(m_csvPath, std::ios::out | std::ios::trunc);
        csv << "step,duration_seconds,included_in_stats,broad_wall_ms,broad_kernel_ms,narrow_wall_ms,narrow_kernel_ms,"
               "host_to_device_bytes,device_to_host_bytes,device_allocation_bytes,broad_gpu_used,narrow_gpu_used,"
               "broad_input_primitive_count,broad_output_pair_count,narrow_input_primitive_count,"
               "narrow_output_pair_count,narrow_output_candidate_count\n";
    }

    m_initializedOutput = true;
}

void GpuPipelineBenchmarkController::flushRows()
{
    if (m_pendingRows.empty())
    {
        return;
    }

    ensureOutputReady();
    std::ofstream csv(m_csvPath, std::ios::out | std::ios::app);
    csv << std::fixed << std::setprecision(9);

    for (const auto& row : m_pendingRows)
    {
        csv << row.step << ','
            << row.stepSeconds << ','
            << row.includedInStats << ','
            << row.broadWallMs << ','
            << row.broadKernelMs << ','
            << row.narrowWallMs << ','
            << row.narrowKernelMs << ','
            << row.hostToDeviceBytes << ','
            << row.deviceToHostBytes << ','
            << row.deviceAllocationBytes << ','
            << row.broadGpuUsed << ','
            << row.narrowGpuUsed << ','
            << row.broadInputPrimitiveCount << ','
            << row.broadOutputPairCount << ','
            << row.narrowInputPrimitiveCount << ','
            << row.narrowOutputPairCount << ','
            << row.narrowOutputCandidateCount << '\n';
    }

    m_pendingRows.clear();
}

void GpuPipelineBenchmarkController::writeSummary()
{
    ensureOutputReady();
    std::ofstream summary(m_summaryPath, std::ios::out | std::ios::trunc);
    summary << std::fixed << std::setprecision(9);
    summary << "label=" << d_label.getValue() << '\n';
    summary << "pipeline_phase=" << d_pipelinePhase.getValue() << '\n';
    if (!d_tissueSolver.getValue().empty())
    {
        summary << "tissue_solver=" << d_tissueSolver.getValue() << '\n';
    }
    if (!d_tissueForceField.getValue().empty())
    {
        summary << "tissue_force_field=" << d_tissueForceField.getValue() << '\n';
    }
    if (!d_collisionStateTemplate.getValue().empty())
    {
        summary << "collision_state_template=" << d_collisionStateTemplate.getValue() << '\n';
    }
    if (!d_collisionMapping.getValue().empty())
    {
        summary << "collision_mapping=" << d_collisionMapping.getValue() << '\n';
    }
    if (!d_visualMapping.getValue().empty())
    {
        summary << "visual_mapping=" << d_visualMapping.getValue() << '\n';
    }
    if (!d_notes.getValue().empty())
    {
        summary << "notes=" << d_notes.getValue() << '\n';
    }
    if (d_simVertexCount.getValue() > 0)
    {
        summary << "sim_vertex_count=" << d_simVertexCount.getValue() << '\n';
    }
    if (d_simElementCount.getValue() > 0)
    {
        summary << "sim_element_count=" << d_simElementCount.getValue() << '\n';
    }
    if (d_collisionVertexCount.getValue() > 0)
    {
        summary << "collision_vertex_count=" << d_collisionVertexCount.getValue() << '\n';
    }
    if (d_collisionElementCount.getValue() > 0)
    {
        summary << "collision_element_count=" << d_collisionElementCount.getValue() << '\n';
    }
    if (d_visualVertexCount.getValue() > 0)
    {
        summary << "visual_vertex_count=" << d_visualVertexCount.getValue() << '\n';
    }
    if (d_visualElementCount.getValue() > 0)
    {
        summary << "visual_element_count=" << d_visualElementCount.getValue() << '\n';
    }
    summary << "measured_steps=" << m_measuredCount << '\n';
    summary << "warmup_steps=" << d_warmupSteps.getValue() << '\n';

    if (m_measuredCount <= 0)
    {
        return;
    }

    const double avgStepSeconds = m_measuredTotalSeconds / static_cast<double>(m_measuredCount);
    const double avgFps = avgStepSeconds > 0.0 ? 1.0 / avgStepSeconds : 0.0;
    summary << "avg_step_seconds=" << avgStepSeconds << '\n';
    summary << "min_step_seconds=" << m_measuredMinSeconds << '\n';
    summary << "max_step_seconds=" << m_measuredMaxSeconds << '\n';
    summary << "avg_fps=" << avgFps << '\n';
    summary << "avg_broad_wall_ms=" << (m_measuredBroadWallMs / m_measuredCount) << '\n';
    summary << "avg_broad_kernel_ms=" << (m_measuredBroadKernelMs / m_measuredCount) << '\n';
    summary << "avg_narrow_wall_ms=" << (m_measuredNarrowWallMs / m_measuredCount) << '\n';
    summary << "avg_narrow_kernel_ms=" << (m_measuredNarrowKernelMs / m_measuredCount) << '\n';
    summary << "avg_host_to_device_bytes=" << (m_measuredHostToDeviceBytes / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_device_to_host_bytes=" << (m_measuredDeviceToHostBytes / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_device_allocation_bytes=" << (m_measuredDeviceAllocationBytes / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_broad_input_primitive_count=" << (m_measuredBroadInputPrimitiveCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_broad_output_pair_count=" << (m_measuredBroadOutputPairCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_input_primitive_count=" << (m_measuredNarrowInputPrimitiveCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_output_pair_count=" << (m_measuredNarrowOutputPairCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_output_candidate_count=" << (m_measuredNarrowOutputCandidateCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "csv_path=" << m_csvPath << '\n';
}

void GpuPipelineBenchmarkController::finalize()
{
    if (m_finalized)
    {
        return;
    }

    flushRows();
    writeSummary();
    m_finalized = true;
}

void GpuPipelineBenchmarkController::onAnimateBegin()
{
    profiling::beginStep();
    m_stepStartTime = Clock::now();
}

void GpuPipelineBenchmarkController::onAnimateEnd()
{
    const auto stepDuration = std::chrono::duration<double>(Clock::now() - m_stepStartTime).count();
    const auto stepSnapshot = profiling::finishStep();
    const bool included = m_currentStep >= d_warmupSteps.getValue();

    PendingRow row;
    row.step = m_currentStep;
    row.stepSeconds = stepDuration;
    row.includedInStats = included ? 1 : 0;
    row.broadWallMs = stepSnapshot.broadPhase.wallMilliseconds;
    row.broadKernelMs = stepSnapshot.broadPhase.gpuKernelMilliseconds;
    row.narrowWallMs = stepSnapshot.narrowPhase.wallMilliseconds;
    row.narrowKernelMs = stepSnapshot.narrowPhase.gpuKernelMilliseconds;
    row.hostToDeviceBytes =
        stepSnapshot.broadPhase.hostToDeviceBytes + stepSnapshot.narrowPhase.hostToDeviceBytes;
    row.deviceToHostBytes =
        stepSnapshot.broadPhase.deviceToHostBytes + stepSnapshot.narrowPhase.deviceToHostBytes;
    row.deviceAllocationBytes =
        stepSnapshot.broadPhase.deviceAllocationBytes + stepSnapshot.narrowPhase.deviceAllocationBytes;
    row.broadGpuUsed = stepSnapshot.broadPhase.gpuUsed ? 1 : 0;
    row.narrowGpuUsed = stepSnapshot.narrowPhase.gpuUsed ? 1 : 0;
    row.broadInputPrimitiveCount = stepSnapshot.broadPhase.inputPrimitiveCount;
    row.broadOutputPairCount = stepSnapshot.broadPhase.outputPairCount;
    row.narrowInputPrimitiveCount = stepSnapshot.narrowPhase.inputPrimitiveCount;
    row.narrowOutputPairCount = stepSnapshot.narrowPhase.outputPairCount;
    row.narrowOutputCandidateCount = stepSnapshot.narrowPhase.outputCandidateCount;
    m_pendingRows.push_back(row);

    if (included)
    {
        ++m_measuredCount;
        m_measuredTotalSeconds += stepDuration;
        m_measuredMinSeconds = (m_measuredCount == 1) ? stepDuration : std::min(m_measuredMinSeconds, stepDuration);
        m_measuredMaxSeconds = (m_measuredCount == 1) ? stepDuration : std::max(m_measuredMaxSeconds, stepDuration);
        m_measuredBroadWallMs += row.broadWallMs;
        m_measuredBroadKernelMs += row.broadKernelMs;
        m_measuredNarrowWallMs += row.narrowWallMs;
        m_measuredNarrowKernelMs += row.narrowKernelMs;
        m_measuredHostToDeviceBytes += row.hostToDeviceBytes;
        m_measuredDeviceToHostBytes += row.deviceToHostBytes;
        m_measuredDeviceAllocationBytes += row.deviceAllocationBytes;
        m_measuredBroadInputPrimitiveCount += row.broadInputPrimitiveCount;
        m_measuredBroadOutputPairCount += row.broadOutputPairCount;
        m_measuredNarrowInputPrimitiveCount += row.narrowInputPrimitiveCount;
        m_measuredNarrowOutputPairCount += row.narrowOutputPairCount;
        m_measuredNarrowOutputCandidateCount += row.narrowOutputCandidateCount;

        if (d_printProgress.getValue() && d_logInterval.getValue() > 0 && (m_measuredCount % d_logInterval.getValue()) == 0)
        {
            const double avgStepSeconds = m_measuredTotalSeconds / static_cast<double>(m_measuredCount);
            const double avgFps = avgStepSeconds > 0.0 ? 1.0 / avgStepSeconds : 0.0;
            msg_info() << "[GpuPipelineBenchmarkController] " << d_label.getValue()
                       << ": measured_steps=" << m_measuredCount
                       << ", avg_step=" << avgStepSeconds << "s"
                       << ", avg_fps=" << avgFps
                       << ", avg_h2d_bytes=" << (m_measuredHostToDeviceBytes / static_cast<std::uint64_t>(m_measuredCount))
                       << ", avg_d2h_bytes=" << (m_measuredDeviceToHostBytes / static_cast<std::uint64_t>(m_measuredCount));
        }
    }

    if (d_flushInterval.getValue() > 0 && static_cast<int>(m_pendingRows.size()) >= d_flushInterval.getValue())
    {
        flushRows();
        writeSummary();
    }

    ++m_currentStep;
}

} // namespace SofaGpuCollision
