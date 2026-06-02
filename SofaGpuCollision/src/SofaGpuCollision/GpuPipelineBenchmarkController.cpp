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
    , d_outputDir(initData(&d_outputDir, std::string("output/benchmark_logs"), "outputDir", "Directory where CSV and summary files are written."))
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
               "narrow_host_preparation_ms,narrow_sofa_triangle_extraction_ms,narrow_backend_triangle_pack_ms,"
               "narrow_h2d_ms,narrow_device_allocation_ms,narrow_clear_grid_ms,narrow_counter_clear_ms,"
               "narrow_tissue_aabb_ms,narrow_tool_aabb_ms,narrow_insert_tissue_ms,narrow_insert_tool_ms,"
               "narrow_generate_pairs_ms,narrow_candidate_readback_ms,narrow_sort_unique_ms,narrow_exact_contact_ms,"
               "narrow_sort_unique_host_ms,narrow_contact_count_readback_ms,narrow_contact_download_ms,"
               "narrow_sofa_output_publish_ms,"
               "host_to_device_bytes,device_to_host_bytes,device_allocation_bytes,kernel_launch_count,cuda_memset_count,"
               "workspace_resize_count,broad_gpu_used,narrow_gpu_used,"
               "broad_input_primitive_count,broad_output_pair_count,narrow_input_primitive_count,"
               "narrow_output_pair_count,narrow_output_candidate_count,narrow_output_contact_count,narrow_raw_candidate_count,"
               "narrow_unique_candidate_count,narrow_duplicate_reduction_ratio,narrow_grid_cell_count,narrow_active_mixed_cell_count,"
               "narrow_tissue_insert_count,narrow_tool_insert_count,narrow_max_tissue_cell_occupancy,"
               "narrow_max_tool_cell_occupancy,narrow_overflow_count,narrow_hash_dedupe_probe_overflow_count,"
               "narrow_feature_based_proximity_kernel_ms,narrow_host_synchronization_ms,"
               "narrow_vf_contact_count,narrow_fv_contact_count,narrow_ee_contact_count\n";
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
            << row.narrowHostPreparationMs << ','
            << row.narrowSofaTriangleExtractionMs << ','
            << row.narrowBackendTrianglePackMs << ','
            << row.narrowHostToDeviceMs << ','
            << row.narrowDeviceAllocationMs << ','
            << row.narrowClearGridMs << ','
            << row.narrowCounterClearMs << ','
            << row.narrowTissueAabbMs << ','
            << row.narrowToolAabbMs << ','
            << row.narrowInsertTissueMs << ','
            << row.narrowInsertToolMs << ','
            << row.narrowGeneratePairsMs << ','
            << row.narrowCandidateReadbackMs << ','
            << row.narrowSortUniqueMs << ','
            << row.narrowExactContactMs << ','
            << row.narrowSortUniqueHostMs << ','
            << row.narrowContactCountReadbackMs << ','
            << row.narrowContactDownloadMs << ','
            << row.narrowSofaOutputPublishMs << ','
            << row.hostToDeviceBytes << ','
            << row.deviceToHostBytes << ','
            << row.deviceAllocationBytes << ','
            << row.kernelLaunchCount << ','
            << row.cudaMemsetCount << ','
            << row.workspaceResizeCount << ','
            << row.broadGpuUsed << ','
            << row.narrowGpuUsed << ','
            << row.broadInputPrimitiveCount << ','
            << row.broadOutputPairCount << ','
            << row.narrowInputPrimitiveCount << ','
            << row.narrowOutputPairCount << ','
            << row.narrowOutputCandidateCount << ','
            << row.narrowOutputContactCount << ','
            << row.narrowRawCandidateCount << ','
            << row.narrowUniqueCandidateCount << ','
            << (row.narrowRawCandidateCount > 0
                    ? 1.0 - static_cast<double>(row.narrowUniqueCandidateCount) / static_cast<double>(row.narrowRawCandidateCount)
                    : 0.0) << ','
            << row.narrowGridCellCount << ','
            << row.narrowActiveMixedCellCount << ','
            << row.narrowTissueInsertCount << ','
            << row.narrowToolInsertCount << ','
            << row.narrowMaxTissueCellOccupancy << ','
            << row.narrowMaxToolCellOccupancy << ','
            << row.narrowOverflowCount << ','
            << row.narrowHashDedupeProbeOverflowCount << ','
            << row.narrowFeatureBasedProximityKernelMs << ','
            << row.narrowHostSynchronizationMs << ','
            << row.narrowVfContactCount << ','
            << row.narrowFvContactCount << ','
            << row.narrowEeContactCount << '\n';
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
    summary << "avg_narrow_host_preparation_ms=" << (m_measuredNarrowHostPreparationMs / m_measuredCount) << '\n';
    summary << "avg_narrow_sofa_triangle_extraction_ms=" << (m_measuredNarrowSofaTriangleExtractionMs / m_measuredCount) << '\n';
    summary << "avg_narrow_backend_triangle_pack_ms=" << (m_measuredNarrowBackendTrianglePackMs / m_measuredCount) << '\n';
    summary << "avg_narrow_h2d_ms=" << (m_measuredNarrowHostToDeviceMs / m_measuredCount) << '\n';
    summary << "avg_narrow_device_allocation_ms=" << (m_measuredNarrowDeviceAllocationMs / m_measuredCount) << '\n';
    summary << "avg_narrow_clear_grid_ms=" << (m_measuredNarrowClearGridMs / m_measuredCount) << '\n';
    summary << "avg_narrow_counter_clear_ms=" << (m_measuredNarrowCounterClearMs / m_measuredCount) << '\n';
    summary << "avg_narrow_tissue_aabb_ms=" << (m_measuredNarrowTissueAabbMs / m_measuredCount) << '\n';
    summary << "avg_narrow_tool_aabb_ms=" << (m_measuredNarrowToolAabbMs / m_measuredCount) << '\n';
    summary << "avg_narrow_insert_tissue_ms=" << (m_measuredNarrowInsertTissueMs / m_measuredCount) << '\n';
    summary << "avg_narrow_insert_tool_ms=" << (m_measuredNarrowInsertToolMs / m_measuredCount) << '\n';
    summary << "avg_narrow_generate_pairs_ms=" << (m_measuredNarrowGeneratePairsMs / m_measuredCount) << '\n';
    summary << "avg_narrow_candidate_readback_ms=" << (m_measuredNarrowCandidateReadbackMs / m_measuredCount) << '\n';
    summary << "avg_narrow_sort_unique_ms=" << (m_measuredNarrowSortUniqueMs / m_measuredCount) << '\n';
    summary << "avg_narrow_sort_unique_host_ms=" << (m_measuredNarrowSortUniqueHostMs / m_measuredCount) << '\n';
    summary << "avg_narrow_exact_contact_ms=" << (m_measuredNarrowExactContactMs / m_measuredCount) << '\n';
    summary << "avg_narrow_contact_count_readback_ms=" << (m_measuredNarrowContactCountReadbackMs / m_measuredCount) << '\n';
    summary << "avg_narrow_contact_download_ms=" << (m_measuredNarrowContactDownloadMs / m_measuredCount) << '\n';
    summary << "avg_narrow_sofa_output_publish_ms=" << (m_measuredNarrowSofaOutputPublishMs / m_measuredCount) << '\n';
    summary << "avg_host_to_device_bytes=" << (m_measuredHostToDeviceBytes / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_device_to_host_bytes=" << (m_measuredDeviceToHostBytes / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_device_allocation_bytes=" << (m_measuredDeviceAllocationBytes / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_kernel_launch_count=" << (static_cast<double>(m_measuredKernelLaunchCount) / m_measuredCount) << '\n';
    summary << "avg_cuda_memset_count=" << (static_cast<double>(m_measuredCudaMemsetCount) / m_measuredCount) << '\n';
    summary << "avg_workspace_resize_count=" << (static_cast<double>(m_measuredWorkspaceResizeCount) / m_measuredCount) << '\n';
    summary << "avg_broad_input_primitive_count=" << (m_measuredBroadInputPrimitiveCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_broad_output_pair_count=" << (m_measuredBroadOutputPairCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_input_primitive_count=" << (m_measuredNarrowInputPrimitiveCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_output_pair_count=" << (m_measuredNarrowOutputPairCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_output_candidate_count=" << (m_measuredNarrowOutputCandidateCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_output_contact_count=" << (m_measuredNarrowOutputContactCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_raw_candidate_count=" << (m_measuredNarrowRawCandidateCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_unique_candidate_count=" << (m_measuredNarrowUniqueCandidateCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_duplicate_reduction_ratio="
            << (m_measuredNarrowRawCandidateCount > 0
                    ? 1.0 - static_cast<double>(m_measuredNarrowUniqueCandidateCount) / static_cast<double>(m_measuredNarrowRawCandidateCount)
                    : 0.0) << '\n';
    summary << "avg_narrow_grid_cell_count=" << (m_measuredNarrowGridCellCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_active_mixed_cell_count=" << (m_measuredNarrowActiveMixedCellCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_tissue_insert_count=" << (m_measuredNarrowTissueInsertCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_tool_insert_count=" << (m_measuredNarrowToolInsertCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_max_tissue_cell_occupancy=" << (m_measuredNarrowMaxTissueCellOccupancy / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_max_tool_cell_occupancy=" << (m_measuredNarrowMaxToolCellOccupancy / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_overflow_count=" << (m_measuredNarrowOverflowCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_hash_dedupe_probe_overflow_count=" << (m_measuredNarrowHashDedupeProbeOverflowCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_feature_based_proximity_kernel_ms=" << (m_measuredNarrowFeatureBasedProximityKernelMs / m_measuredCount) << '\n';
    summary << "avg_narrow_host_synchronization_ms=" << (m_measuredNarrowHostSynchronizationMs / m_measuredCount) << '\n';
    summary << "avg_narrow_vf_contact_count=" << (m_measuredNarrowVfContactCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_fv_contact_count=" << (m_measuredNarrowFvContactCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
    summary << "avg_narrow_ee_contact_count=" << (m_measuredNarrowEeContactCount / static_cast<std::uint64_t>(m_measuredCount)) << '\n';
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
    row.narrowHostPreparationMs = stepSnapshot.narrowPhase.hostPreparationMilliseconds;
    row.narrowSofaTriangleExtractionMs = stepSnapshot.narrowPhase.sofaTriangleExtractionMilliseconds;
    row.narrowBackendTrianglePackMs = stepSnapshot.narrowPhase.backendTrianglePackMilliseconds;
    row.narrowHostToDeviceMs = stepSnapshot.narrowPhase.hostToDeviceMilliseconds;
    row.narrowDeviceAllocationMs = stepSnapshot.narrowPhase.deviceAllocationMilliseconds;
    row.narrowClearGridMs = stepSnapshot.narrowPhase.denseGridClearMilliseconds;
    row.narrowCounterClearMs = stepSnapshot.narrowPhase.denseGridCounterClearMilliseconds;
    row.narrowTissueAabbMs = stepSnapshot.narrowPhase.denseGridTissueAabbMilliseconds;
    row.narrowToolAabbMs = stepSnapshot.narrowPhase.denseGridToolAabbMilliseconds;
    row.narrowInsertTissueMs = stepSnapshot.narrowPhase.denseGridInsertTissueMilliseconds;
    row.narrowInsertToolMs = stepSnapshot.narrowPhase.denseGridInsertToolMilliseconds;
    row.narrowGeneratePairsMs = stepSnapshot.narrowPhase.denseGridGeneratePairsMilliseconds;
    row.narrowCandidateReadbackMs = stepSnapshot.narrowPhase.denseGridCandidateReadbackMilliseconds;
    row.narrowSortUniqueMs = stepSnapshot.narrowPhase.denseGridSortUniqueMilliseconds;
    row.narrowSortUniqueHostMs = stepSnapshot.narrowPhase.denseGridSortUniqueHostMilliseconds;
    row.narrowExactContactMs = stepSnapshot.narrowPhase.denseGridExactContactMilliseconds;
    row.narrowContactCountReadbackMs = stepSnapshot.narrowPhase.denseGridContactCountReadbackMilliseconds;
    row.narrowContactDownloadMs = stepSnapshot.narrowPhase.denseGridContactDownloadMilliseconds;
    row.narrowSofaOutputPublishMs = stepSnapshot.narrowPhase.sofaDetectionOutputPublishMilliseconds;
    row.hostToDeviceBytes =
        stepSnapshot.broadPhase.hostToDeviceBytes + stepSnapshot.narrowPhase.hostToDeviceBytes;
    row.deviceToHostBytes =
        stepSnapshot.broadPhase.deviceToHostBytes + stepSnapshot.narrowPhase.deviceToHostBytes;
    row.deviceAllocationBytes =
        stepSnapshot.broadPhase.deviceAllocationBytes + stepSnapshot.narrowPhase.deviceAllocationBytes;
    row.kernelLaunchCount = stepSnapshot.broadPhase.kernelLaunchCount + stepSnapshot.narrowPhase.kernelLaunchCount;
    row.cudaMemsetCount = stepSnapshot.broadPhase.cudaMemsetCount + stepSnapshot.narrowPhase.cudaMemsetCount;
    row.workspaceResizeCount = stepSnapshot.broadPhase.workspaceResizeCount + stepSnapshot.narrowPhase.workspaceResizeCount;
    row.broadGpuUsed = stepSnapshot.broadPhase.gpuUsed ? 1 : 0;
    row.narrowGpuUsed = stepSnapshot.narrowPhase.gpuUsed ? 1 : 0;
    row.broadInputPrimitiveCount = stepSnapshot.broadPhase.inputPrimitiveCount;
    row.broadOutputPairCount = stepSnapshot.broadPhase.outputPairCount;
    row.narrowInputPrimitiveCount = stepSnapshot.narrowPhase.inputPrimitiveCount;
    row.narrowOutputPairCount = stepSnapshot.narrowPhase.outputPairCount;
    row.narrowOutputCandidateCount = stepSnapshot.narrowPhase.outputCandidateCount;
    row.narrowOutputContactCount = stepSnapshot.narrowPhase.outputContactCount;
    row.narrowRawCandidateCount = stepSnapshot.narrowPhase.rawCandidateCount;
    row.narrowUniqueCandidateCount = stepSnapshot.narrowPhase.uniqueCandidateCount;
    row.narrowGridCellCount = stepSnapshot.narrowPhase.gridCellCount;
    row.narrowActiveMixedCellCount = stepSnapshot.narrowPhase.activeMixedCellCount;
    row.narrowTissueInsertCount = stepSnapshot.narrowPhase.tissueInsertCount;
    row.narrowToolInsertCount = stepSnapshot.narrowPhase.toolInsertCount;
    row.narrowMaxTissueCellOccupancy = stepSnapshot.narrowPhase.maxTissueCellOccupancy;
    row.narrowMaxToolCellOccupancy = stepSnapshot.narrowPhase.maxToolCellOccupancy;
    row.narrowOverflowCount = stepSnapshot.narrowPhase.overflowCount;
    row.narrowHashDedupeProbeOverflowCount = stepSnapshot.narrowPhase.hashDedupeProbeOverflowCount;
    row.narrowFeatureBasedProximityKernelMs = stepSnapshot.narrowPhase.featureBasedProximityKernelMilliseconds;
    row.narrowHostSynchronizationMs = stepSnapshot.narrowPhase.hostSynchronizationMilliseconds;
    row.narrowVfContactCount = stepSnapshot.narrowPhase.vfContactCount;
    row.narrowFvContactCount = stepSnapshot.narrowPhase.fvContactCount;
    row.narrowEeContactCount = stepSnapshot.narrowPhase.eeContactCount;
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
        m_measuredNarrowHostPreparationMs += row.narrowHostPreparationMs;
        m_measuredNarrowSofaTriangleExtractionMs += row.narrowSofaTriangleExtractionMs;
        m_measuredNarrowBackendTrianglePackMs += row.narrowBackendTrianglePackMs;
        m_measuredNarrowHostToDeviceMs += row.narrowHostToDeviceMs;
        m_measuredNarrowDeviceAllocationMs += row.narrowDeviceAllocationMs;
        m_measuredNarrowClearGridMs += row.narrowClearGridMs;
        m_measuredNarrowCounterClearMs += row.narrowCounterClearMs;
        m_measuredNarrowTissueAabbMs += row.narrowTissueAabbMs;
        m_measuredNarrowToolAabbMs += row.narrowToolAabbMs;
        m_measuredNarrowInsertTissueMs += row.narrowInsertTissueMs;
        m_measuredNarrowInsertToolMs += row.narrowInsertToolMs;
        m_measuredNarrowGeneratePairsMs += row.narrowGeneratePairsMs;
        m_measuredNarrowCandidateReadbackMs += row.narrowCandidateReadbackMs;
        m_measuredNarrowSortUniqueMs += row.narrowSortUniqueMs;
        m_measuredNarrowSortUniqueHostMs += row.narrowSortUniqueHostMs;
        m_measuredNarrowExactContactMs += row.narrowExactContactMs;
        m_measuredNarrowContactCountReadbackMs += row.narrowContactCountReadbackMs;
        m_measuredNarrowContactDownloadMs += row.narrowContactDownloadMs;
        m_measuredNarrowSofaOutputPublishMs += row.narrowSofaOutputPublishMs;
        m_measuredHostToDeviceBytes += row.hostToDeviceBytes;
        m_measuredDeviceToHostBytes += row.deviceToHostBytes;
        m_measuredDeviceAllocationBytes += row.deviceAllocationBytes;
        m_measuredKernelLaunchCount += row.kernelLaunchCount;
        m_measuredCudaMemsetCount += row.cudaMemsetCount;
        m_measuredWorkspaceResizeCount += row.workspaceResizeCount;
        m_measuredBroadInputPrimitiveCount += row.broadInputPrimitiveCount;
        m_measuredBroadOutputPairCount += row.broadOutputPairCount;
        m_measuredNarrowInputPrimitiveCount += row.narrowInputPrimitiveCount;
        m_measuredNarrowOutputPairCount += row.narrowOutputPairCount;
        m_measuredNarrowOutputCandidateCount += row.narrowOutputCandidateCount;
        m_measuredNarrowOutputContactCount += row.narrowOutputContactCount;
        m_measuredNarrowRawCandidateCount += row.narrowRawCandidateCount;
        m_measuredNarrowUniqueCandidateCount += row.narrowUniqueCandidateCount;
        m_measuredNarrowGridCellCount += row.narrowGridCellCount;
        m_measuredNarrowActiveMixedCellCount += row.narrowActiveMixedCellCount;
        m_measuredNarrowTissueInsertCount += row.narrowTissueInsertCount;
        m_measuredNarrowToolInsertCount += row.narrowToolInsertCount;
        m_measuredNarrowMaxTissueCellOccupancy += row.narrowMaxTissueCellOccupancy;
        m_measuredNarrowMaxToolCellOccupancy += row.narrowMaxToolCellOccupancy;
        m_measuredNarrowOverflowCount += row.narrowOverflowCount;
        m_measuredNarrowHashDedupeProbeOverflowCount += row.narrowHashDedupeProbeOverflowCount;
        m_measuredNarrowFeatureBasedProximityKernelMs += row.narrowFeatureBasedProximityKernelMs;
        m_measuredNarrowHostSynchronizationMs += row.narrowHostSynchronizationMs;
        m_measuredNarrowVfContactCount += row.narrowVfContactCount;
        m_measuredNarrowFvContactCount += row.narrowFvContactCount;
        m_measuredNarrowEeContactCount += row.narrowEeContactCount;

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
