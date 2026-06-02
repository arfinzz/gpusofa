#include <SofaGpuCollision/GpuPipelineProfiling.h>

#include <mutex>

namespace SofaGpuCollision::profiling
{

namespace
{

struct RuntimeState
{
    StepSnapshot currentStep;
    AggregateSnapshot aggregate;
    std::mutex mutex;
};

RuntimeState& runtimeState()
{
    static RuntimeState state;
    return state;
}

void accumulateStage(
    const StageSnapshot& stage,
    double& wallTotal,
    double& kernelTotal,
    std::uint64_t& h2dTotal,
    std::uint64_t& d2hTotal,
    std::uint64_t& allocTotal,
    std::uint64_t& gpuStepCount,
    std::uint64_t& cpuFallbackCount,
    std::uint64_t& inputPrimitiveTotal,
    std::uint64_t& outputPairTotal,
    std::uint64_t& outputCandidateTotal,
    std::uint64_t* outputContactTotal = nullptr)
{
    wallTotal += stage.wallMilliseconds;
    kernelTotal += stage.gpuKernelMilliseconds;
    h2dTotal += stage.hostToDeviceBytes;
    d2hTotal += stage.deviceToHostBytes;
    allocTotal += stage.deviceAllocationBytes;
    inputPrimitiveTotal += stage.inputPrimitiveCount;
    outputPairTotal += stage.outputPairCount;
    outputCandidateTotal += stage.outputCandidateCount;
    if (outputContactTotal != nullptr)
    {
        *outputContactTotal += stage.outputContactCount;
    }

    if (stage.gpuUsed)
    {
        ++gpuStepCount;
    }

    if (stage.cpuFallbackUsed)
    {
        ++cpuFallbackCount;
    }
}

} // namespace

void beginStep()
{
    auto& state = runtimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.currentStep = StepSnapshot {};
}

StepSnapshot finishStep()
{
    auto& state = runtimeState();
    std::lock_guard<std::mutex> lock(state.mutex);

    const StepSnapshot finishedStep = state.currentStep;
    ++state.aggregate.completedSteps;

    accumulateStage(
        finishedStep.broadPhase,
        state.aggregate.broadWallMilliseconds,
        state.aggregate.broadGpuKernelMilliseconds,
        state.aggregate.broadHostToDeviceBytes,
        state.aggregate.broadDeviceToHostBytes,
        state.aggregate.broadDeviceAllocationBytes,
        state.aggregate.broadGpuStepCount,
        state.aggregate.broadCpuFallbackCount,
        state.aggregate.broadInputPrimitiveCount,
        state.aggregate.broadOutputPairCount,
        state.aggregate.broadOutputCandidateCount);

    accumulateStage(
        finishedStep.narrowPhase,
        state.aggregate.narrowWallMilliseconds,
        state.aggregate.narrowGpuKernelMilliseconds,
        state.aggregate.narrowHostToDeviceBytes,
        state.aggregate.narrowDeviceToHostBytes,
        state.aggregate.narrowDeviceAllocationBytes,
        state.aggregate.narrowGpuStepCount,
        state.aggregate.narrowCpuFallbackCount,
        state.aggregate.narrowInputPrimitiveCount,
        state.aggregate.narrowOutputPairCount,
        state.aggregate.narrowOutputCandidateCount,
        &state.aggregate.narrowOutputContactCount);

    state.currentStep = StepSnapshot {};
    return finishedStep;
}

AggregateSnapshot aggregateSnapshot()
{
    auto& state = runtimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.aggregate;
}

void recordBroadPhase(const StageSnapshot& stage)
{
    auto& state = runtimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.currentStep.broadPhase = stage;
}

void recordNarrowPhase(const StageSnapshot& stage)
{
    auto& state = runtimeState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.currentStep.narrowPhase = stage;
}

} // namespace SofaGpuCollision::profiling
