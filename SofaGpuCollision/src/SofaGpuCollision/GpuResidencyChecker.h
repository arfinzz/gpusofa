#pragma once

#include <SofaGpuCollision/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/gpu/cuda/CudaTypes.h>

#include <cstdint>
#include <string>
#include <vector>

namespace SofaGpuCollision
{

// ============================================================================
// Gate 5 — the zero-transfer assertion.
//
// `sofa::type::vector_device` (which backs every CudaVec3f state vector) keeps
// two validity flags and exposes them publicly:
//
//     isHostValid()          -- the CPU copy is up to date
//     isDeviceValid(gpu)     -- the GPU copy is up to date
//
// The mechanics that make this a *proof* rather than a heuristic:
//   * a GPU kernel writing through deviceWrite() sets hostIsValid = false;
//   * any CPU component reading through hostRead()/ReadAccessor copies the
//     vector back and sets hostIsValid = true.
//
// So after a frame that ran entirely on the GPU, x / v / f must ALL report
// isHostValid() == false. If any of them reports true, something pulled the
// state to the host this frame — and this component names which vector and
// which mechanical object, instead of leaving you to hunt through a profile.
//
// This is deliberately an assertion component, not a profiler: it needs no
// external tooling, works under WSL2 where nsys cannot capture GPU timelines,
// and costs nothing but a flag read per frame.
// ============================================================================
class SOFA_GPU_COLLISION_API GpuResidencyChecker : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(GpuResidencyChecker, sofa::core::objectmodel::BaseObject);

    using DataTypes = sofa::gpu::cuda::CudaVec3fTypes;

    GpuResidencyChecker();
    ~GpuResidencyChecker() override = default;

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;

private:
    using DataBool = sofa::core::objectmodel::Data<bool>;
    using DataUInt = sofa::core::objectmodel::Data<unsigned int>;

    DataBool d_checkPosition;
    DataBool d_checkVelocity;
    DataBool d_checkForce;
    DataUInt d_startFrame;      ///< skip init/warm-up frames, which legitimately touch the host
    DataUInt d_reportInterval;  ///< 0 = report only violations
    DataBool d_failFast;        ///< msg_error (visible + test-failing) instead of msg_warning

    void checkFrame(const char* phase);

    std::uint64_t m_frame { 0 };
    std::uint64_t m_violationFrames { 0 };
    std::uint64_t m_cleanFrames { 0 };
    bool m_reportedViolation { false };
};

} // namespace SofaGpuCollision
