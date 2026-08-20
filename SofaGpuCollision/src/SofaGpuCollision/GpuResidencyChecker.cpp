#include <SofaGpuCollision/GpuResidencyChecker.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sstream>

namespace SofaGpuCollision
{

int GpuResidencyCheckerClass = sofa::core::RegisterObject(
    "Asserts that no CudaVec3f state vector was pulled to the host during the frame "
    "(Gate 5: proves the simulation stayed device-resident). Names the offending "
    "mechanical object and vector when a transfer happens.")
    .add<GpuResidencyChecker>();

GpuResidencyChecker::GpuResidencyChecker()
    : d_checkPosition(initData(&d_checkPosition, true, "checkPosition",
        "Assert the position vector was not copied to the host this frame."))
    , d_checkVelocity(initData(&d_checkVelocity, true, "checkVelocity",
        "Assert the velocity vector was not copied to the host this frame."))
    , d_checkForce(initData(&d_checkForce, true, "checkForce",
        "Assert the force vector was not copied to the host this frame."))
    , d_startFrame(initData(&d_startFrame, static_cast<unsigned int>(3), "startFrame",
        "First frame to check. Early frames legitimately touch the host during init/upload, so skip them."))
    , d_reportInterval(initData(&d_reportInterval, static_cast<unsigned int>(0), "reportInterval",
        "Print a clean-frame summary every N frames. 0 = only report violations and the final summary."))
    , d_failFast(initData(&d_failFast, false, "failFast",
        "Emit msg_error instead of msg_warning on the first violation, so batch runs fail loudly."))
{
    this->f_listening.setValue(true);
}

void GpuResidencyChecker::init()
{
    sofa::core::objectmodel::BaseObject::init();
    this->f_listening.setValue(true);
}

void GpuResidencyChecker::checkFrame(const char* phase)
{
    // Every CudaVec3f mechanical state below this component's context.
    std::vector<sofa::core::behavior::MechanicalState<DataTypes>*> states;
    this->getContext()->getRootContext()->template get<sofa::core::behavior::MechanicalState<DataTypes>>(
        &states, sofa::core::objectmodel::BaseContext::SearchDown);

    std::ostringstream offenders;
    std::size_t violationCount = 0;

    for (auto* state : states)
    {
        if (state == nullptr) continue;

        const auto report = [&](const char* vectorName, bool hostValid) {
            if (!hostValid) return;
            if (violationCount > 0) offenders << ", ";
            offenders << state->getName() << '.' << vectorName;
            ++violationCount;
        };

        if (d_checkPosition.getValue())
        {
            report("position", state->read(sofa::core::vec_id::read_access::position)->getValue().isHostValid());
        }
        if (d_checkVelocity.getValue())
        {
            report("velocity", state->read(sofa::core::vec_id::read_access::velocity)->getValue().isHostValid());
        }
        if (d_checkForce.getValue())
        {
            report("force", state->read(sofa::core::vec_id::read_access::force)->getValue().isHostValid());
        }
    }

    if (violationCount > 0)
    {
        ++m_violationFrames;
        if (!m_reportedViolation)
        {
            m_reportedViolation = true;
            const std::string message =
                std::string("DEVICE->HOST TRANSFER DETECTED at ") + phase + " of frame " + std::to_string(m_frame) +
                ". These vectors were copied to the host: " + offenders.str() +
                ". A CPU component is reading GPU state (helper::ReadAccessor / operator[] / "
                "hostRead all trigger this). Further occurrences suppressed; a summary follows at the end.";
            if (d_failFast.getValue()) { msg_error() << message; }
            else                       { msg_warning() << message; }
        }
    }
    else
    {
        ++m_cleanFrames;
    }

    const unsigned int interval = d_reportInterval.getValue();
    if (interval > 0 && (m_frame % interval) == 0)
    {
        msg_info() << "GPU residency @frame " << m_frame
                   << ": clean=" << m_cleanFrames << " violations=" << m_violationFrames;
    }
}

void GpuResidencyChecker::handleEvent(sofa::core::objectmodel::Event* event)
{
    // Sample at BOTH ends of the step, not just the end.
    //
    // A single end-of-frame sample can be fooled: hostRead() sets hostIsValid
    // true but leaves deviceIsValid true, so a CPU component that reads state
    // mid-frame and is FOLLOWED by any deviceWrite() (e.g. the contact force
    // field) leaves hostIsValid false again by frame end — the transfer happened
    // and the end-of-frame check would still report clean. Sampling at the start
    // of the step catches exactly that window, because nothing has written to
    // the device yet to mask it.
    //
    // This still cannot prove the absence of a mid-step read that is masked
    // before BOTH sample points; that would need memcpy-level interception in
    // SofaCUDA. Documented rather than overclaimed.
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        if (m_frame >= static_cast<std::uint64_t>(d_startFrame.getValue()))
        {
            checkFrame("frame-begin");
        }
    }
    else if (sofa::simulation::AnimateEndEvent::checkEventType(event))
    {
        ++m_frame;
        if (m_frame >= static_cast<std::uint64_t>(d_startFrame.getValue()))
        {
            checkFrame("frame-end");
        }
    }
    sofa::core::objectmodel::BaseObject::handleEvent(event);
}

} // namespace SofaGpuCollision
