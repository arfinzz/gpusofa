#include <SofaGpuCollision/GpuCollisionPipeline.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <SofaCUDA/component/collision/geometry/CudaTriangleModel.h>

namespace SofaGpuCollision
{

int GpuCollisionPipelineClass = sofa::core::RegisterObject(
    "SOFA's CollisionPipeline without the per-frame CPU bounding trees of GPU triangle models (which copy "
    "their positions back to the CPU every frame).")
    .add<GpuCollisionPipeline>();

GpuCollisionPipeline::GpuCollisionPipeline()
    : d_skipGpuBoundingTrees(initData(&d_skipGpuBoundingTrees, true, "skipGpuBoundingTrees",
        "Build a GPU triangle model's bounding tree only once (no per-frame GPU-to-CPU copy of its positions)."))
{
}

void GpuCollisionPipeline::doCollisionDetection(const sofa::type::vector<sofa::core::CollisionModel*>& collisionModels)
{
    // CollisionPipeline::doCollisionDetection, except for GPU models' bounding trees.
    SCOPED_TIMER_VARNAME(docollisiontimer, "doCollisionDetection");
    sofa::type::vector<sofa::core::CollisionModel*> vectBoundingVolume;
    {
        SCOPED_TIMER_VARNAME(bboxtimer, "ComputeBoundingTree");
        const bool continuous = intersectionMethod->useContinuous();
        const auto continuousIntersectionType = intersectionMethod->continuousIntersectionType();
        const SReal dt = getContext()->getDt();
        const int usedDepth = ((broadPhaseDetection && broadPhaseDetection->needsDeepBoundingTree()) ||
                               (narrowPhaseDetection && narrowPhaseDetection->needsDeepBoundingTree()))
                                  ? d_depth.getValue() : 0;
        for (auto* model : collisionModels)
        {
            if (!model->isActive()) continue;
            const bool gpuModel = dynamic_cast<sofa::gpu::cuda::CudaTriangleCollisionModel*>(model) != nullptr;
            const bool treeBuilt = model->getPrevious() != nullptr;
            if (!(d_skipGpuBoundingTrees.getValue() && gpuModel && treeBuilt))
            {
                if (continuous) model->computeContinuousBoundingTree(dt, continuousIntersectionType, usedDepth);
                else model->computeBoundingTree(usedDepth);
            }
            vectBoundingVolume.push_back(model->getFirst());
        }
    }
    if (broadPhaseDetection == nullptr) return;
    {
        SCOPED_TIMER_VARNAME(broadphase, "BroadPhase");
        intersectionMethod->beginBroadPhase();
        broadPhaseDetection->beginBroadPhase();
        broadPhaseDetection->addCollisionModels(vectBoundingVolume);
        broadPhaseDetection->endBroadPhase();
        intersectionMethod->endBroadPhase();
    }
    if (narrowPhaseDetection == nullptr) return;
    {
        SCOPED_TIMER_VARNAME(narrowphase, "NarrowPhase");
        intersectionMethod->beginNarrowPhase();
        narrowPhaseDetection->beginNarrowPhase();
        const auto& pairs = broadPhaseDetection->getCollisionModelPairs();
        narrowPhaseDetection->addCollisionPairs(pairs);
        narrowPhaseDetection->endNarrowPhase();
        intersectionMethod->endNarrowPhase();
    }
}

} // namespace SofaGpuCollision
