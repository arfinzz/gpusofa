#pragma once

#include <SofaGpuCollision/config.h>

#include <sofa/component/collision/detection/algorithm/CollisionPipeline.h>

namespace SofaGpuCollision
{

// ============================================================================
// SOFA's CollisionPipeline, minus the per-frame CPU bounding trees of GPU models.
//
// SOFA's pipeline calls computeBoundingTree on every collision model every frame.
// For a GPU triangle model (CudaTriangleCollisionModel) that reads the positions
// back to the CPU: a GPU-to-CPU copy of the whole surface each frame. With
// GpuCollisionBroadPhase (no box culling) and GpuCollisionNarrowPhase (positions
// read in place on the GPU), the boxes are not needed: this pipeline builds a GPU
// model's bounding tree once (so the model has the root cube SOFA's intersector
// lookup expects) and never again. Everything else is SOFA's pipeline unchanged.
// Pair it with GpuCollisionBroadPhase's testGpuModelBoxes=false, since the root
// boxes of GPU models then stay at their first-frame value.
// ============================================================================
class SOFA_GPU_COLLISION_API GpuCollisionPipeline : public sofa::component::collision::detection::algorithm::CollisionPipeline
{
public:
    SOFA_CLASS(GpuCollisionPipeline, sofa::component::collision::detection::algorithm::CollisionPipeline);

    sofa::core::objectmodel::Data<bool> d_skipGpuBoundingTrees;

protected:
    GpuCollisionPipeline();
    void doCollisionDetection(const sofa::type::vector<sofa::core::CollisionModel*>& collisionModels) override;
};

} // namespace SofaGpuCollision
