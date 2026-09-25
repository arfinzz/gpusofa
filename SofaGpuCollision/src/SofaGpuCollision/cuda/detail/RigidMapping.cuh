// RigidMapping.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
//
// A rigid body's surface points on the GPU (GpuRigidMapping): positions and
// velocities mapped from the rigid pose, and the surface forces mapped back to a
// force and torque. SofaCUDA's RigidMapping kernels do the same, but its force
// kernel (RigidMappingCuda3f_applyJT_kernel, SOFA v25.12) writes the per-block
// partial forces into the torque slots after its reduction, so the body receives
// a wrong torque. Here the reduction runs in double precision over all points in
// one block and returns force and torque separately.

namespace
{

constexpr int kRigidMappingThreads = 256;

// out_i = R p_i + t; rotated_i = R p_i.
__global__ void rigidMappingApplyKernel(const int n, const double r00, const double r01, const double r02,
                                        const double r10, const double r11, const double r12,
                                        const double r20, const double r21, const double r22,
                                        const double tx, const double ty, const double tz,
                                        const float* __restrict__ points, float* __restrict__ out, float* __restrict__ rotated)
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        const double px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
        const double qx = r00 * px + r01 * py + r02 * pz;
        const double qy = r10 * px + r11 * py + r12 * pz;
        const double qz = r20 * px + r21 * py + r22 * pz;
        rotated[3 * i] = static_cast<float>(qx);
        rotated[3 * i + 1] = static_cast<float>(qy);
        rotated[3 * i + 2] = static_cast<float>(qz);
        out[3 * i] = static_cast<float>(qx + tx);
        out[3 * i + 1] = static_cast<float>(qy + ty);
        out[3 * i + 2] = static_cast<float>(qz + tz);
    }
}

// out_i = v + w x rotated_i (overwritten, or added to with accumulate).
__global__ void rigidMappingApplyJKernel(const int n, const double vx, const double vy, const double vz,
                                         const double wx, const double wy, const double wz,
                                         const float* __restrict__ rotated, float* __restrict__ out, const int accumulate)
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        const double rx = rotated[3 * i], ry = rotated[3 * i + 1], rz = rotated[3 * i + 2];
        const double ox = vx + (wy * rz - wz * ry);
        const double oy = vy + (wz * rx - wx * rz);
        const double oz = vz + (wx * ry - wy * rx);
        if (accumulate)
        {
            out[3 * i] += static_cast<float>(ox);
            out[3 * i + 1] += static_cast<float>(oy);
            out[3 * i + 2] += static_cast<float>(oz);
        }
        else
        {
            out[3 * i] = static_cast<float>(ox);
            out[3 * i + 1] = static_cast<float>(oy);
            out[3 * i + 2] = static_cast<float>(oz);
        }
    }
}

// result[0..2] = sum f_i, result[3..5] = sum rotated_i x f_i, in double, one block.
__global__ void rigidMappingApplyJTKernel(const int n, const float* __restrict__ rotated, const float* __restrict__ force,
                                          double* __restrict__ result)
{
    __shared__ double partial[6][kRigidMappingThreads];
    double acc[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const double fx = force[3 * i], fy = force[3 * i + 1], fz = force[3 * i + 2];
        const double rx = rotated[3 * i], ry = rotated[3 * i + 1], rz = rotated[3 * i + 2];
        acc[0] += fx;
        acc[1] += fy;
        acc[2] += fz;
        acc[3] += ry * fz - rz * fy;
        acc[4] += rz * fx - rx * fz;
        acc[5] += rx * fy - ry * fx;
    }
    for (int k = 0; k < 6; ++k) partial[k][threadIdx.x] = acc[k];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            for (int k = 0; k < 6; ++k) partial[k][threadIdx.x] += partial[k][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < 6) result[threadIdx.x] = partial[threadIdx.x][0];
}

} // namespace

namespace SofaGpuCollision::backend
{

namespace
{
double* g_rigidMappingResult = nullptr;   // 6 doubles on the device, shared by all rigid mappings

unsigned rigidMappingBlocks(const int n)
{
    return static_cast<unsigned>(std::max(1, std::min((n + kRigidMappingThreads - 1) / kRigidMappingThreads, 1024)));
}
} // namespace

bool rigidMappingApply(const int n, const double rotation[9], const double translation[3], const void* points, void* out,
                       void* rotated, std::string& diagnostic)
{
    if (n <= 0) { diagnostic.clear(); return true; }
    rigidMappingApplyKernel<<<rigidMappingBlocks(n), kRigidMappingThreads>>>(
        n, rotation[0], rotation[1], rotation[2], rotation[3], rotation[4], rotation[5], rotation[6], rotation[7], rotation[8],
        translation[0], translation[1], translation[2], static_cast<const float*>(points), static_cast<float*>(out),
        static_cast<float*>(rotated));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { diagnostic = std::string("Rigid mapping (positions): ") + cudaGetErrorString(err); return false; }
    diagnostic.clear();
    return true;
}

bool rigidMappingApplyJ(const int n, const double velocity[3], const double angular[3], const void* rotated, void* out,
                        const bool accumulate, std::string& diagnostic)
{
    if (n <= 0) { diagnostic.clear(); return true; }
    rigidMappingApplyJKernel<<<rigidMappingBlocks(n), kRigidMappingThreads>>>(
        n, velocity[0], velocity[1], velocity[2], angular[0], angular[1], angular[2], static_cast<const float*>(rotated),
        static_cast<float*>(out), accumulate ? 1 : 0);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { diagnostic = std::string("Rigid mapping (velocities): ") + cudaGetErrorString(err); return false; }
    diagnostic.clear();
    return true;
}

bool rigidMappingApplyJT(const int n, const void* rotated, const void* force, double forceAndTorque[6], std::string& diagnostic)
{
    for (int k = 0; k < 6; ++k) forceAndTorque[k] = 0.0;
    if (n <= 0) { diagnostic.clear(); return true; }
    if (g_rigidMappingResult == nullptr &&
        cudaMalloc(reinterpret_cast<void**>(&g_rigidMappingResult), 6 * sizeof(double)) != cudaSuccess)
    {
        g_rigidMappingResult = nullptr;
        diagnostic = "Rigid mapping (forces): could not allocate the result buffer.";
        return false;
    }
    rigidMappingApplyJTKernel<<<1, kRigidMappingThreads>>>(n, static_cast<const float*>(rotated),
                                                           static_cast<const float*>(force), g_rigidMappingResult);
    const cudaError_t err = cudaMemcpy(forceAndTorque, g_rigidMappingResult, 6 * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { diagnostic = std::string("Rigid mapping (forces): ") + cudaGetErrorString(err); return false; }
    diagnostic.clear();
    return true;
}

} // namespace SofaGpuCollision::backend
