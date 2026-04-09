#include <SofaGpuCollision/GpuCollisionBackend.h>

#include <cuda_runtime.h>

#include <vector>

namespace
{

struct DeviceAabb
{
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
};

struct DeviceTreePairRange
{
    std::uint32_t firstOffset;
    std::uint32_t firstCount;
    std::uint32_t secondOffset;
    std::uint32_t secondCount;
};

__device__ bool overlapsOnAxis(const float aMin, const float aMax, const float bMin, const float bMax)
{
    return aMin <= bMax && bMin <= aMax;
}

__global__ void broadPhaseKernel(const DeviceAabb* aabbs, const std::uint32_t count, std::uint8_t* flags)
{
    const std::uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count || j >= count)
    {
        return;
    }

    const std::uint32_t flatIndex = i * count + j;
    if (j >= i)
    {
        flags[flatIndex] = 0;
        return;
    }

    const auto a = aabbs[i];
    const auto b = aabbs[j];

    const bool overlaps =
        overlapsOnAxis(a.minX, a.maxX, b.minX, b.maxX) &&
        overlapsOnAxis(a.minY, a.maxY, b.minY, b.maxY) &&
        overlapsOnAxis(a.minZ, a.maxZ, b.minZ, b.maxZ);

    flags[flatIndex] = overlaps ? 1 : 0;
}

__global__ void noopKernel()
{
}

__global__ void treePairOverlapKernel(
    const DeviceAabb* firstTrees,
    const DeviceAabb* secondTrees,
    const DeviceTreePairRange* pairRanges,
    const std::uint32_t pairCount,
    int* pairFlags)
{
    const std::uint32_t pairIndex = blockIdx.x;
    if (pairIndex >= pairCount)
    {
        return;
    }

    const auto range = pairRanges[pairIndex];
    const std::uint64_t totalTests =
        static_cast<std::uint64_t>(range.firstCount) * static_cast<std::uint64_t>(range.secondCount);

    for (std::uint64_t linearIndex = threadIdx.x; linearIndex < totalTests; linearIndex += blockDim.x)
    {
        const std::uint32_t i = static_cast<std::uint32_t>(linearIndex / range.secondCount);
        const std::uint32_t j = static_cast<std::uint32_t>(linearIndex % range.secondCount);

        const auto a = firstTrees[range.firstOffset + i];
        const auto b = secondTrees[range.secondOffset + j];

        const bool overlaps =
            overlapsOnAxis(a.minX, a.maxX, b.minX, b.maxX) &&
            overlapsOnAxis(a.minY, a.maxY, b.minY, b.maxY) &&
            overlapsOnAxis(a.minZ, a.maxZ, b.minZ, b.maxZ);

        if (overlaps)
        {
            atomicExch(&pairFlags[pairIndex], 1);
            return;
        }
    }
}

} // namespace

namespace SofaGpuCollision::backend
{

BackendStatus probe()
{
    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        return { false, cudaGetErrorString(err) };
    }

    if (deviceCount <= 0)
    {
        return { false, "CUDA runtime was found, but no GPU devices were reported." };
    }

    return {
        true,
        "CUDA runtime detected. Kernel integration points are compiled and ready for implementation."
    };
}

bool computeBroadPhasePairs(
    const std::vector<AxisAlignedBoundingBox>& boxes,
    std::vector<BroadPhaseIndexPair>& pairs,
    std::string& diagnostic)
{
    pairs.clear();

    if (boxes.size() < 2)
    {
        diagnostic.clear();
        return true;
    }

    std::vector<DeviceAabb> hostBoxes;
    hostBoxes.reserve(boxes.size());
    for (const auto& box : boxes)
    {
        hostBoxes.push_back(DeviceAabb {
            box.minX, box.minY, box.minZ,
            box.maxX, box.maxY, box.maxZ
        });
    }

    DeviceAabb* deviceBoxes = nullptr;
    std::uint8_t* deviceFlags = nullptr;

    const std::size_t boxBytes = hostBoxes.size() * sizeof(DeviceAabb);
    const std::size_t flagBytes = hostBoxes.size() * hostBoxes.size() * sizeof(std::uint8_t);

    cudaError_t err = cudaMalloc(&deviceBoxes, boxBytes);
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMalloc(&deviceFlags, flagBytes);
    if (err != cudaSuccess)
    {
        cudaFree(deviceBoxes);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMemcpy(deviceBoxes, hostBoxes.data(), boxBytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = cudaMemset(deviceFlags, 0, flagBytes);
    }

    if (err == cudaSuccess)
    {
        constexpr std::uint32_t tileSize = 16;
        const dim3 blockSize(tileSize, tileSize, 1);
        const dim3 gridSize(
            (static_cast<std::uint32_t>(hostBoxes.size()) + tileSize - 1) / tileSize,
            (static_cast<std::uint32_t>(hostBoxes.size()) + tileSize - 1) / tileSize,
            1);

        broadPhaseKernel<<<gridSize, blockSize>>>(deviceBoxes, static_cast<std::uint32_t>(hostBoxes.size()), deviceFlags);
        err = cudaDeviceSynchronize();
    }

    std::vector<std::uint8_t> hostFlags(hostBoxes.size() * hostBoxes.size(), 0);
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(hostFlags.data(), deviceFlags, flagBytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(deviceFlags);
    cudaFree(deviceBoxes);

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(hostBoxes.size()); ++i)
    {
        for (std::uint32_t j = 0; j < i; ++j)
        {
            if (hostFlags[i * hostBoxes.size() + j] != 0)
            {
                pairs.emplace_back(i, j);
            }
        }
    }

    diagnostic.clear();
    return true;
}

bool prefilterNarrowPhasePairs(
    const std::vector<NarrowPhaseTreePair>& inputTrees,
    std::vector<std::uint32_t>& survivingPairIndices,
    std::string& diagnostic)
{
    survivingPairIndices.clear();

    if (inputTrees.empty())
    {
        diagnostic.clear();
        return true;
    }

    std::vector<DeviceAabb> flatFirstTrees;
    std::vector<DeviceAabb> flatSecondTrees;
    std::vector<DeviceTreePairRange> pairRanges;
    std::vector<std::uint32_t> originalPairIndices;

    flatFirstTrees.reserve(1024);
    flatSecondTrees.reserve(1024);
    pairRanges.reserve(inputTrees.size());
    originalPairIndices.reserve(inputTrees.size());

    for (std::uint32_t pairIndex = 0; pairIndex < inputTrees.size(); ++pairIndex)
    {
        const auto& treePair = inputTrees[pairIndex];

        if (treePair.firstTree.empty() || treePair.secondTree.empty())
        {
            survivingPairIndices.push_back(pairIndex);
            continue;
        }

        const auto firstBegin = treePair.firstTree.size() > 1 ? 1u : 0u;
        const auto secondBegin = treePair.secondTree.size() > 1 ? 1u : 0u;
        const auto firstOffset = static_cast<std::uint32_t>(flatFirstTrees.size());
        const auto secondOffset = static_cast<std::uint32_t>(flatSecondTrees.size());

        for (std::size_t i = firstBegin; i < treePair.firstTree.size(); ++i)
        {
            const auto& box = treePair.firstTree[i];
            flatFirstTrees.push_back(DeviceAabb { box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ });
        }

        for (std::size_t i = secondBegin; i < treePair.secondTree.size(); ++i)
        {
            const auto& box = treePair.secondTree[i];
            flatSecondTrees.push_back(DeviceAabb { box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ });
        }

        const auto firstCount = static_cast<std::uint32_t>(flatFirstTrees.size()) - firstOffset;
        const auto secondCount = static_cast<std::uint32_t>(flatSecondTrees.size()) - secondOffset;

        if (firstCount == 0 || secondCount == 0)
        {
            survivingPairIndices.push_back(pairIndex);
            continue;
        }

        pairRanges.push_back(DeviceTreePairRange { firstOffset, firstCount, secondOffset, secondCount });
        originalPairIndices.push_back(pairIndex);
    }

    if (pairRanges.empty())
    {
        diagnostic.clear();
        return true;
    }

    DeviceAabb* deviceFirstTrees = nullptr;
    DeviceAabb* deviceSecondTrees = nullptr;
    DeviceTreePairRange* devicePairRanges = nullptr;
    int* devicePairFlags = nullptr;

    cudaError_t err = cudaMalloc(&deviceFirstTrees, flatFirstTrees.size() * sizeof(DeviceAabb));
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceSecondTrees, flatSecondTrees.size() * sizeof(DeviceAabb));
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&devicePairRanges, pairRanges.size() * sizeof(DeviceTreePairRange));
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&devicePairFlags, pairRanges.size() * sizeof(int));
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        cudaFree(deviceFirstTrees);
        cudaFree(deviceSecondTrees);
        cudaFree(devicePairRanges);
        cudaFree(devicePairFlags);
        return false;
    }

    err = cudaMemcpy(
        deviceFirstTrees, flatFirstTrees.data(), flatFirstTrees.size() * sizeof(DeviceAabb), cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(
            deviceSecondTrees, flatSecondTrees.data(), flatSecondTrees.size() * sizeof(DeviceAabb), cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(
            devicePairRanges, pairRanges.data(), pairRanges.size() * sizeof(DeviceTreePairRange), cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset(devicePairFlags, 0, pairRanges.size() * sizeof(int));
    }

    if (err == cudaSuccess)
    {
        constexpr std::uint32_t threadCount = 256;
        treePairOverlapKernel<<<static_cast<std::uint32_t>(pairRanges.size()), threadCount>>>(
            deviceFirstTrees,
            deviceSecondTrees,
            devicePairRanges,
            static_cast<std::uint32_t>(pairRanges.size()),
            devicePairFlags);
        err = cudaDeviceSynchronize();
    }

    std::vector<int> hostPairFlags(pairRanges.size(), 0);
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(
            hostPairFlags.data(), devicePairFlags, pairRanges.size() * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(deviceFirstTrees);
    cudaFree(deviceSecondTrees);
    cudaFree(devicePairRanges);
    cudaFree(devicePairFlags);

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    for (std::size_t i = 0; i < hostPairFlags.size(); ++i)
    {
        if (hostPairFlags[i] != 0)
        {
            survivingPairIndices.push_back(originalPairIndices[i]);
        }
    }

    diagnostic.clear();
    return true;
}

} // namespace SofaGpuCollision::backend
