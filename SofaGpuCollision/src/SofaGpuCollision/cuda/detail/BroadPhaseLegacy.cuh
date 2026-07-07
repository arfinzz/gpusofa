// BroadPhaseLegacy.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included only by cuda/GpuCollisionBackend.cu, in dependency order; not
// independently compilable (kernels must be visible to their launch sites
// without -rdc/device-linking, and one TU keeps codegen identical to the
// pre-split monolith). Split from the monolithic backend on 2026-07-03.
// Legacy AABB-tree broad-phase kernels, the brute-force exact-contact
// kernel, and the probe/broad-phase/prefilter/exact host drivers.

namespace
{

__global__ void broadPhaseCompactKernel(
    const DeviceAabb* aabbs,
    const std::uint32_t count,
    DeviceIndexPair* pairs,
    std::uint32_t* pairCount)
{
    const std::uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count || j >= count || i >= j)
    {
        return;
    }

    const auto a = aabbs[i];
    const auto b = aabbs[j];

    const bool overlaps =
        overlapsOnAxis(a.minX, a.maxX, b.minX, b.maxX) &&
        overlapsOnAxis(a.minY, a.maxY, b.minY, b.maxY) &&
        overlapsOnAxis(a.minZ, a.maxZ, b.minZ, b.maxZ);

    if (overlaps)
    {
        const std::uint32_t outputIndex = atomicAdd(pairCount, 1u);
        pairs[outputIndex] = DeviceIndexPair { i, j };
    }
}

__global__ void treePairOverlapKernel(
    const DeviceAabb* firstTrees,
    const DeviceAabb* secondTrees,
    const DeviceTreePairRange* pairRanges,
    const std::uint32_t pairCount,
    DeviceContactCandidate* contactCandidates,
    std::uint32_t* candidateCount)
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
            const std::uint32_t outputIndex = atomicAdd(candidateCount, 1u);
            contactCandidates[outputIndex] = DeviceContactCandidate {
                pairIndex,
                range.firstBaseIndex + i,
                range.secondBaseIndex + j,
            };
        }
    }
}

__global__ void exactTriangleContactKernel(
    const DeviceTriangle* firstTriangles,
    const std::uint32_t firstTriangleCount,
    const DeviceTriangle* secondTriangles,
    const std::uint32_t secondTriangleCount,
    DeviceExactContact* contacts,
    std::uint32_t* contactCount)
{
    const std::uint32_t firstIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t secondIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (firstIndex >= firstTriangleCount || secondIndex >= secondTriangleCount)
    {
        return;
    }

    DeviceExactContact exactContact {};
    if (exactTriangleIntersection(firstTriangles[firstIndex], secondTriangles[secondIndex], exactContact))
    {
        const std::uint32_t outputIndex = atomicAdd(contactCount, 1u);
        contacts[outputIndex] = exactContact;
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
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    pairs.clear();
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }

    if (boxes.size() < 2)
    {
        diagnostic.clear();
        return true;
    }
    if (executionStats != nullptr)
    {
        executionStats->inputPrimitiveCount = static_cast<std::uint32_t>(boxes.size());
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
    DeviceIndexPair* devicePairs = nullptr;
    std::uint32_t* devicePairCount = nullptr;

    const std::size_t boxBytes = hostBoxes.size() * sizeof(DeviceAabb);
    const std::size_t maxPairCount = (hostBoxes.size() * (hostBoxes.size() - 1)) / 2;
    const std::size_t pairBytes = maxPairCount * sizeof(DeviceIndexPair);
    const std::size_t pairCountBytes = sizeof(std::uint32_t);
    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(boxBytes);
        executionStats->deviceAllocationBytes += static_cast<std::uint64_t>(boxBytes + pairBytes + pairCountBytes);
    }

    cudaError_t err = cudaMalloc(&deviceBoxes, boxBytes);
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMalloc(&devicePairs, pairBytes);
    if (err != cudaSuccess)
    {
        cudaFree(deviceBoxes);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMalloc(&devicePairCount, pairCountBytes);
    if (err != cudaSuccess)
    {
        cudaFree(devicePairs);
        cudaFree(deviceBoxes);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMemcpy(deviceBoxes, hostBoxes.data(), boxBytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = cudaMemset(devicePairCount, 0, pairCountBytes);
    }

    if (err == cudaSuccess)
    {
        constexpr std::uint32_t tileSize = 16;
        const dim3 blockSize(tileSize, tileSize, 1);
        const dim3 gridSize(
            (static_cast<std::uint32_t>(hostBoxes.size()) + tileSize - 1) / tileSize,
            (static_cast<std::uint32_t>(hostBoxes.size()) + tileSize - 1) / tileSize,
            1);
        cudaEvent_t kernelStart {};
        cudaEvent_t kernelEnd {};
        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelEnd);
        cudaEventRecord(kernelStart);
        broadPhaseCompactKernel<<<gridSize, blockSize>>>(
            deviceBoxes,
            static_cast<std::uint32_t>(hostBoxes.size()),
            devicePairs,
            devicePairCount);
        cudaEventRecord(kernelEnd);
        err = cudaDeviceSynchronize();
        if (err == cudaSuccess && executionStats != nullptr)
        {
            float elapsedMs = 0.0f;
            cudaEventElapsedTime(&elapsedMs, kernelStart, kernelEnd);
            executionStats->gpuKernelMilliseconds += static_cast<double>(elapsedMs);
            executionStats->kernelLaunchCount += 1;
        }
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd);
    }

    std::uint32_t hostPairCount = 0;
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(&hostPairCount, devicePairCount, pairCountBytes, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess)
    {
        cudaFree(devicePairCount);
        cudaFree(devicePairs);
        cudaFree(deviceBoxes);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->deviceToHostBytes += static_cast<std::uint64_t>(
            pairCountBytes + hostPairCount * sizeof(DeviceIndexPair));
    }

    std::vector<DeviceIndexPair> hostPairs(hostPairCount);
    if (hostPairCount > 0)
    {
        err = cudaMemcpy(hostPairs.data(), devicePairs, hostPairCount * sizeof(DeviceIndexPair), cudaMemcpyDeviceToHost);
    }

    cudaFree(devicePairCount);
    cudaFree(devicePairs);
    cudaFree(deviceBoxes);

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->outputPairCount = hostPairCount;
    }

    pairs.reserve(hostPairs.size());
    for (const auto& pair : hostPairs)
    {
        pairs.emplace_back(pair.first, pair.second);
    }

    diagnostic.clear();
    return true;
}

bool prefilterNarrowPhasePairs(
    const std::vector<NarrowPhaseTreePair>& inputTrees,
    std::vector<std::uint32_t>& survivingPairIndices,
    std::vector<NarrowPhaseContactCandidate>* contactCandidates,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    survivingPairIndices.clear();
    if (contactCandidates != nullptr)
    {
        contactCandidates->clear();
    }
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount = static_cast<std::uint32_t>(inputTrees.size());
    }

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

        pairRanges.push_back(DeviceTreePairRange {
            firstOffset,
            firstCount,
            secondOffset,
            secondCount,
            static_cast<std::uint32_t>(firstBegin),
            static_cast<std::uint32_t>(secondBegin),
        });
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
    DeviceContactCandidate* deviceContactCandidates = nullptr;
    std::uint32_t* deviceCandidateCount = nullptr;

    std::uint64_t maxCandidateCount = 0;
    for (const auto& range : pairRanges)
    {
        maxCandidateCount += static_cast<std::uint64_t>(range.firstCount) * static_cast<std::uint64_t>(range.secondCount);
    }

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
        err = cudaMalloc(&deviceContactCandidates, maxCandidateCount * sizeof(DeviceContactCandidate));
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceCandidateCount, sizeof(std::uint32_t));
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        cudaFree(deviceFirstTrees);
        cudaFree(deviceSecondTrees);
        cudaFree(devicePairRanges);
        cudaFree(deviceContactCandidates);
        cudaFree(deviceCandidateCount);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(
            flatFirstTrees.size() * sizeof(DeviceAabb) +
            flatSecondTrees.size() * sizeof(DeviceAabb) +
            pairRanges.size() * sizeof(DeviceTreePairRange));
        executionStats->deviceAllocationBytes += static_cast<std::uint64_t>(
            flatFirstTrees.size() * sizeof(DeviceAabb) +
            flatSecondTrees.size() * sizeof(DeviceAabb) +
            pairRanges.size() * sizeof(DeviceTreePairRange) +
            maxCandidateCount * sizeof(DeviceContactCandidate) +
            sizeof(std::uint32_t));
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
        err = cudaMemset(deviceCandidateCount, 0, sizeof(std::uint32_t));
    }

    if (err == cudaSuccess)
    {
        constexpr std::uint32_t threadCount = 256;
        cudaEvent_t kernelStart {};
        cudaEvent_t kernelEnd {};
        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelEnd);
        cudaEventRecord(kernelStart);
        treePairOverlapKernel<<<static_cast<std::uint32_t>(pairRanges.size()), threadCount>>>(
            deviceFirstTrees,
            deviceSecondTrees,
            devicePairRanges,
            static_cast<std::uint32_t>(pairRanges.size()),
            deviceContactCandidates,
            deviceCandidateCount);
        cudaEventRecord(kernelEnd);
        err = cudaDeviceSynchronize();
        if (err == cudaSuccess && executionStats != nullptr)
        {
            float elapsedMs = 0.0f;
            cudaEventElapsedTime(&elapsedMs, kernelStart, kernelEnd);
            executionStats->gpuKernelMilliseconds += static_cast<double>(elapsedMs);
            executionStats->kernelLaunchCount += 1;
        }
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd);
    }

    std::uint32_t hostCandidateCount = 0;
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(&hostCandidateCount, deviceCandidateCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess)
    {
        cudaFree(deviceFirstTrees);
        cudaFree(deviceSecondTrees);
        cudaFree(devicePairRanges);
        cudaFree(deviceContactCandidates);
        cudaFree(deviceCandidateCount);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->deviceToHostBytes += static_cast<std::uint64_t>(
            sizeof(std::uint32_t) + hostCandidateCount * sizeof(DeviceContactCandidate));
    }

    std::vector<DeviceContactCandidate> hostCandidates(hostCandidateCount);
    if (hostCandidateCount > 0)
    {
        err = cudaMemcpy(
            hostCandidates.data(),
            deviceContactCandidates,
            hostCandidateCount * sizeof(DeviceContactCandidate),
            cudaMemcpyDeviceToHost);
    }

    cudaFree(deviceFirstTrees);
    cudaFree(deviceSecondTrees);
    cudaFree(devicePairRanges);
    cudaFree(deviceContactCandidates);
    cudaFree(deviceCandidateCount);

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->outputCandidateCount = hostCandidateCount;
    }

    std::vector<std::uint8_t> keepPair(pairRanges.size(), 0);
    for (const auto& candidate : hostCandidates)
    {
        if (candidate.pairIndex < keepPair.size())
        {
            keepPair[candidate.pairIndex] = 1;
            if (contactCandidates != nullptr)
            {
                contactCandidates->push_back(NarrowPhaseContactCandidate {
                    originalPairIndices[candidate.pairIndex],
                    candidate.firstLeafIndex,
                    candidate.secondLeafIndex,
                });
            }
        }
    }

    for (std::size_t i = 0; i < keepPair.size(); ++i)
    {
        if (keepPair[i] != 0)
        {
            survivingPairIndices.push_back(originalPairIndices[i]);
        }
    }

    if (executionStats != nullptr)
    {
        executionStats->outputPairCount = static_cast<std::uint32_t>(survivingPairIndices.size());
    }

    diagnostic.clear();
    return true;
}

bool computeExactTriangleContacts(
    const std::vector<TrianglePrimitive>& firstTriangles,
    const std::vector<TrianglePrimitive>& secondTriangles,
    std::vector<ExactContact>& contacts,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    contacts.clear();
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount =
            static_cast<std::uint32_t>(firstTriangles.size() + secondTriangles.size());
    }

    if (firstTriangles.empty() || secondTriangles.empty())
    {
        diagnostic.clear();
        return true;
    }

    std::vector<DeviceTriangle> hostFirstTriangles;
    std::vector<DeviceTriangle> hostSecondTriangles;
    hostFirstTriangles.reserve(firstTriangles.size());
    hostSecondTriangles.reserve(secondTriangles.size());

    for (const auto& triangle : firstTriangles)
    {
        hostFirstTriangles.push_back(DeviceTriangle {
            make_float3(triangle.p0.x, triangle.p0.y, triangle.p0.z),
            make_float3(triangle.p1.x, triangle.p1.y, triangle.p1.z),
            make_float3(triangle.p2.x, triangle.p2.y, triangle.p2.z),
            triangle.triangleIndex,
        });
    }

    for (const auto& triangle : secondTriangles)
    {
        hostSecondTriangles.push_back(DeviceTriangle {
            make_float3(triangle.p0.x, triangle.p0.y, triangle.p0.z),
            make_float3(triangle.p1.x, triangle.p1.y, triangle.p1.z),
            make_float3(triangle.p2.x, triangle.p2.y, triangle.p2.z),
            triangle.triangleIndex,
        });
    }

    DeviceTriangle* deviceFirstTriangles = nullptr;
    DeviceTriangle* deviceSecondTriangles = nullptr;
    DeviceExactContact* deviceContacts = nullptr;
    std::uint32_t* deviceContactCount = nullptr;

    const std::size_t firstBytes = hostFirstTriangles.size() * sizeof(DeviceTriangle);
    const std::size_t secondBytes = hostSecondTriangles.size() * sizeof(DeviceTriangle);
    const std::size_t maxContactCount = hostFirstTriangles.size() * hostSecondTriangles.size();
    const std::size_t contactBytes = maxContactCount * sizeof(DeviceExactContact);
    const std::size_t contactCountBytes = sizeof(std::uint32_t);

    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(firstBytes + secondBytes);
        executionStats->deviceAllocationBytes += static_cast<std::uint64_t>(firstBytes + secondBytes + contactBytes + contactCountBytes);
    }

    cudaError_t err = cudaMalloc(&deviceFirstTriangles, firstBytes);
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceSecondTriangles, secondBytes);
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceContacts, contactBytes);
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceContactCount, contactCountBytes);
    }

    if (err != cudaSuccess)
    {
        cudaFree(deviceFirstTriangles);
        cudaFree(deviceSecondTriangles);
        cudaFree(deviceContacts);
        cudaFree(deviceContactCount);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMemcpy(deviceFirstTriangles, hostFirstTriangles.data(), firstBytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(deviceSecondTriangles, hostSecondTriangles.data(), secondBytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset(deviceContactCount, 0, contactCountBytes);
    }

    if (err == cudaSuccess)
    {
        constexpr std::uint32_t tileSize = 16;
        const dim3 blockSize(tileSize, tileSize, 1);
        const dim3 gridSize(
            (static_cast<std::uint32_t>(hostSecondTriangles.size()) + tileSize - 1) / tileSize,
            (static_cast<std::uint32_t>(hostFirstTriangles.size()) + tileSize - 1) / tileSize,
            1);

        cudaEvent_t kernelStart {};
        cudaEvent_t kernelEnd {};
        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelEnd);
        cudaEventRecord(kernelStart);
        exactTriangleContactKernel<<<gridSize, blockSize>>>(
            deviceFirstTriangles,
            static_cast<std::uint32_t>(hostFirstTriangles.size()),
            deviceSecondTriangles,
            static_cast<std::uint32_t>(hostSecondTriangles.size()),
            deviceContacts,
            deviceContactCount);
        cudaEventRecord(kernelEnd);
        err = cudaDeviceSynchronize();
        if (err == cudaSuccess && executionStats != nullptr)
        {
            float elapsedMs = 0.0f;
            cudaEventElapsedTime(&elapsedMs, kernelStart, kernelEnd);
            executionStats->gpuKernelMilliseconds += static_cast<double>(elapsedMs);
            executionStats->kernelLaunchCount += 1;
        }
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd);
    }

    std::uint32_t hostContactCount = 0;
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(&hostContactCount, deviceContactCount, contactCountBytes, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess)
    {
        cudaFree(deviceFirstTriangles);
        cudaFree(deviceSecondTriangles);
        cudaFree(deviceContacts);
        cudaFree(deviceContactCount);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->deviceToHostBytes += static_cast<std::uint64_t>(
            contactCountBytes + hostContactCount * sizeof(DeviceExactContact));
    }

    std::vector<DeviceExactContact> hostContacts(hostContactCount);
    if (hostContactCount > 0)
    {
        err = cudaMemcpy(
            hostContacts.data(),
            deviceContacts,
            hostContactCount * sizeof(DeviceExactContact),
            cudaMemcpyDeviceToHost);
    }

    cudaFree(deviceFirstTriangles);
    cudaFree(deviceSecondTriangles);
    cudaFree(deviceContacts);
    cudaFree(deviceContactCount);

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->outputPairCount = hostContactCount > 0 ? 1u : 0u;
        executionStats->outputCandidateCount = hostContactCount;
        executionStats->outputContactCount = hostContactCount;
    }

    contacts.reserve(hostContacts.size());
    for (const auto& contact : hostContacts)
    {
        contacts.push_back(ExactContact {
            contact.firstTriangleIndex,
            contact.secondTriangleIndex,
            TriangleVertex { contact.pointOnFirst.x, contact.pointOnFirst.y, contact.pointOnFirst.z },
            TriangleVertex { contact.pointOnSecond.x, contact.pointOnSecond.y, contact.pointOnSecond.z },
            TriangleVertex { contact.normal.x, contact.normal.y, contact.normal.z },
            contact.signedDistance,
        });
    }

    diagnostic.clear();
    return true;
}



} // namespace SofaGpuCollision::backend
