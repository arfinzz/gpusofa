#include <SofaGpuCollision/GpuCollisionBackend.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using SofaGpuCollision::backend::DenseGridConfig;
using SofaGpuCollision::backend::ExactContact;
using SofaGpuCollision::backend::FeatureBasedProximityConfig;
using SofaGpuCollision::backend::FeatureBasedProximityStats;
using SofaGpuCollision::backend::PointCloudSurface;
using SofaGpuCollision::backend::ProximityContact;
using SofaGpuCollision::backend::ProximityFeatureKind;
using SofaGpuCollision::backend::TriangleIndexedSurface;
using SofaGpuCollision::backend::TrianglePrimitive;
using SofaGpuCollision::backend::TriangleVertex;

int envInt(const char* name, const int fallback)
{
    if (const char* raw = std::getenv(name))
    {
        return std::max(0, std::atoi(raw));
    }
    return fallback;
}

float envFloat(const char* name, const float fallback)
{
    if (const char* raw = std::getenv(name))
    {
        return std::atof(raw);
    }
    return fallback;
}

bool envBool(const char* name, const bool fallback)
{
    if (const char* raw = std::getenv(name))
    {
        const std::string value(raw);
        return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "yes";
    }
    return fallback;
}

TriangleVertex makeVertex(const float x, const float y, const float z)
{
    return TriangleVertex { x, y, z };
}

TrianglePrimitive makeTriangle(
    const TriangleVertex& p0,
    const TriangleVertex& p1,
    const TriangleVertex& p2,
    const std::uint32_t index)
{
    return TrianglePrimitive { p0, p1, p2, index };
}

TriangleVertex addScaled(const TriangleVertex& p, const TriangleVertex& u, const float us, const TriangleVertex& v, const float vs)
{
    return TriangleVertex {
        p.x + u.x * us + v.x * vs,
        p.y + u.y * us + v.y * vs,
        p.z + u.z * us + v.z * vs,
    };
}

void addPlane(
    std::vector<TrianglePrimitive>& triangles,
    const TriangleVertex& origin,
    const TriangleVertex& u,
    const TriangleVertex& v,
    const int uSegments,
    const int vSegments)
{
    for (int j = 0; j < vSegments; ++j)
    {
        for (int i = 0; i < uSegments; ++i)
        {
            const float u0 = static_cast<float>(i) / static_cast<float>(uSegments);
            const float u1 = static_cast<float>(i + 1) / static_cast<float>(uSegments);
            const float v0 = static_cast<float>(j) / static_cast<float>(vSegments);
            const float v1 = static_cast<float>(j + 1) / static_cast<float>(vSegments);
            const auto p00 = addScaled(origin, u, u0, v, v0);
            const auto p10 = addScaled(origin, u, u1, v, v0);
            const auto p01 = addScaled(origin, u, u0, v, v1);
            const auto p11 = addScaled(origin, u, u1, v, v1);
            const auto firstIndex = static_cast<std::uint32_t>(triangles.size());
            triangles.push_back(makeTriangle(p00, p10, p11, firstIndex));
            triangles.push_back(makeTriangle(p00, p11, p01, firstIndex + 1));
        }
    }
}

std::vector<TrianglePrimitive> makeTissueGrid(const int n, const float size)
{
    std::vector<TrianglePrimitive> triangles;
    triangles.reserve(static_cast<std::size_t>(std::max(0, n - 1) * std::max(0, n - 1) * 2));
    if (n < 2)
    {
        return triangles;
    }

    const float half = size * 0.5f;
    for (int z = 0; z < n - 1; ++z)
    {
        for (int x = 0; x < n - 1; ++x)
        {
            const float x0 = -half + size * static_cast<float>(x) / static_cast<float>(n - 1);
            const float x1 = -half + size * static_cast<float>(x + 1) / static_cast<float>(n - 1);
            const float z0 = -half + size * static_cast<float>(z) / static_cast<float>(n - 1);
            const float z1 = -half + size * static_cast<float>(z + 1) / static_cast<float>(n - 1);
            const auto p00 = makeVertex(x0, 0.0f, z0);
            const auto p10 = makeVertex(x1, 0.0f, z0);
            const auto p01 = makeVertex(x0, 0.0f, z1);
            const auto p11 = makeVertex(x1, 0.0f, z1);
            const auto firstIndex = static_cast<std::uint32_t>(triangles.size());
            triangles.push_back(makeTriangle(p00, p10, p11, firstIndex));
            triangles.push_back(makeTriangle(p00, p11, p01, firstIndex + 1));
        }
    }
    return triangles;
}

std::vector<TrianglePrimitive> makeBladeBox(
    const float length,
    const float height,
    const float thickness,
    const int segmentsX,
    const int segmentsY,
    const int segmentsZ)
{
    std::vector<TrianglePrimitive> triangles;
    triangles.reserve(static_cast<std::size_t>(
        4 * segmentsX * segmentsY + 4 * segmentsX * segmentsZ + 4 * segmentsY * segmentsZ));

    const float hx = length * 0.5f;
    const float hy = height * 0.5f;
    const float hz = thickness * 0.5f;
    addPlane(triangles, makeVertex(-hx, -hy, -hz), makeVertex(length, 0, 0), makeVertex(0, height, 0), segmentsX, segmentsY);
    addPlane(triangles, makeVertex(-hx, -hy, hz), makeVertex(length, 0, 0), makeVertex(0, height, 0), segmentsX, segmentsY);
    addPlane(triangles, makeVertex(-hx, -hy, -hz), makeVertex(length, 0, 0), makeVertex(0, 0, thickness), segmentsX, segmentsZ);
    addPlane(triangles, makeVertex(-hx, hy, -hz), makeVertex(length, 0, 0), makeVertex(0, 0, thickness), segmentsX, segmentsZ);
    addPlane(triangles, makeVertex(-hx, -hy, -hz), makeVertex(0, height, 0), makeVertex(0, 0, thickness), segmentsY, segmentsZ);
    addPlane(triangles, makeVertex(hx, -hy, -hz), makeVertex(0, height, 0), makeVertex(0, 0, thickness), segmentsY, segmentsZ);
    return triangles;
}

void writeCsvHeader(std::ofstream& csv)
{
    csv << "step,wall_ms,gpu_kernel_ms,host_preparation_ms,backend_triangle_pack_ms,h2d_ms,"
           "device_allocation_ms,clear_grid_ms,counter_clear_ms,tissue_aabb_ms,tool_aabb_ms,"
           "insert_tissue_ms,insert_tool_ms,generate_pairs_ms,candidate_readback_ms,sort_unique_ms,"
           "sort_unique_host_ms,exact_contact_ms,contact_count_readback_ms,contact_download_ms,"
           "h2d_bytes,d2h_bytes,device_allocation_bytes,kernel_launch_count,cuda_memset_count,"
           "workspace_resize_count,raw_candidates,unique_candidates,contacts,grid_cells,active_mixed_cells,"
           "tissue_inserts,tool_inserts,max_tissue_cell_occupancy,max_tool_cell_occupancy,overflow,"
           "hash_dedupe_probe_overflow,host_validated_unique_candidates\n";
}

// --- Phase 11/12 bench-tool support: indexed-surface and point-cloud builders ---

struct IndexedSurfaceData
{
    std::vector<TriangleVertex> positions;
    std::vector<std::uint32_t>  indices;
};

// Convert a packed TrianglePrimitive vector to an indexed (positions + indices)
// representation by deduplicating identical vertex positions. Uses a tolerance
// of 0.0 because the geometry generators above produce exactly-equal vertex
// positions at shared corners (no floating-point reconstruction).
IndexedSurfaceData packedToIndexed(const std::vector<TrianglePrimitive>& packed)
{
    IndexedSurfaceData out;
    out.indices.reserve(packed.size() * 3);
    // Map (x, y, z) bit-pattern triple -> vertex index, via bit_cast through
    // a 96-bit key. uint64 hash with the trailing 32 bits XORed in.
    struct Key { float x, y, z; };
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            std::uint32_t a, b, c;
            std::memcpy(&a, &k.x, 4);
            std::memcpy(&b, &k.y, 4);
            std::memcpy(&c, &k.z, 4);
            std::uint64_t h = static_cast<std::uint64_t>(a) * 0x9E3779B97F4A7C15ull;
            h ^= static_cast<std::uint64_t>(b) + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
            h ^= static_cast<std::uint64_t>(c) + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
            return static_cast<std::size_t>(h);
        }
    };
    struct KeyEq {
        bool operator()(const Key& a, const Key& b) const noexcept {
            return a.x == b.x && a.y == b.y && a.z == b.z;
        }
    };
    std::unordered_map<Key, std::uint32_t, KeyHash, KeyEq> dedupe;
    dedupe.reserve(packed.size() * 3);

    auto pushVertex = [&](const TriangleVertex& v) -> std::uint32_t {
        const Key k { v.x, v.y, v.z };
        auto it = dedupe.find(k);
        if (it != dedupe.end()) return it->second;
        const auto idx = static_cast<std::uint32_t>(out.positions.size());
        out.positions.push_back(v);
        dedupe.emplace(k, idx);
        return idx;
    };

    for (const auto& tri : packed)
    {
        out.indices.push_back(pushVertex(tri.p0));
        out.indices.push_back(pushVertex(tri.p1));
        out.indices.push_back(pushVertex(tri.p2));
    }
    return out;
}

// Shared bench-leg builders (were per-leg copy-pasted blocks).
TriangleIndexedSurface makeIndexedSurface(
    const IndexedSurfaceData& indexed,
    const std::size_t triangleCount,
    const std::uint64_t surfaceId)
{
    TriangleIndexedSurface surface;
    surface.positions = indexed.positions.data();
    surface.devicePositions = nullptr;
    surface.vertexCount = static_cast<std::uint32_t>(indexed.positions.size());
    surface.triangleIndices = indexed.indices.data();
    surface.triangleCount = static_cast<std::uint32_t>(triangleCount);
    surface.surfaceId = surfaceId;
    surface.topologyVersion = 1;
    return surface;
}

FeatureBasedProximityConfig makeBenchFbpConfig(const DenseGridConfig& config)
{
    FeatureBasedProximityConfig fbpConfig;
    fbpConfig.contactDistance = config.contactDistance;
    fbpConfig.computeBarycentrics = true;
    fbpConfig.keepContactsOnDevice = false;
    fbpConfig.readContactCounter = true;
    fbpConfig.maxContacts = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_FBP_MAX_CONTACTS", 1000000));
    return fbpConfig;
}

SofaGpuCollision::backend::HashPrefixSumConfig makeBenchHashConfig()
{
    SofaGpuCollision::backend::HashPrefixSumConfig hashConfig;
    hashConfig.hashTableSize = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_HASH_TABLE_SIZE", 0));
    hashConfig.maxProbe = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_HASH_MAX_PROBE", 64));
    return hashConfig;
}

void writeFbpCsvHeader(std::ofstream& csv)
{
    csv << "step,wall_ms,gpu_kernel_ms,fbp_kernel_ms,h2d_bytes,d2h_bytes,"
           "kernel_launch_count,cuda_memset_count,raw_candidates,unique_candidates,"
           "emitted_contacts,vf_contacts,fv_contacts,ee_contacts,overflow\n";
}

void writeVtCsvHeader(std::ofstream& csv)
{
    csv << "step,wall_ms,gpu_kernel_ms,h2d_bytes,d2h_bytes,kernel_launch_count,cuda_memset_count,"
           "emitted_contacts,vf_contacts,overflow\n";
}

} // namespace

int main()
{
    const int steps = envInt("SOFA_BACKEND_BENCH_STEPS", 30);
    const int warmup = envInt("SOFA_BACKEND_BENCH_WARMUP", 5);
    const int tissueN = envInt("SOFA_LARGE_TISSUE_NX", 181);
    const int bladeSegmentsX = envInt("SOFA_LARGE_BLADE_SEGMENTS_X", 128);
    const int bladeSegmentsY = envInt("SOFA_LARGE_BLADE_SEGMENTS_Y", 24);
    const int bladeSegmentsZ = envInt("SOFA_LARGE_BLADE_SEGMENTS_Z", 4);

    auto tissue = makeTissueGrid(tissueN, envFloat("SOFA_BACKEND_TISSUE_SIZE", 12.0f));
    auto blade = makeBladeBox(
        envFloat("SOFA_BACKEND_BLADE_LENGTH", 5.5f),
        envFloat("SOFA_BACKEND_BLADE_HEIGHT", 0.8f),
        envFloat("SOFA_BACKEND_BLADE_THICKNESS", 0.16f),
        bladeSegmentsX,
        bladeSegmentsY,
        bladeSegmentsZ);

    DenseGridConfig config;
    config.gridMinX = envFloat("SOFA_GRID_MIN_X", -6.5f);
    config.gridMinY = envFloat("SOFA_GRID_MIN_Y", -0.7f);
    config.gridMinZ = envFloat("SOFA_GRID_MIN_Z", -6.5f);
    config.gridMaxX = envFloat("SOFA_GRID_MAX_X", 6.5f);
    config.gridMaxY = envFloat("SOFA_GRID_MAX_Y", 0.7f);
    config.gridMaxZ = envFloat("SOFA_GRID_MAX_Z", 6.5f);
    config.gridResolutionX = static_cast<std::uint32_t>(envInt("SOFA_GRID_RESOLUTION_X", 96));
    config.gridResolutionY = static_cast<std::uint32_t>(envInt("SOFA_GRID_RESOLUTION_Y", 8));
    config.gridResolutionZ = static_cast<std::uint32_t>(envInt("SOFA_GRID_RESOLUTION_Z", 96));
    config.contactDistance = envFloat("SOFA_CONTACT_DISTANCE", 0.03f);
    config.maxTissueTrianglesPerCell = static_cast<std::uint32_t>(envInt("SOFA_MAX_TISSUE_TRIANGLES_PER_CELL", 192));
    config.maxToolTrianglesPerCell = static_cast<std::uint32_t>(envInt("SOFA_MAX_TOOL_TRIANGLES_PER_CELL", 192));
    config.maxCandidatePairs = static_cast<std::uint32_t>(envInt("SOFA_MAX_CANDIDATE_PAIRS", 8000000));
    config.deduplicatePairs = envBool("SOFA_DEDUPLICATE_PAIRS", true);
    config.copyContactsToHost = envBool("SOFA_COPY_CONTACTS_TO_HOST", false);
    config.detailedProfiling = envBool("SOFA_GPU_DETAILED_PROFILING", true);
    config.usePinnedHostStaging = envBool("SOFA_USE_PINNED_HOST_STAGING", true);
    config.useGpuHashDedupe = envBool("SOFA_USE_GPU_HASH_DEDUPE", config.deduplicatePairs);
    config.canonicalPairEmission = envBool("SOFA_CANONICAL_PAIR_EMISSION", false);
    config.validateDedupeOnHost = envBool("SOFA_VALIDATE_DEDUPE_ON_HOST", false);

    const std::filesystem::path outputPath = [] {
        if (const char* raw = std::getenv("SOFA_BACKEND_BENCH_CSV"))
        {
            return std::filesystem::path(raw);
        }
        return std::filesystem::path("output/benchmark_logs/backend_dense_grid_benchmark.csv");
    }();
    std::filesystem::create_directories(outputPath.parent_path());
    std::ofstream csv(outputPath, std::ios::out | std::ios::trunc);
    csv << std::fixed << std::setprecision(9);
    writeCsvHeader(csv);

    int measured = 0;
    double wallTotalMs = 0.0;
    double gpuTotalMs = 0.0;
    std::uint64_t lastRawCandidates = 0;
    std::uint64_t lastUniqueCandidates = 0;
    std::uint64_t lastContacts = 0;
    std::uint64_t lastOverflow = 0;
    std::uint64_t lastHashProbeOverflow = 0;
    std::uint64_t lastHostValidatedUniqueCandidates = 0;

    for (int step = 0; step < steps + warmup; ++step)
    {
        std::vector<ExactContact> contacts;
        std::string diagnostic;
        SofaGpuCollision::backend::BackendExecutionStats stats;
        const auto start = std::chrono::steady_clock::now();
        const bool ok = SofaGpuCollision::backend::computeDenseGridTriangleContacts(
            tissue,
            blade,
            config,
            contacts,
            diagnostic,
            &stats);
        const double wallMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (!ok)
        {
            std::cerr << "backend dense-grid benchmark failed: " << diagnostic << "\n";
            return 2;
        }

        const bool included = step >= warmup;
        if (included)
        {
            ++measured;
            wallTotalMs += wallMs;
            gpuTotalMs += stats.gpuKernelMilliseconds;
            lastRawCandidates = stats.rawCandidateCount;
            lastUniqueCandidates = stats.uniqueCandidateCount;
            lastContacts = stats.outputContactCount;
            lastOverflow = stats.overflowCount;
            lastHashProbeOverflow = stats.hashDedupeProbeOverflowCount;
            lastHostValidatedUniqueCandidates = stats.hostValidatedUniqueCandidateCount;
        }

        csv << step << ','
            << wallMs << ','
            << stats.gpuKernelMilliseconds << ','
            << stats.hostPreparationMilliseconds << ','
            << stats.backendTrianglePackMilliseconds << ','
            << stats.hostToDeviceMilliseconds << ','
            << stats.deviceAllocationMilliseconds << ','
            << stats.denseGridClearMilliseconds << ','
            << stats.denseGridCounterClearMilliseconds << ','
            << stats.denseGridTissueAabbMilliseconds << ','
            << stats.denseGridToolAabbMilliseconds << ','
            << stats.denseGridInsertTissueMilliseconds << ','
            << stats.denseGridInsertToolMilliseconds << ','
            << stats.denseGridGeneratePairsMilliseconds << ','
            << stats.denseGridCandidateReadbackMilliseconds << ','
            << stats.denseGridSortUniqueMilliseconds << ','
            << stats.denseGridSortUniqueHostMilliseconds << ','
            << stats.denseGridExactContactMilliseconds << ','
            << stats.denseGridContactCountReadbackMilliseconds << ','
            << stats.denseGridContactDownloadMilliseconds << ','
            << stats.hostToDeviceBytes << ','
            << stats.deviceToHostBytes << ','
            << stats.deviceAllocationBytes << ','
            << stats.kernelLaunchCount << ','
            << stats.cudaMemsetCount << ','
            << stats.workspaceResizeCount << ','
            << stats.rawCandidateCount << ','
            << stats.uniqueCandidateCount << ','
            << stats.outputContactCount << ','
            << stats.gridCellCount << ','
            << stats.activeMixedCellCount << ','
            << stats.tissueInsertCount << ','
            << stats.toolInsertCount << ','
            << stats.maxTissueCellOccupancy << ','
            << stats.maxToolCellOccupancy << ','
            << stats.overflowCount << ','
            << stats.hashDedupeProbeOverflowCount << ','
            << stats.hostValidatedUniqueCandidateCount << '\n';
    }

    std::cout << std::fixed << std::setprecision(6)
              << "tissue_triangles=" << tissue.size() << '\n'
              << "blade_triangles=" << blade.size() << '\n'
              << "measured_steps=" << measured << '\n'
              << "wall_avg_ms=" << (measured > 0 ? wallTotalMs / measured : 0.0) << '\n'
              << "gpu_kernel_avg_ms=" << (measured > 0 ? gpuTotalMs / measured : 0.0) << '\n'
              << "raw_candidates=" << lastRawCandidates << '\n'
              << "unique_candidates=" << lastUniqueCandidates << '\n'
              << "contacts=" << lastContacts << '\n'
              << "overflow=" << lastOverflow << '\n'
              << "hash_dedupe_probe_overflow=" << lastHashProbeOverflow << '\n'
              << "host_validated_unique_candidates=" << lastHostValidatedUniqueCandidates << '\n'
              << "csv=" << outputPath.string() << '\n';

    // -----------------------------------------------------------------------
    // Phase 11: tri-tri feature-based proximity bench
    // -----------------------------------------------------------------------
    if (envBool("SOFA_BACKEND_BENCH_RUN_FBP", true))
    {
        std::cout << "\n--- bench: feature-based proximity (tri-tri) ---\n";
        const auto tissueIndexed = packedToIndexed(tissue);
        const auto bladeIndexed = packedToIndexed(blade);

        const TriangleIndexedSurface tissueSurface = makeIndexedSurface(tissueIndexed, tissue.size(), 0xA001ull);
        const TriangleIndexedSurface bladeSurface = makeIndexedSurface(bladeIndexed, blade.size(), 0xA002ull);

        FeatureBasedProximityConfig fbpConfig;
        fbpConfig.contactDistance = config.contactDistance;
        fbpConfig.emitOnePerPair = true;
        fbpConfig.computeBarycentrics = true;
        fbpConfig.keepContactsOnDevice = false;        // download so we can count
        fbpConfig.readContactCounter = true;            // visibility
        fbpConfig.maxContacts = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_FBP_MAX_CONTACTS", 1000000));

        const std::filesystem::path fbpPath = outputPath.parent_path() / (outputPath.stem().string() + "_fbp.csv");
        std::ofstream fbpCsv(fbpPath, std::ios::out | std::ios::trunc);
        fbpCsv << std::fixed << std::setprecision(9);
        writeFbpCsvHeader(fbpCsv);

        double fbpWallTotal = 0.0;
        double fbpKernelTotal = 0.0;
        int fbpMeasured = 0;
        std::uint64_t fbpLastContacts = 0;
        std::uint64_t fbpLastVf = 0, fbpLastFv = 0, fbpLastEe = 0;

        for (int step = 0; step < steps + warmup; ++step)
        {
            std::vector<ProximityContact> contacts;
            SofaGpuCollision::backend::BackendExecutionStats stats;
            FeatureBasedProximityStats proximityStats;
            std::string diagnostic;

            const auto start = std::chrono::steady_clock::now();
            const bool ok = SofaGpuCollision::backend::computeFeatureBasedProximityContacts(
                tissueSurface, bladeSurface, config, fbpConfig,
                contacts, &proximityStats, diagnostic, &stats);
            const double wallMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (!ok)
            {
                std::cerr << "FBP bench failed: " << diagnostic << "\n";
                return 3;
            }

            const bool included = step >= warmup;
            if (included)
            {
                ++fbpMeasured;
                fbpWallTotal += wallMs;
                fbpKernelTotal += stats.gpuKernelMilliseconds;
                fbpLastContacts = proximityStats.emittedContactCount;
                fbpLastVf = proximityStats.vfContactCount;
                fbpLastFv = proximityStats.fvContactCount;
                fbpLastEe = proximityStats.eeContactCount;
            }

            fbpCsv << step << ',' << wallMs << ',' << stats.gpuKernelMilliseconds << ','
                   << stats.featureBasedProximityKernelMilliseconds << ','
                   << stats.hostToDeviceBytes << ',' << stats.deviceToHostBytes << ','
                   << stats.kernelLaunchCount << ',' << stats.cudaMemsetCount << ','
                   << stats.rawCandidateCount << ',' << stats.uniqueCandidateCount << ','
                   << proximityStats.emittedContactCount << ','
                   << proximityStats.vfContactCount << ',' << proximityStats.fvContactCount << ','
                   << proximityStats.eeContactCount << ',' << stats.overflowCount << '\n';
        }

        std::cout << "fbp_measured_steps=" << fbpMeasured << '\n'
                  << "fbp_wall_avg_ms=" << (fbpMeasured > 0 ? fbpWallTotal / fbpMeasured : 0.0) << '\n'
                  << "fbp_gpu_kernel_avg_ms=" << (fbpMeasured > 0 ? fbpKernelTotal / fbpMeasured : 0.0) << '\n'
                  << "fbp_contacts=" << fbpLastContacts << '\n'
                  << "fbp_vf=" << fbpLastVf << " fv=" << fbpLastFv << " ee=" << fbpLastEe << '\n'
                  << "fbp_csv=" << fbpPath.string() << '\n';
    }

    // -----------------------------------------------------------------------
    // Phase 12: vertex-triangle feature-based proximity bench (cross-model)
    //
    // We treat the blade vertex set as a point cloud and the tissue triangle
    // mesh as the triangle surface. SurfaceIds differ so own-corner exclusion
    // stays OFF — every blade vertex tests against every nearby tissue tri.
    // -----------------------------------------------------------------------
    if (envBool("SOFA_BACKEND_BENCH_RUN_VT", true))
    {
        std::cout << "\n--- bench: vertex-triangle proximity (cross-model) ---\n";
        const auto tissueIndexed = packedToIndexed(tissue);
        const auto bladeIndexed = packedToIndexed(blade);

        TriangleIndexedSurface tissueSurface;
        tissueSurface.positions = tissueIndexed.positions.data();
        tissueSurface.devicePositions = nullptr;
        tissueSurface.vertexCount = static_cast<std::uint32_t>(tissueIndexed.positions.size());
        tissueSurface.triangleIndices = tissueIndexed.indices.data();
        tissueSurface.triangleCount = static_cast<std::uint32_t>(tissue.size());
        tissueSurface.surfaceId = 0xB001ull;
        tissueSurface.topologyVersion = 1;

        PointCloudSurface bladePoints;
        bladePoints.positions = bladeIndexed.positions.data();
        bladePoints.devicePositions = nullptr;
        bladePoints.pointCount = static_cast<std::uint32_t>(bladeIndexed.positions.size());
        bladePoints.surfaceId = 0xB002ull;  // different from tissue.surfaceId -> exclusion off
        bladePoints.topologyVersion = 0;

        FeatureBasedProximityConfig vtConfig;
        vtConfig.contactDistance = config.contactDistance;
        vtConfig.emitOnePerPair = true;
        vtConfig.computeBarycentrics = true;
        vtConfig.keepContactsOnDevice = false;
        vtConfig.readContactCounter = true;
        vtConfig.maxContacts = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_VT_MAX_CONTACTS", 1000000));

        const std::filesystem::path vtPath = outputPath.parent_path() / (outputPath.stem().string() + "_vt.csv");
        std::ofstream vtCsv(vtPath, std::ios::out | std::ios::trunc);
        vtCsv << std::fixed << std::setprecision(9);
        writeVtCsvHeader(vtCsv);

        double vtWallTotal = 0.0;
        double vtKernelTotal = 0.0;
        int vtMeasured = 0;
        std::uint64_t vtLastContacts = 0;
        std::uint64_t vtLastVf = 0;

        for (int step = 0; step < steps + warmup; ++step)
        {
            std::vector<ProximityContact> contacts;
            SofaGpuCollision::backend::BackendExecutionStats stats;
            FeatureBasedProximityStats proximityStats;
            std::string diagnostic;

            const auto start = std::chrono::steady_clock::now();
            const bool ok = SofaGpuCollision::backend::computeFeatureBasedVertexTriangleContacts(
                bladePoints, tissueSurface, config, vtConfig,
                contacts, &proximityStats, diagnostic, &stats);
            const double wallMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (!ok)
            {
                std::cerr << "V-t bench failed: " << diagnostic << "\n";
                return 4;
            }

            const bool included = step >= warmup;
            if (included)
            {
                ++vtMeasured;
                vtWallTotal += wallMs;
                vtKernelTotal += stats.gpuKernelMilliseconds;
                vtLastContacts = proximityStats.emittedContactCount;
                vtLastVf = proximityStats.vfContactCount;
            }

            vtCsv << step << ',' << wallMs << ',' << stats.gpuKernelMilliseconds << ','
                  << stats.hostToDeviceBytes << ',' << stats.deviceToHostBytes << ','
                  << stats.kernelLaunchCount << ',' << stats.cudaMemsetCount << ','
                  << proximityStats.emittedContactCount << ','
                  << proximityStats.vfContactCount << ',' << stats.overflowCount << '\n';
        }

        std::cout << "vt_measured_steps=" << vtMeasured << '\n'
                  << "vt_wall_avg_ms=" << (vtMeasured > 0 ? vtWallTotal / vtMeasured : 0.0) << '\n'
                  << "vt_gpu_kernel_avg_ms=" << (vtMeasured > 0 ? vtKernelTotal / vtMeasured : 0.0) << '\n'
                  << "vt_contacts=" << vtLastContacts << '\n'
                  << "vt_vf_contacts=" << vtLastVf << '\n'
                  << "vt_csv=" << vtPath.string() << '\n';
    }

    // -----------------------------------------------------------------------
    // EXPERIMENTAL: hash + prefix-sum broad cull (must match the tri-tri FBP
    // phase contact count exactly — same geometry, same narrow kernel).
    // -----------------------------------------------------------------------
    if (envBool("SOFA_BACKEND_BENCH_RUN_HASH", true))
    {
        std::cout << "\n--- bench: hash + prefix-sum (tri-tri) ---\n";
        const auto tissueIndexed = packedToIndexed(tissue);
        const auto bladeIndexed = packedToIndexed(blade);

        const TriangleIndexedSurface tissueSurface = makeIndexedSurface(tissueIndexed, tissue.size(), 0xC001ull);
        const TriangleIndexedSurface bladeSurface = makeIndexedSurface(bladeIndexed, blade.size(), 0xC002ull);

        const SofaGpuCollision::backend::HashPrefixSumConfig hashConfig = makeBenchHashConfig();

        const FeatureBasedProximityConfig fbpConfig = makeBenchFbpConfig(config);

        const std::filesystem::path hashPath = outputPath.parent_path() / (outputPath.stem().string() + "_hash.csv");
        std::ofstream hashCsv(hashPath, std::ios::out | std::ios::trunc);
        hashCsv << std::fixed << std::setprecision(9);
        hashCsv << "step,wall_ms,gpu_kernel_ms,h2d_bytes,d2h_bytes,kernel_launch_count,cuda_memset_count,"
                   "hash_table_size,occupied_slots,raw_pairs,unique_pairs,emitted_contacts,vf,fv,ee,"
                   "bucket_overflow,probe_overflow\n";

        double hashWallTotal = 0.0, hashKernelTotal = 0.0;
        int hashMeasured = 0;
        std::uint64_t hashLastContacts = 0, hashLastUnique = 0, hashLastOccupied = 0, hashLastTable = 0;
        std::uint64_t hashLastVf = 0, hashLastFv = 0, hashLastEe = 0, hashLastBucketOf = 0, hashLastProbeOf = 0;

        for (int step = 0; step < steps + warmup; ++step)
        {
            std::vector<ProximityContact> hashContacts;
            SofaGpuCollision::backend::BackendExecutionStats stats;
            FeatureBasedProximityStats proximityStats;
            SofaGpuCollision::backend::HashPrefixSumStats hashStatsOut;
            std::string diagnostic;

            const auto start = std::chrono::steady_clock::now();
            const bool ok = SofaGpuCollision::backend::computeHashPrefixSumProximityContacts(
                tissueSurface, bladeSurface, config, hashConfig, fbpConfig,
                hashContacts, &proximityStats, &hashStatsOut, diagnostic, &stats);
            const double wallMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (!ok) { std::cerr << "Hash bench failed: " << diagnostic << "\n"; return 5; }

            if (step >= warmup)
            {
                ++hashMeasured;
                hashWallTotal += wallMs;
                hashKernelTotal += stats.gpuKernelMilliseconds;
                hashLastContacts = proximityStats.emittedContactCount;
                hashLastUnique = hashStatsOut.uniquePairCount;
                hashLastOccupied = hashStatsOut.occupiedSlotCount;
                hashLastTable = hashStatsOut.hashTableSize;
                hashLastVf = proximityStats.vfContactCount;
                hashLastFv = proximityStats.fvContactCount;
                hashLastEe = proximityStats.eeContactCount;
                hashLastBucketOf = hashStatsOut.bucketOverflowCount;
                hashLastProbeOf = hashStatsOut.hashProbeOverflowCount;
            }

            hashCsv << step << ',' << wallMs << ',' << stats.gpuKernelMilliseconds << ','
                    << stats.hostToDeviceBytes << ',' << stats.deviceToHostBytes << ','
                    << stats.kernelLaunchCount << ',' << stats.cudaMemsetCount << ','
                    << hashStatsOut.hashTableSize << ',' << hashStatsOut.occupiedSlotCount << ','
                    << hashStatsOut.rawPairCount << ',' << hashStatsOut.uniquePairCount << ','
                    << proximityStats.emittedContactCount << ','
                    << proximityStats.vfContactCount << ',' << proximityStats.fvContactCount << ','
                    << proximityStats.eeContactCount << ','
                    << hashStatsOut.bucketOverflowCount << ',' << hashStatsOut.hashProbeOverflowCount << '\n';
        }

        std::cout << "hash_measured_steps=" << hashMeasured << '\n'
                  << "hash_wall_avg_ms=" << (hashMeasured > 0 ? hashWallTotal / hashMeasured : 0.0) << '\n'
                  << "hash_gpu_kernel_avg_ms=" << (hashMeasured > 0 ? hashKernelTotal / hashMeasured : 0.0) << '\n'
                  << "hash_table_size=" << hashLastTable << '\n'
                  << "hash_occupied_slots=" << hashLastOccupied << '\n'
                  << "hash_unique_pairs=" << hashLastUnique << '\n'
                  << "hash_contacts=" << hashLastContacts << " (vf=" << hashLastVf
                  << " fv=" << hashLastFv << " ee=" << hashLastEe << ")\n"
                  << "hash_bucket_overflow=" << hashLastBucketOf
                  << " hash_probe_overflow=" << hashLastProbeOf << '\n'
                  << "hash_csv=" << hashPath.string() << '\n'
                  << "CORRECTNESS: hash_contacts (" << hashLastContacts
                  << ") should equal fbp_contacts above.\n";
    }

    if (envBool("SOFA_BACKEND_BENCH_RUN_SIMPLE_HASH", true))
    {
        std::cout << "\n--- bench: simple direct-bucket hash (tri-tri, 4th way) ---\n";
        const auto tissueIndexed = packedToIndexed(tissue);
        const auto bladeIndexed = packedToIndexed(blade);

        const TriangleIndexedSurface tissueSurface = makeIndexedSurface(tissueIndexed, tissue.size(), 0xD001ull);
        const TriangleIndexedSurface bladeSurface = makeIndexedSurface(bladeIndexed, blade.size(), 0xD002ull);

        const SofaGpuCollision::backend::HashPrefixSumConfig hashConfig = makeBenchHashConfig();

        const FeatureBasedProximityConfig fbpConfig = makeBenchFbpConfig(config);

        const std::filesystem::path simplePath = outputPath.parent_path() / (outputPath.stem().string() + "_simplehash.csv");
        std::ofstream simpleCsv(simplePath, std::ios::out | std::ios::trunc);
        simpleCsv << std::fixed << std::setprecision(9);
        simpleCsv << "step,wall_ms,gpu_kernel_ms,h2d_bytes,d2h_bytes,kernel_launch_count,cuda_memset_count,"
                     "hash_table_size,active_slots,raw_pairs,unique_pairs,emitted_contacts,vf,fv,ee,"
                     "bucket_overflow,probe_overflow\n";

        double simpleWallTotal = 0.0, simpleKernelTotal = 0.0;
        int simpleMeasured = 0;
        std::uint64_t simpleLastContacts = 0, simpleLastUnique = 0, simpleLastActive = 0, simpleLastTable = 0;
        std::uint64_t simpleLastVf = 0, simpleLastFv = 0, simpleLastEe = 0, simpleLastBucketOf = 0, simpleLastProbeOf = 0;

        for (int step = 0; step < steps + warmup; ++step)
        {
            std::vector<ProximityContact> simpleContacts;
            SofaGpuCollision::backend::BackendExecutionStats stats;
            FeatureBasedProximityStats proximityStats;
            SofaGpuCollision::backend::HashPrefixSumStats hashStatsOut;
            std::string diagnostic;

            const auto start = std::chrono::steady_clock::now();
            const bool ok = SofaGpuCollision::backend::computeSimpleHashProximityContacts(
                tissueSurface, bladeSurface, config, hashConfig, fbpConfig,
                simpleContacts, &proximityStats, &hashStatsOut, diagnostic, &stats);
            const double wallMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (!ok) { std::cerr << "Simple-hash bench failed: " << diagnostic << "\n"; return 6; }

            if (step >= warmup)
            {
                ++simpleMeasured;
                simpleWallTotal += wallMs;
                simpleKernelTotal += stats.gpuKernelMilliseconds;
                simpleLastContacts = proximityStats.emittedContactCount;
                simpleLastUnique = hashStatsOut.uniquePairCount;
                simpleLastActive = hashStatsOut.occupiedSlotCount;
                simpleLastTable = hashStatsOut.hashTableSize;
                simpleLastVf = proximityStats.vfContactCount;
                simpleLastFv = proximityStats.fvContactCount;
                simpleLastEe = proximityStats.eeContactCount;
                simpleLastBucketOf = hashStatsOut.bucketOverflowCount;
                simpleLastProbeOf = hashStatsOut.hashProbeOverflowCount;
            }

            simpleCsv << step << ',' << wallMs << ',' << stats.gpuKernelMilliseconds << ','
                      << stats.hostToDeviceBytes << ',' << stats.deviceToHostBytes << ','
                      << stats.kernelLaunchCount << ',' << stats.cudaMemsetCount << ','
                      << hashStatsOut.hashTableSize << ',' << hashStatsOut.occupiedSlotCount << ','
                      << hashStatsOut.rawPairCount << ',' << hashStatsOut.uniquePairCount << ','
                      << proximityStats.emittedContactCount << ','
                      << proximityStats.vfContactCount << ',' << proximityStats.fvContactCount << ','
                      << proximityStats.eeContactCount << ','
                      << hashStatsOut.bucketOverflowCount << ',' << hashStatsOut.hashProbeOverflowCount << '\n';
        }

        std::cout << "simplehash_measured_steps=" << simpleMeasured << '\n'
                  << "simplehash_wall_avg_ms=" << (simpleMeasured > 0 ? simpleWallTotal / simpleMeasured : 0.0) << '\n'
                  << "simplehash_gpu_kernel_avg_ms=" << (simpleMeasured > 0 ? simpleKernelTotal / simpleMeasured : 0.0) << '\n'
                  << "simplehash_table_size=" << simpleLastTable << '\n'
                  << "simplehash_active_slots=" << simpleLastActive << '\n'
                  << "simplehash_unique_pairs=" << simpleLastUnique << '\n'
                  << "simplehash_contacts=" << simpleLastContacts << " (vf=" << simpleLastVf
                  << " fv=" << simpleLastFv << " ee=" << simpleLastEe << ")\n"
                  << "simplehash_bucket_overflow=" << simpleLastBucketOf
                  << " simplehash_probe_overflow=" << simpleLastProbeOf << '\n'
                  << "simplehash_csv=" << simplePath.string() << '\n'
                  << "CORRECTNESS: simplehash_contacts (" << simpleLastContacts
                  << ") should equal fbp_contacts above (zero bucket_overflow).\n";
    }

    if (envBool("SOFA_BACKEND_BENCH_RUN_SORTED_GRID", true))
    {
        std::cout << "\n--- bench: sorted-grid tiled binning (tri-tri, 5th way) ---\n";
        const auto tissueIndexed = packedToIndexed(tissue);
        const auto bladeIndexed = packedToIndexed(blade);

        const TriangleIndexedSurface tissueSurface = makeIndexedSurface(tissueIndexed, tissue.size(), 0xE001ull);
        const TriangleIndexedSurface bladeSurface = makeIndexedSurface(bladeIndexed, blade.size(), 0xE002ull);

        SofaGpuCollision::backend::SortedGridConfig sortedConfig;
        sortedConfig.useCubRadixSort = envBool("SOFA_BACKEND_BENCH_SORTED_CUB", false);
        sortedConfig.usePairHashDedup = envBool("SOFA_BACKEND_BENCH_SORTED_PAIRHASH", false);

        const FeatureBasedProximityConfig fbpConfig = makeBenchFbpConfig(config);

        const std::filesystem::path sortedPath = outputPath.parent_path() / (outputPath.stem().string() + "_sortedgrid.csv");
        std::ofstream sortedCsv(sortedPath, std::ios::out | std::ios::trunc);
        sortedCsv << std::fixed << std::setprecision(9);
        sortedCsv << "step,wall_ms,gpu_kernel_ms,h2d_bytes,d2h_bytes,kernel_launch_count,cuda_memset_count,"
                     "bin_count,incidences,mixed_cells,unique_pairs,emitted_contacts,vf,fv,ee,"
                     "incidence_overflow,pair_overflow\n";

        double sortedWallTotal = 0.0, sortedKernelTotal = 0.0;
        int sortedMeasured = 0;
        std::uint64_t sortedLastContacts = 0, sortedLastUnique = 0, sortedLastMixed = 0, sortedLastIncidences = 0;
        std::uint64_t sortedLastVf = 0, sortedLastFv = 0, sortedLastEe = 0, sortedLastIncOf = 0, sortedLastPairOf = 0;
        std::uint64_t sortedLastVerify = 0;

        for (int step = 0; step < steps + warmup; ++step)
        {
            std::vector<ProximityContact> sortedContacts;
            SofaGpuCollision::backend::BackendExecutionStats stats;
            FeatureBasedProximityStats proximityStats;
            SofaGpuCollision::backend::SortedGridStats sortedStatsOut;
            std::string diagnostic;

            const auto start = std::chrono::steady_clock::now();
            const bool ok = SofaGpuCollision::backend::computeSortedGridProximityContacts(
                tissueSurface, bladeSurface, config, sortedConfig, fbpConfig,
                sortedContacts, &proximityStats, &sortedStatsOut, diagnostic, &stats);
            const double wallMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (!ok) { std::cerr << "Sorted-grid bench failed: " << diagnostic << "\n"; return 7; }

            if (step >= warmup)
            {
                ++sortedMeasured;
                sortedWallTotal += wallMs;
                sortedKernelTotal += stats.gpuKernelMilliseconds;
                sortedLastContacts = proximityStats.emittedContactCount;
                sortedLastUnique = sortedStatsOut.uniquePairCount;
                sortedLastMixed = sortedStatsOut.mixedCellCount;
                sortedLastIncidences = sortedStatsOut.incidenceCount;
                sortedLastVf = proximityStats.vfContactCount;
                sortedLastFv = proximityStats.fvContactCount;
                sortedLastEe = proximityStats.eeContactCount;
                sortedLastIncOf = sortedStatsOut.incidenceOverflowCount;
                sortedLastPairOf = sortedStatsOut.pairOverflowCount;
                sortedLastVerify = stats.hashDedupeProbeOverflowCount;
            }

            sortedCsv << step << ',' << wallMs << ',' << stats.gpuKernelMilliseconds << ','
                      << stats.hostToDeviceBytes << ',' << stats.deviceToHostBytes << ','
                      << stats.kernelLaunchCount << ',' << stats.cudaMemsetCount << ','
                      << sortedStatsOut.binCount << ',' << sortedStatsOut.incidenceCount << ','
                      << sortedStatsOut.mixedCellCount << ',' << sortedStatsOut.uniquePairCount << ','
                      << proximityStats.emittedContactCount << ','
                      << proximityStats.vfContactCount << ',' << proximityStats.fvContactCount << ','
                      << proximityStats.eeContactCount << ','
                      << sortedStatsOut.incidenceOverflowCount << ',' << sortedStatsOut.pairOverflowCount << '\n';
        }

        std::cout << "sortedgrid_engine=" << (sortedConfig.useCubRadixSort ? "cub_radix" : "counting") << '\n'
                  << "sortedgrid_dedup=" << (sortedConfig.usePairHashDedup ? "pair_hash" : "home_cell") << '\n'
                  << "sortedgrid_measured_steps=" << sortedMeasured << '\n'
                  << "sortedgrid_wall_avg_ms=" << (sortedMeasured > 0 ? sortedWallTotal / sortedMeasured : 0.0) << '\n'
                  << "sortedgrid_gpu_kernel_avg_ms=" << (sortedMeasured > 0 ? sortedKernelTotal / sortedMeasured : 0.0) << '\n'
                  << "sortedgrid_incidences=" << sortedLastIncidences << '\n'
                  << "sortedgrid_mixed_cells=" << sortedLastMixed << '\n'
                  << "sortedgrid_unique_pairs=" << sortedLastUnique << '\n'
                  << "sortedgrid_contacts=" << sortedLastContacts << " (vf=" << sortedLastVf
                  << " fv=" << sortedLastFv << " ee=" << sortedLastEe << ")\n"
                  << "sortedgrid_incidence_overflow=" << sortedLastIncOf
                  << " sortedgrid_pair_overflow=" << sortedLastPairOf << '\n'
                  << "sortedgrid_verify_violations=" << sortedLastVerify
                  << " (0 unless SOFA_SORTED_GRID_VERIFY=1)\n"
                  << "sortedgrid_csv=" << sortedPath.string() << '\n'
                  << "CORRECTNESS: sortedgrid_contacts (" << sortedLastContacts
                  << ") should equal fbp_contacts above (unique_pairs may be lower in home-cell mode).\n";
    }

    if (envBool("SOFA_BACKEND_BENCH_RUN_BIGCELL", true))
    {
        std::cout << "\n--- bench: big-cell fused generation + narrow phase (tri-tri, 6th way) ---\n";
        const auto tissueIndexed = packedToIndexed(tissue);
        const auto bladeIndexed = packedToIndexed(blade);

        const TriangleIndexedSurface tissueSurface = makeIndexedSurface(tissueIndexed, tissue.size(), 0xF001ull);
        const TriangleIndexedSurface bladeSurface = makeIndexedSurface(bladeIndexed, blade.size(), 0xF002ull);

        SofaGpuCollision::backend::BigCellConfig bigConfig;
        bigConfig.bigCellFactor = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_BIGCELL_FACTOR", 2));
        bigConfig.toolTileCapacity = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_BIGCELL_TILE", 256));
        bigConfig.useHashTableBuild = envBool("SOFA_BACKEND_BENCH_BIGCELL_HASH_BUILD", false);
        bigConfig.hashSlotsPerBigCell = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_BIGCELL_HASH_SLOTS", 1024));
        bigConfig.sharedBuildMode = static_cast<std::uint32_t>(envInt("SOFA_BACKEND_BENCH_BIGCELL_SHARED_BUILD", 1));
        bigConfig.profileFusedInternals = envBool("SOFA_BACKEND_BENCH_BIGCELL_PROFILE_INTERNALS", false);

        const FeatureBasedProximityConfig fbpConfig = makeBenchFbpConfig(config);

        const std::filesystem::path bigPath = outputPath.parent_path() / (outputPath.stem().string() + "_bigcell.csv");
        std::ofstream bigCsv(bigPath, std::ios::out | std::ios::trunc);
        bigCsv << std::fixed << std::setprecision(9);
        bigCsv << "step,wall_ms,gpu_kernel_ms,h2d_bytes,d2h_bytes,kernel_launch_count,cuda_memset_count,"
                  "big_cells,entries,mixed_big_cells,pairs_tested,emitted_contacts,vf,fv,ee,"
                  "entry_overflow,build_overflow,shared_spill,total_overflow,"
                  "event_reset_ms,event_build_clear_ms,event_marker_gap_ms,event_first_count_or_insert_ms,"
                  "event_second_count_or_insert_ms,event_scan_ms,event_first_fill_ms,event_second_fill_ms,"
                  "event_mixed_ms,event_proximity_reset_ms,event_fused_ms,event_total_ms,"
                  "prod_grid_blocks,prod_block_threads,prod_registers_per_thread,prod_static_shared_bytes,"
                  "prod_local_bytes_per_thread,prod_max_threads_per_block,prod_active_blocks_per_sm,"
                  "device_sms,device_max_threads_per_sm,device_warp_size,device_clock_rate_khz,prod_theoretical_occupancy_pct,"
                  "profile_registers_per_thread,profile_static_shared_bytes,profile_local_bytes_per_thread,"
                  "profile_active_blocks_per_sm,profile_theoretical_occupancy_pct,"
                  "profile_big_cells,profile_tiles,profile_tool_entries_staged,profile_tissue_entries_visited,"
                  "profile_small_cell_pair_visits,profile_inflated_aabb_rejects,profile_home_cell_rejects,"
                  "profile_raw_aabb_rejects,profile_fbp_calls,profile_fbp_no_contact,"
                  "profile_tile_setup_block_cycles,profile_bin_prefix_block_cycles,profile_tool_gather_block_cycles,"
                  "profile_tissue_sweep_block_cycles,profile_inflated_aabb_thread_cycles,"
                  "profile_home_cell_thread_cycles,profile_raw_aabb_thread_cycles,profile_fbp_thread_cycles,"
                  "profile_tool_sort_scatter_thread_cycles,profile_tool_data_load_thread_cycles,"
                  "profile_contact_emit_thread_cycles\n";

        double bigWallTotal = 0.0, bigKernelTotal = 0.0;
        int bigMeasured = 0;
        std::uint64_t bigLastContacts = 0, bigLastPairs = 0, bigLastMixed = 0, bigLastEntries = 0;
        std::uint64_t bigLastVf = 0, bigLastFv = 0, bigLastEe = 0, bigLastEntryOf = 0, bigLastBuildOf = 0;
        std::uint64_t bigLastSharedSpill = 0, bigLastTotalOverflow = 0;
        SofaGpuCollision::backend::BigCellStats bigLastStats {}, bigTimingTotals {};

        for (int step = 0; step < steps + warmup; ++step)
        {
            std::vector<ProximityContact> bigContacts;
            SofaGpuCollision::backend::BackendExecutionStats stats;
            FeatureBasedProximityStats proximityStats;
            SofaGpuCollision::backend::BigCellStats bigStatsOut;
            std::string diagnostic;

            const auto start = std::chrono::steady_clock::now();
            const bool ok = SofaGpuCollision::backend::computeBigCellFusedProximityContacts(
                tissueSurface, bladeSurface, config, bigConfig, fbpConfig,
                bigContacts, &proximityStats, &bigStatsOut, diagnostic, &stats);
            const double wallMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (!ok) { std::cerr << "Big-cell bench failed: " << diagnostic << "\n"; return 8; }

            if (step >= warmup)
            {
                ++bigMeasured;
                bigWallTotal += wallMs;
                bigKernelTotal += stats.gpuKernelMilliseconds;
                bigLastContacts = proximityStats.emittedContactCount;
                bigLastPairs = bigStatsOut.pairsTestedCount;
                bigLastMixed = bigStatsOut.mixedBigCellCount;
                bigLastEntries = bigStatsOut.entryCount;
                bigLastVf = proximityStats.vfContactCount;
                bigLastFv = proximityStats.fvContactCount;
                bigLastEe = proximityStats.eeContactCount;
                bigLastEntryOf = bigStatsOut.entryOverflowCount;
                bigLastBuildOf = bigStatsOut.buildOverflowCount;
                bigLastSharedSpill = bigStatsOut.sharedSpillCount;
                bigLastTotalOverflow = stats.overflowCount;
                bigLastStats = bigStatsOut;
                bigTimingTotals.resetMilliseconds += bigStatsOut.resetMilliseconds;
                bigTimingTotals.buildClearMilliseconds += bigStatsOut.buildClearMilliseconds;
                bigTimingTotals.eventMarkerGapMilliseconds += bigStatsOut.eventMarkerGapMilliseconds;
                bigTimingTotals.firstCountOrInsertMilliseconds += bigStatsOut.firstCountOrInsertMilliseconds;
                bigTimingTotals.secondCountOrInsertMilliseconds += bigStatsOut.secondCountOrInsertMilliseconds;
                bigTimingTotals.scanMilliseconds += bigStatsOut.scanMilliseconds;
                bigTimingTotals.firstFillMilliseconds += bigStatsOut.firstFillMilliseconds;
                bigTimingTotals.secondFillMilliseconds += bigStatsOut.secondFillMilliseconds;
                bigTimingTotals.mixedCellBuildMilliseconds += bigStatsOut.mixedCellBuildMilliseconds;
                bigTimingTotals.proximityResetMilliseconds += bigStatsOut.proximityResetMilliseconds;
                bigTimingTotals.fusedKernelMilliseconds += bigStatsOut.fusedKernelMilliseconds;
                bigTimingTotals.totalPipelineMilliseconds += bigStatsOut.totalPipelineMilliseconds;
            }

            bigCsv << step << ',' << wallMs << ',' << stats.gpuKernelMilliseconds << ','
                   << stats.hostToDeviceBytes << ',' << stats.deviceToHostBytes << ','
                   << stats.kernelLaunchCount << ',' << stats.cudaMemsetCount << ','
                   << bigStatsOut.bigCellCount << ',' << bigStatsOut.entryCount << ','
                   << bigStatsOut.mixedBigCellCount << ',' << bigStatsOut.pairsTestedCount << ','
                   << proximityStats.emittedContactCount << ','
                   << proximityStats.vfContactCount << ',' << proximityStats.fvContactCount << ','
                   << proximityStats.eeContactCount << ','
                   << bigStatsOut.entryOverflowCount << ',' << bigStatsOut.buildOverflowCount << ','
                   << bigStatsOut.sharedSpillCount << ',' << stats.overflowCount << ','
                   << bigStatsOut.resetMilliseconds << ',' << bigStatsOut.buildClearMilliseconds << ','
                   << bigStatsOut.eventMarkerGapMilliseconds << ','
                   << bigStatsOut.firstCountOrInsertMilliseconds << ',' << bigStatsOut.secondCountOrInsertMilliseconds << ','
                   << bigStatsOut.scanMilliseconds << ',' << bigStatsOut.firstFillMilliseconds << ','
                   << bigStatsOut.secondFillMilliseconds << ',' << bigStatsOut.mixedCellBuildMilliseconds << ','
                   << bigStatsOut.proximityResetMilliseconds << ',' << bigStatsOut.fusedKernelMilliseconds << ','
                   << bigStatsOut.totalPipelineMilliseconds << ','
                   << bigStatsOut.fusedGridBlocks << ',' << bigStatsOut.fusedBlockThreads << ','
                   << bigStatsOut.fusedRegistersPerThread << ',' << bigStatsOut.fusedStaticSharedBytes << ','
                   << bigStatsOut.fusedLocalBytesPerThread << ',' << bigStatsOut.fusedMaxThreadsPerBlock << ','
                   << bigStatsOut.fusedActiveBlocksPerSm << ',' << bigStatsOut.deviceMultiprocessorCount << ','
                   << bigStatsOut.deviceMaxThreadsPerSm << ',' << bigStatsOut.deviceWarpSize << ','
                   << bigStatsOut.deviceClockRateKHz << ',' << bigStatsOut.fusedTheoreticalOccupancyPercent << ','
                   << bigStatsOut.profiledRegistersPerThread << ',' << bigStatsOut.profiledStaticSharedBytes << ','
                   << bigStatsOut.profiledLocalBytesPerThread << ',' << bigStatsOut.profiledActiveBlocksPerSm << ','
                   << bigStatsOut.profiledTheoreticalOccupancyPercent << ','
                   << bigStatsOut.profiledBigCellIterations << ',' << bigStatsOut.profiledTileIterations << ','
                   << bigStatsOut.profiledToolEntriesStaged << ',' << bigStatsOut.profiledTissueEntriesVisited << ','
                   << bigStatsOut.profiledSmallCellPairVisits << ',' << bigStatsOut.profiledInflatedAabbRejects << ','
                   << bigStatsOut.profiledHomeCellRejects << ',' << bigStatsOut.profiledRawAabbRejects << ','
                   << bigStatsOut.profiledFbpCalls << ',' << bigStatsOut.profiledFbpNoContact << ','
                   << bigStatsOut.profiledTileSetupBlockCycles << ',' << bigStatsOut.profiledBinPrefixBlockCycles << ','
                   << bigStatsOut.profiledToolGatherBlockCycles << ',' << bigStatsOut.profiledTissueSweepBlockCycles << ','
                   << bigStatsOut.profiledInflatedAabbThreadCycles << ',' << bigStatsOut.profiledHomeCellThreadCycles << ','
                   << bigStatsOut.profiledRawAabbThreadCycles << ',' << bigStatsOut.profiledFbpThreadCycles << ','
                   << bigStatsOut.profiledToolSortScatterThreadCycles << ','
                   << bigStatsOut.profiledToolDataLoadThreadCycles << ','
                   << bigStatsOut.profiledContactEmitThreadCycles << '\n';
        }

        std::cout << "bigcell_build=" << (bigConfig.useHashTableBuild ? "hash_table" : "csr") << '\n'
                  << "bigcell_shared_build=" << (bigConfig.sharedBuildMode == 1u ? "shared_hash"
                                                 : bigConfig.sharedBuildMode == 2u ? "shared_sort" : "off") << '\n'
                  << "bigcell_factor=" << bigConfig.bigCellFactor << '\n'
                  << "bigcell_tool_tile=" << bigConfig.toolTileCapacity << '\n'
                  << "bigcell_measured_steps=" << bigMeasured << '\n'
                  << "bigcell_wall_avg_ms=" << (bigMeasured > 0 ? bigWallTotal / bigMeasured : 0.0) << '\n'
                  << "bigcell_gpu_kernel_avg_ms=" << (bigMeasured > 0 ? bigKernelTotal / bigMeasured : 0.0) << '\n'
                  << "bigcell_profile_internals=" << (bigConfig.profileFusedInternals ? 1 : 0) << '\n'
                  << "bigcell_event_reset_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.resetMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_build_clear_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.buildClearMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_marker_gap_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.eventMarkerGapMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_first_count_or_insert_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.firstCountOrInsertMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_second_count_or_insert_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.secondCountOrInsertMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_scan_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.scanMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_first_fill_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.firstFillMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_second_fill_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.secondFillMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_mixed_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.mixedCellBuildMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_proximity_reset_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.proximityResetMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_fused_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.fusedKernelMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_event_total_avg_ms=" << (bigMeasured > 0 ? bigTimingTotals.totalPipelineMilliseconds / bigMeasured : 0.0) << '\n'
                  << "bigcell_prod_launch=" << bigLastStats.fusedGridBlocks << "x" << bigLastStats.fusedBlockThreads << '\n'
                  << "bigcell_prod_resources=regs:" << bigLastStats.fusedRegistersPerThread
                  << " static_shared_bytes:" << bigLastStats.fusedStaticSharedBytes
                  << " local_bytes_per_thread:" << bigLastStats.fusedLocalBytesPerThread << '\n'
                  << "bigcell_prod_occupancy=active_blocks_per_sm:" << bigLastStats.fusedActiveBlocksPerSm
                  << " theoretical_pct:" << bigLastStats.fusedTheoreticalOccupancyPercent << '\n'
                  << "bigcell_profile_resources=regs:" << bigLastStats.profiledRegistersPerThread
                  << " static_shared_bytes:" << bigLastStats.profiledStaticSharedBytes
                  << " local_bytes_per_thread:" << bigLastStats.profiledLocalBytesPerThread << '\n'
                  << "bigcell_profile_occupancy=active_blocks_per_sm:" << bigLastStats.profiledActiveBlocksPerSm
                  << " theoretical_pct:" << bigLastStats.profiledTheoreticalOccupancyPercent << '\n'
                  << "bigcell_device=sms:" << bigLastStats.deviceMultiprocessorCount
                  << " max_threads_per_sm:" << bigLastStats.deviceMaxThreadsPerSm
                  << " warp_size:" << bigLastStats.deviceWarpSize
                  << " clock_rate_khz:" << bigLastStats.deviceClockRateKHz << '\n'
                  << "bigcell_profile_work=big_cells:" << bigLastStats.profiledBigCellIterations
                  << " tiles:" << bigLastStats.profiledTileIterations
                  << " tool_staged:" << bigLastStats.profiledToolEntriesStaged
                  << " tissue_visited:" << bigLastStats.profiledTissueEntriesVisited
                  << " pair_visits:" << bigLastStats.profiledSmallCellPairVisits << '\n'
                  << "bigcell_profile_rejects=inflated_aabb:" << bigLastStats.profiledInflatedAabbRejects
                  << " home_cell:" << bigLastStats.profiledHomeCellRejects
                  << " raw_aabb:" << bigLastStats.profiledRawAabbRejects
                  << " fbp_calls:" << bigLastStats.profiledFbpCalls
                  << " fbp_no_contact:" << bigLastStats.profiledFbpNoContact << '\n'
                  << "bigcell_profile_block_cycles=tile_setup:" << bigLastStats.profiledTileSetupBlockCycles
                  << " bin_prefix:" << bigLastStats.profiledBinPrefixBlockCycles
                  << " tool_gather:" << bigLastStats.profiledToolGatherBlockCycles
                  << " tissue_sweep:" << bigLastStats.profiledTissueSweepBlockCycles << '\n'
                  << "bigcell_profile_thread_cycles=inflated_aabb:" << bigLastStats.profiledInflatedAabbThreadCycles
                  << " home_cell:" << bigLastStats.profiledHomeCellThreadCycles
                  << " raw_aabb:" << bigLastStats.profiledRawAabbThreadCycles
                  << " fbp:" << bigLastStats.profiledFbpThreadCycles
                  << " tool_sort_scatter:" << bigLastStats.profiledToolSortScatterThreadCycles
                  << " tool_data_load:" << bigLastStats.profiledToolDataLoadThreadCycles
                  << " contact_emit:" << bigLastStats.profiledContactEmitThreadCycles << '\n'
                  << "bigcell_entries=" << bigLastEntries << '\n'
                  << "bigcell_mixed_big_cells=" << bigLastMixed << '\n'
                  << "bigcell_pairs_tested=" << bigLastPairs << '\n'
                  << "bigcell_contacts=" << bigLastContacts << " (vf=" << bigLastVf
                  << " fv=" << bigLastFv << " ee=" << bigLastEe << ")\n"
                  << "bigcell_entry_overflow=" << bigLastEntryOf
                  << " bigcell_build_overflow=" << bigLastBuildOf
                  << " bigcell_shared_spill=" << bigLastSharedSpill << '\n'
                  << "bigcell_total_overflow=" << bigLastTotalOverflow << '\n'
                  << "bigcell_csv=" << bigPath.string() << '\n'
                  << "CORRECTNESS: bigcell_contacts (" << bigLastContacts
                  << ") should equal fbp_contacts above; bigcell_pairs_tested should equal "
                  << "sortedgrid_unique_pairs (home-cell rule at the same small-cell granularity).\n";
    }

    return 0;
}
