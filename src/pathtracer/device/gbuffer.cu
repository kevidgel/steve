#include <integrator.cuh>
#include <geometry.cuh>
#include <material.cuh>

#include <optix_device.h>

__inline__ __device__ owl::vec2f getMotionVector(const owl::vec2i &pixelId, const RayInfo& prd) {
    const auto& prevCam = optixLaunchParams.prevCamera;
    const auto& cam = optixLaunchParams.camera;

    owl::vec2f cur = projectToScreen(cam, prd.hitInfo.p);
    owl::vec2f prev = projectToScreen(prevCam, prd.hitInfo.p);

    return cur - prev;
}

/// Geometry pass
OPTIX_RAYGEN_PROGRAM(GeometryPass)() {
    const auto &self = owl::getProgramData<GeometryPassData>();
    const owl::vec2i pixelId = owl::getLaunchIndex();
    const int pboOffset = pixelId.x + self.pboSize.x * pixelId.y;

    // Init prd
    RayInfo prd;
    prd.random.init(pboOffset, optixLaunchParams.frame.id);

    // Generate ray sample
    const owl::vec2f pixelOffset(prd.random(), prd.random());
    const owl::vec2f uv = (owl::vec2f(pixelId) + pixelOffset);
    const owl::vec2f lensSample = {prd.random(), prd.random()};
    owl::Ray ray = generateRay(optixLaunchParams.camera, uv.x, uv.y, lensSample);
    traceRay(optixLaunchParams.world, ray, prd);

    // Eval material
    MaterialResult mat;

    // Motion vectors for temporal reproj.
    GBufferInfo g;
    if (prd.hitInfo.intersectEvent == RayScattered) {
        g.motion = getMotionVector(pixelId, prd);
        getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd.hitInfo, mat);
    } else {
        g.motion = {-10, -10};
    }
    g.hitInfo = prd.hitInfo;
    g.mat = mat;

    // Set gBuffer
    // Also recycle spatial -> prev
    if (optixLaunchParams.curReservoir == 0) {
        optixLaunchParams.gBuffer0[pboOffset] = g;
        optixLaunchParams.reservoir1[pboOffset] = optixLaunchParams.spatialReservoir[pboOffset];
    } else {
        optixLaunchParams.gBuffer1[pboOffset] = g;
        optixLaunchParams.reservoir0[pboOffset] = optixLaunchParams.spatialReservoir[pboOffset];
    }


}

OPTIX_MISS_PROGRAM(Miss)() {
    const MissProgData &self = owl::getProgramData<MissProgData>();
    RayInfo &prd = owl::getPRD<RayInfo>();

    // owl::vec3f dir = optixGetWorldRayDirection();

    // For now, grey background
    prd.hitInfo.emitted = 0.f;
    prd.hitInfo.intersectEvent = RayMissed;
}
