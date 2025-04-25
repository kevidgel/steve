#include "lights.cuh"
#include "material.cuh"

#include <geometry.cuh>
#include <integrator.cuh>

#include <optix_device.h>

__inline__ __device__ bool isVisible(const LightSampleInfo &to, const HitInfo &from, RayInfo &record) {
    owl::vec3f dir = normalize(to.p - from.p);
    owl::Ray ray(from.p, dir, RAY_EPS, RAY_MAX);
    traceRay(optixLaunchParams.world, ray, record);
    return (length(record.hitInfo.p - to.p) < 0.001f && record.hitInfo.id == to.primI && dot(dir, from.sn) > 0.f);
}

__inline__ __device__ owl::vec3f evalTargetPdf(const LightSampleInfo &lightSample, const HitInfo &hit,
                                               const MaterialResult &mat, const ONB &onb, RayInfo &nextHit) {
    const owl::vec3f dirOut = normalize(lightSample.p - hit.p);
    const owl::vec3f sn_dirIn = onb.toLocal(-hit.dirIn);
    const owl::vec3f sn_dirOut = onb.toLocal(dirOut);
    const owl::vec3f sn_half = normalize(sn_dirIn + sn_dirOut);
    const float cosThetaL = fmaxf(dot(lightSample.gn, -dirOut), 0.f);

    owl::vec3f eval = evalMat(mat, sn_dirIn, sn_dirOut, sn_half);

    const bool V = isVisible(lightSample, hit, nextHit);
    const float dist2 = dot(lightSample.p - hit.p, lightSample.p - hit.p);
    const float G = cosThetaL / dist2;
    const owl::vec3f Le = V ? nextHit.hitInfo.emitted : 0.f;

    return eval * G * Le;
}

OPTIX_RAYGEN_PROGRAM(RayGen)() {
    const auto &self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelId = owl::getLaunchIndex();
    const int pboOffset = pixelId.x + self.pboSize.x * pixelId.y;
    Reservoir &curReservoir_ = optixLaunchParams.curReservoir == 0 ? optixLaunchParams.reservoir0[pboOffset]
                                                                   : optixLaunchParams.reservoir1[pboOffset];
    // Collect
    const GBufferInfo &gBuf = optixLaunchParams.curReservoir == 0 ? optixLaunchParams.gBuffer0[pboOffset]
                                                                  : optixLaunchParams.gBuffer1[pboOffset];

    const HitInfo &hit = gBuf.hitInfo;

    owl::vec3f Li(0.f);
    if (hit.intersectEvent == RayScattered) {
        // Setup reservoirs
        Reservoir prevReservoir, curReservoir;

        // Setup shading frame
        ONB onb(hit.sn);

        // Setup material result
        MaterialResult mat;
        getMatResult(optixLaunchParams.mats[hit.mat], hit, mat);

        // Next hit info
        RayInfo nextHit;
        nextHit.random.init(pboOffset, optixLaunchParams.frame.id);

        // Obtain previous frame pixel id
        prevReservoir.valid = false;
        auto p = static_cast<owl::vec2f>(pixelId) - gBuf.motion;
        const owl::vec2i prevPixelId = owl::vec2i(int(round(p.x)), int(round(p.y)));
        const int prevPboOffset = prevPixelId.x + self.pboSize.x * prevPixelId.y;
        if (0 <= prevPboOffset && prevPboOffset < self.pboSize.x * self.pboSize.y) {
            prevReservoir = (optixLaunchParams.curReservoir == 0) ? optixLaunchParams.reservoir1[prevPboOffset]
                                                                  : optixLaunchParams.reservoir0[prevPboOffset];
            const GBufferInfo &gBufPrev = optixLaunchParams.curReservoir == 0
                                              ? optixLaunchParams.gBuffer1[prevPboOffset]
                                              : optixLaunchParams.gBuffer0[prevPboOffset];
            const HitInfo &hitPrev = gBufPrev.hitInfo;
            // GBuffer checks
            if (fabs(hit.t - hitPrev.t) > 0.1f || hitPrev.mat != hit.mat || dot(hitPrev.sn, hit.sn) < 0.9999f ||
                length(hit.p - hitPrev.p) > 0.01f) {
                prevReservoir.valid = false;
            }

            // Invalidate if not visible
            if (prevReservoir.valid) {
                if (!isVisible(prevReservoir.sample, hit, nextHit)) {
                    prevReservoir.valid = false;
                }
            }
        }

        // Init current reservoir
        LightSampleInfo lightSample;
        sampleLight(hit.p, {nextHit.random(), nextHit.random(), nextHit.random()}, lightSample);
        owl::vec3f targetPdf = evalTargetPdf(lightSample, hit, mat, onb, nextHit);
        curReservoir.sample = lightSample;
        curReservoir.sumWeights = luminance(targetPdf) / lightSample.areaPdf;
        curReservoir.count = 1;
        curReservoir.valid = true;

        // MERGEEEE
        if (prevReservoir.valid) {
            float wPrev = prevReservoir.sumWeights;
            float wCur = curReservoir.sumWeights;
            float sum = wPrev + wCur;

            if (nextHit.random() < (wPrev / sum)) {
                curReservoir.sample = prevReservoir.sample;
            }

            curReservoir.sumWeights = sum;
            curReservoir.count += prevReservoir.count;
        }

        owl::vec3f pChosen = evalTargetPdf(curReservoir.sample, hit, mat, onb, nextHit);
        float W = (curReservoir.sumWeights / (curReservoir.count * luminance(pChosen)));
        Li = W * pChosen;
        curReservoir_ = curReservoir;
        if (luminance(mat.emission) > 0.001f) {
            Li = mat.emission;
        }
        // Li *= (prevReservoir.valid) ? 1.f : owl::vec3f(1.f, 0.f, 1.f);
    } else {
        Li = hit.emitted;
    }

    self.pboPtr[pboOffset] = {Li, 1.f};
}

OPTIX_MISS_PROGRAM(Miss2)() {
    // const MissProgData &self = owl::getProgramData<MissProgData>();
    RayInfo &prd = owl::getPRD<RayInfo>();

    // For now, grey background
    prd.hitInfo.emitted = 1.f;
    prd.hitInfo.intersectEvent = RayMissed;
}
