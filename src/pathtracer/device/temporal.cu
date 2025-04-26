#include "lights.cuh"
#include "material.cuh"
#include "reservoir.cuh"
#include "restir.cuh"
#include "geometry.cuh"
#include "integrator.cuh"

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
        const MaterialResult &mat = gBuf.mat;

        // Next hit info
        RayInfo nextHit;
        nextHit.random.init(pboOffset, optixLaunchParams.frame.id);

        // Temporal reprojection
        // ts pmo icl ngl
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
            if (fabs(hit.t - hitPrev.t) > 0.1f || hitPrev.mat != hit.mat || dot(hitPrev.sn, hit.sn) < 0.90f ||
                length(hit.p - hitPrev.p) > 0.1f) {
                prevReservoir.valid = false;
            }
        }


        const int nSamples = (prevReservoir.valid) ? 2 : 24;
        // Init current reservoir
        initReservoir(curReservoir);
        for (int i = 0; i < nSamples; ++i) {
            LightSampleInfo lightSample;
            sampleLight(hit.p, {nextHit.random(), nextHit.random(), nextHit.random()}, lightSample);
            owl::vec3f targetPdf = evalTargetPdf(lightSample, hit, mat, onb, true);
            const float weight = luminance(targetPdf) / lightSample.areaPdf;
            updateReservoir(curReservoir, lightSample, weight, 1, nextHit.random());
        }

        // MERGEEEE
        if (prevReservoir.valid) {
            // Cap count
            prevReservoir.wSum /= prevReservoir.count;
            prevReservoir.wSum *= 20 * curReservoir.count;
            prevReservoir.count = 20 * curReservoir.count;

            const LightSampleInfo &prevSample = prevReservoir.sample;
            owl::vec3f prevTargetPdf = evalTargetPdf(prevSample, hit, mat, onb, false);
            const float lum = luminance(prevTargetPdf);
            const float weight = lum * prevReservoir.W * prevReservoir.count;
            updateReservoir(curReservoir, prevSample, weight, prevReservoir.count, nextHit.random());
        }

        // Re-evaluate weight
        owl::vec3f newTargetPdf = evalTargetPdf(curReservoir.sample, hit, mat, onb, false);
        float lum = luminance(newTargetPdf);
        curReservoir.W = (lum > 0.f) ? curReservoir.wSum / (curReservoir.count * lum) : 0.f;
        if (isnan(curReservoir.W) || isinf(curReservoir.W)) {
            curReservoir.W = 0.f;
        }

        curReservoir_ = curReservoir;
    }
}

OPTIX_MISS_PROGRAM(Miss2)() {
    const MissProgData &self = owl::getProgramData<MissProgData>();
    RayInfo &prd = owl::getPRD<RayInfo>();

    // For now, grey background
    prd.hitInfo.emitted = 0.f;
    prd.hitInfo.intersectEvent = RayMissed;
}
