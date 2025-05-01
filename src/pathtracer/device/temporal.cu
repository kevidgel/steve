#include "geometry.cuh"
#include "integrator.cuh"
#include "lights.cuh"
#include "material.cuh"
#include "reservoir.cuh"
#include "restir.cuh"

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
        const owl::vec3f sn_dirIn = onb.toLocal(-hit.dirIn);

        // Temporal reprojection and determine if previous reservoir is valid
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
            if (hit.mat != hitPrev.mat) {
                prevReservoir.valid = false;
            }
            if (dot(hit.sn, hitPrev.sn) < 0.9f) {
                prevReservoir.valid = false;
            }
            if (fabsf(hitPrev.t / hit.t - 1.f) > 0.1) {
                prevReservoir.valid = false;
            }
        }

        // prevReservoir.valid = false;

        // Init current reservoir
        initReservoir(curReservoir);
        const int nLightSamples = (prevReservoir.valid) ? 2 : 32;
        const int nBRDFSamples = (prevReservoir.valid) ? 0 : 0;
        for (int i = 0; i < nLightSamples; ++i) {
            LightSampleInfo lightSample;
            sampleLight(hit.p, {nextHit.random(), nextHit.random(), nextHit.random()}, lightSample);
            owl::vec3f targetPdf = evalTargetPdf(lightSample, hit, mat, onb, true);

            // const float pdfLight = lightSample.pdf;
            //
            // owl::vec3f sn_dirOut = normalize(onb.toLocal(lightSample.p - hit.p));
            // const float pdfBRDF = pdfMat(mat, sn_dirIn, sn_dirOut, normalize(sn_dirIn + sn_dirOut));
            //
            // const float m = multiPowerHeuristic(nLightSamples, pdfLight, nBRDFSamples, pdfBRDF, 1);

            const float Wx = 1 / lightSample.areaPdf;
            const float weight = luminance(targetPdf) * Wx;
            updateReservoir(curReservoir, lightSample, weight, 1, nextHit.random());
        }

        for (int i = 0; i < nBRDFSamples; ++i) {
            owl::vec3f sn_dirOut;
            LightSampleInfo lightSample;
            float G;
            sampleMat(mat, sn_dirIn, {nextHit.random(), nextHit.random(), nextHit.random()}, sn_dirOut);
            owl::vec3f targetPdf = evalTargetPdfBRDF(sn_dirIn, sn_dirOut, hit, mat, onb, lightSample, G);
            const float pdfBRDF = pdfMat(mat, sn_dirIn, sn_dirOut, normalize(sn_dirIn + sn_dirOut));
            const float m = multiPowerHeuristic(nBRDFSamples, pdfBRDF, nLightSamples, lightSample.pdf, 1);

            // convert pdf to area measure
            const float Wx = 1 / (pdfBRDF * G);
            const float weight = m * luminance(targetPdf) * Wx;

            if (luminance(targetPdf) > 0.f) {
                updateReservoir(curReservoir, lightSample, weight, 1, nextHit.random());
            } else {
                curReservoir.count += 0;
            }
        }

        // Cap count
        if (prevReservoir.valid) {
            prevReservoir.wSum /= prevReservoir.count;
            prevReservoir.wSum *= 20 * curReservoir.count;
            prevReservoir.count = 20 * curReservoir.count;
        }

        // Merge reservoirs
        mergeReservoirs(curReservoir, prevReservoir, hit, mat, onb, false, nextHit.random());

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
