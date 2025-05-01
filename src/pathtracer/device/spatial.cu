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
    const Reservoir &tempReservoir = optixLaunchParams.curReservoir == 0 ? optixLaunchParams.reservoir0[pboOffset]
                                                                         : optixLaunchParams.reservoir1[pboOffset];

    // Collect
    const GBufferInfo &gBuf = optixLaunchParams.curReservoir == 0 ? optixLaunchParams.gBuffer0[pboOffset]
                                                                  : optixLaunchParams.gBuffer1[pboOffset];
    Reservoir &spatialReservoir_ = optixLaunchParams.spatialReservoir[pboOffset];

    const HitInfo &hit = gBuf.hitInfo;

    // Setup shading frame
    ONB onb(hit.sn);

    // Setup material result
    const MaterialResult &mat = gBuf.mat;

    // Next hit info
    RayInfo nextHit;
    nextHit.random.init(pboOffset, optixLaunchParams.frame.id);

    owl::vec3f Li(0.f);
    spatialReservoir_.valid = false;
    if (hit.intersectEvent == RayScattered) {
        if (luminance(mat.emission) > 0.f) {
            spatialReservoir_.valid = false;
            Li = mat.emission;
        } else {
            Reservoir curReservoir = tempReservoir;
            const int nSamples = 3;
            float Z = curReservoir.count;
            int neighbors[nSamples];
            int counts[nSamples];
            int sampleId = 0;
            while (sampleId < nSamples) {
                // Collect neighbor vars
                owl::vec2f diskSample = 30.f * randomInUnitDisk({nextHit.random(), nextHit.random()});
                owl::vec2i neighbor = pixelId + owl::vec2i(diskSample);
                const int neighborOffset = neighbor.x + self.pboSize.x * neighbor.y;
                if (neighborOffset < 0 || neighborOffset >= self.pboSize.x * self.pboSize.y) {
                    continue;
                }
                const Reservoir &neighborReservoir = optixLaunchParams.curReservoir == 0
                                                         ? optixLaunchParams.reservoir0[neighborOffset]
                                                         : optixLaunchParams.reservoir1[neighborOffset];
                const GBufferInfo &neighborGBuf = optixLaunchParams.curReservoir == 0
                                                      ? optixLaunchParams.gBuffer0[neighborOffset]
                                                      : optixLaunchParams.gBuffer1[neighborOffset];

                // Rejection
                if (!neighborReservoir.valid) {
                    continue;
                }
                if (hit.mat != neighborGBuf.hitInfo.mat) {
                    continue;
                }
                if (dot(hit.sn, neighborGBuf.hitInfo.sn) < 0.95f) {
                    continue;
                }
                if (fabsf(neighborGBuf.hitInfo.t / hit.t - 1.f) > 0.1) {
                    continue;
                }

                neighbors[sampleId] = neighborOffset;
                counts[sampleId] = neighborReservoir.count;

                // Update
                mergeReservoirs(curReservoir, neighborReservoir, hit, mat, onb, false, nextHit.random());

                sampleId++;
            }

            if (optixLaunchParams.camera.integrator == 7) {
                Z = curReservoir.count;
            } else {
                for (int i = 0; i < sampleId; ++i) {
                    const GBufferInfo &neighbor = optixLaunchParams.curReservoir == 0
                                                      ? optixLaunchParams.gBuffer0[neighbors[i]]
                                                      : optixLaunchParams.gBuffer1[neighbors[i]];
                    ONB neighborONB(neighbor.hitInfo.sn);
                    if (luminance(evalTargetPdf(curReservoir.sample, neighbor.hitInfo, neighbor.mat, neighborONB,
                                                true)) > 0) {
                        Z += counts[i];
                    }
                }
            }

            owl::vec3f newTargetPdf = evalTargetPdf(curReservoir.sample, hit, mat, onb, true);
            float lum = luminance(newTargetPdf);
            curReservoir.W = (lum > 0 && Z > 0) ? curReservoir.wSum / (Z * lum) : 0;

            Li = curReservoir.W * newTargetPdf;
            spatialReservoir_ = curReservoir;
        }
    } else {
        Li = hit.emitted;
    }

    if (optixLaunchParams.frame.dirty) {
        self.pboPtr[pboOffset] = owl::vec4f(Li, 1.f);
    } else {
        const owl::vec4f cur = self.pboPtr[pboOffset];
        const owl::vec4f unnormalizedCur = ((float)optixLaunchParams.frame.accum - 1) * cur;
        const owl::vec4f unnormalizedNew = unnormalizedCur + owl::vec4f(Li, 1.f);
        self.pboPtr[pboOffset] = unnormalizedNew / ((float)optixLaunchParams.frame.accum);
    }
}

OPTIX_MISS_PROGRAM(Miss2)() {
    const MissProgData &self = owl::getProgramData<MissProgData>();
    RayInfo &prd = owl::getPRD<RayInfo>();

    // For now, grey background
    prd.hitInfo.emitted = 0.f;
    prd.hitInfo.intersectEvent = RayMissed;
}
