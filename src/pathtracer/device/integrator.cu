#include "geometry.cuh"
#include "integrator.cuh"

#include "lights.cuh"
#include "material.cuh"
#include "ray.cuh"

#include <optix_device.h>

#define SPP 2
#define MAX_DEPTH 16

__inline__ __device__ float powerHeuristic(float pdf1, float pdf2, float power) {
    const float pow1 = powf(pdf1, power);
    const float pow2 = powf(pdf2, power);
    return (pow1) / (pow1 + pow2);
}

__inline__ __device__ float multiPowerHeuristic(float n1, float pdf1, float n2, float pdf2, float power) {
    const float pow1 = powf(pdf1, power);
    const float pow2 = powf(pdf2, power);
    return (pow1) / (n1 * pow1 + n2 * pow2);
}

__inline__ __device__ int randomIndex(float w0, float w1, float w2, float w3, float sum, float sample) {
    sample -= (w0 / sum);
    if (sample <= 0) {
        return 0;
    }
    sample -= (w1 / sum);
    if (sample <= 0) {
        return 1;
    }
    sample -= (w2 / sum);
    if (sample <= 0) {
        return 2;
    }
    sample -= (w3 / sum);
    if (sample <= 0) {
        return 3;
    }
    return -1;
}

/// Integrator
__inline__ __device__ owl::vec3f LiMIS(owl::Ray &ray, RayInfo &prd, float power) {
    owl::vec3f result(0.f);
    owl::vec3f throughput(1.f);
    float misWeight = 1.f;
    ONB onb({0.f, 0.f, 1.f});
    LightSampleInfo lightRecord;
    RayInfo lightPrd;

    // First shoot primary ray
    prd.hitInfo.emitted = 0.f;
    traceRay(optixLaunchParams.world, ray, prd);
    MaterialResult mat;
    for (int depth = 0; depth <= MAX_DEPTH; ++depth) {
        // Intersect
        if (prd.hitInfo.intersectEvent == RayScattered) {

            // Obtain material at point
            getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd.hitInfo, mat);

            // First check if material is emissive
            owl::vec3f emit = misWeight * mat.emission;

            // Terminate if depth is reached
            if (depth == MAX_DEPTH || luminance(emit) > 0.01f) {
                result += throughput * emit;
                break;
            }

            // Cutout material
            if (prd.random() < 1.f - mat.alpha) {
                ray = owl::Ray(prd.hitInfo.p, ray.direction, RAY_EPS, RAY_MAX);
                traceRay(optixLaunchParams.world, ray, prd);
                continue;
            }

            // Create local shading frame
            onb = ONB(prd.hitInfo.sn);
            const owl::vec3f dirIn = -normalize(onb.toLocal(ray.direction));

            // Lights strategy
            {
                // Sample lights
                const owl::vec3f lightSample = {prd.random(), prd.random(), prd.random()};
                sampleLight(prd.hitInfo.p, lightSample, lightRecord);
                owl::vec3f lightDir = normalize(lightRecord.p - prd.hitInfo.p);
                owl::Ray lightRay(prd.hitInfo.p, lightDir, RAY_EPS, RAY_MAX);

                const owl::vec3f lightDirOut = onb.toLocal(lightDir);
                const owl::vec3f half = normalize(dirIn + lightDirOut);

                float pdfLight1 = lightRecord.pdf;
                if (pdfLight1 > 0.f) {
                    // Attempt to find emitted light
                    traceRay(optixLaunchParams.world, lightRay, lightPrd);
                    Material &localMat = optixLaunchParams.mats[lightPrd.hitInfo.mat];
                    owl::vec2f &uv = lightPrd.hitInfo.uv;
                    float alpha;
                    getTexResult1f(localMat.hasAlphaTex, localMat.alphaTex, uv.u, uv.v, alpha);
                    if (alpha < 0.5f) {
                        lightRay = owl::Ray(lightPrd.hitInfo.p, lightRay.direction, RAY_EPS, RAY_MAX);
                        traceRay(optixLaunchParams.world, lightRay, lightPrd);
                    }

                    owl::vec3f emitLight = optixLaunchParams.mats[lightPrd.hitInfo.mat].emission;

                    // Check if occluded
                    if (lightPrd.hitInfo.id != lightRecord.primI) {
                        emitLight = 0.f;
                    }
                    if (dot(lightRay.direction, prd.hitInfo.sn) < 0.f) {
                        emitLight = 0.f;
                    }

                    float pdfLight2 = pdfMat(mat, dirIn, lightDirOut, half);
                    owl::vec3f evalLight = evalMat(mat, dirIn, lightDirOut, half);
                    float misWeightLight = powerHeuristic(pdfLight1, pdfLight2, power);
                    result += throughput * misWeightLight * evalLight * emitLight / pdfLight1;
                } else {
                    break;
                }
            }

            // BRDF strategy
            {
                // First sample BRDF
                owl::vec3f brdfDirOut;
                owl::vec3f brdfSample = {prd.random(), prd.random(), prd.random()};
                if (!sampleMat(mat, dirIn, brdfSample, brdfDirOut)) {
                    break;
                }

                const owl::vec3f half = normalize(dirIn + brdfDirOut);
                // Find pdf
                float pdfBRDF1 = pdfMat(mat, dirIn, brdfDirOut, half);
                if (pdfBRDF1 >= 0.f) {
                    ray = owl::Ray(prd.hitInfo.p, onb.toWorld(brdfDirOut), RAY_EPS, RAY_MAX);
                    // We're gonna do something tricky and combine next intersection test with occulusion test
                    traceRay(optixLaunchParams.world, ray, prd);

                    float pdfBRDF2 = pdfLight(ray, prd);
                    misWeight = powerHeuristic(pdfBRDF1, pdfBRDF2, power);

                    owl::vec3f evalBRDF = evalMat(mat, dirIn, brdfDirOut, half);
                    result += throughput * emit;
                    throughput *= evalBRDF / pdfBRDF1;
                } else {
                    break;
                }
            }

            // Update
        } else if (prd.hitInfo.intersectEvent == RayMissed) {
            // hit background
            result += throughput * misWeight * prd.hitInfo.emitted;
            break;
        } else {
            // debug color
            result = {1.f, 0.f, 1.f};
            break;
        }

        const float q = luminance(throughput);
        const float rrThreshold = 0.001f;
        if (q < rrThreshold) {
            if (prd.random() < 1.f - q) {
                break;
            }
            throughput /= q;
        }
    }

    return result;
}

__inline__ __device__ owl::vec3f LiDebugGBuffer(const GBufferInfo &gBufferInfo) {
    return (gBufferInfo.hitInfo.sn + 1.f) * 0.5f;
}

__inline__ __device__ owl::vec3f LiDebug(owl::Ray &ray, RayInfo &prd) {
    traceRay(optixLaunchParams.world, ray, prd);

    owl::vec3f res(0.f);
    if (prd.hitInfo.intersectEvent == RayScattered) {
        MaterialResult mat;
        getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd.hitInfo, mat);
        res = mat.baseColor;
    }

    return res;
}

/// Lights integrator
__inline__ __device__ owl::vec3f LiDirect(owl::Ray &ray, RayInfo &prd) {
    owl::vec3f result(0.f);
    traceRay(optixLaunchParams.world, ray, prd);
    // Intersect
    if (prd.hitInfo.intersectEvent == RayScattered) {

        // First check if material is emissive
        if (luminance(prd.hitInfo.emitted) > 0.f) {
            return prd.hitInfo.emitted;
        }

        // Obtain material at point
        MaterialResult mat;
        getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd.hitInfo, mat);

        // Cutout material
        if (prd.random() < 1.f - mat.alpha) {
            ray = owl::Ray(prd.hitInfo.p, ray.direction, RAY_EPS, RAY_MAX);
            traceRay(optixLaunchParams.world, ray, prd);
        }

        // Create local shading frame
        ONB onb(prd.hitInfo.sn);
        const owl::vec3f dirIn = -normalize(onb.toLocal(ray.direction));

        // Sample lights
        LightSampleInfo lightSample;
        sampleLight(prd.hitInfo.p, {prd.random(), prd.random(), prd.random()}, lightSample);
        if (lightSample.pdf > 0.f) {
            const owl::vec3f lightDirOutWorld = normalize(lightSample.p - prd.hitInfo.p);
            const owl::vec3f lightDirOut = onb.toLocal(lightDirOutWorld);

            // Visibility check
            owl::Ray lightRay(prd.hitInfo.p, lightDirOutWorld, RAY_EPS, RAY_MAX);
            RayInfo lightRec;
            traceRay(optixLaunchParams.world, lightRay, lightRec);
            if (length(lightRec.hitInfo.p - lightSample.p) >= 0.001f || lightRec.hitInfo.id != lightSample.primI || (dirIn.z * lightDirOut.z) < 0.f) {
                return 0.f;
            }

            const owl::vec3f half = normalize(dirIn + lightDirOut);
            const owl::vec3f eval = evalMat(mat, dirIn, lightDirOut, half);
            result = (eval / lightSample.pdf) * lightSample.emission;
        }
    } else if (prd.hitInfo.intersectEvent == RayMissed) {
        // hit background
        result = prd.hitInfo.emitted;
    }
    return result;
}

/// Normal integrator
__inline__ __device__ owl::vec3f Li(owl::Ray &ray, RayInfo &prd) {
    owl::vec3f result(0.f);
    owl::vec3f throughput(1.f);
    ONB onb({0.f, 0.f, 1.f});

    prd.hitInfo.emitted = 0.f;
    for (int depth = 0; depth <= 10; ++depth) {
        traceRay(optixLaunchParams.world, ray, prd);
        // Intersect
        if (prd.hitInfo.intersectEvent == RayScattered) {

            // Obtain material at point
            MaterialResult mat;
            getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd.hitInfo, mat);

            // First check if material is emissive
            owl::vec3f emit = mat.emission;

            // Terminate if depth is reached
            if (depth == MAX_DEPTH || luminance(emit) > 0.01f) {
                result += throughput * emit;
                break;
            }

            // Cutout material
            if (prd.random() < 1.f - mat.alpha) {
                ray = owl::Ray(prd.hitInfo.p, ray.direction, 1e-3f, 1e10f);
                traceRay(optixLaunchParams.world, ray, prd);
                continue;
            }

            // Create local shading frame
            onb = ONB(prd.hitInfo.sn);
            const owl::vec3f dirIn = -normalize(onb.toLocal(ray.direction));

            // Sample brdf
            owl::vec3f brdfDirOut;
            owl::vec3f brdfSample = {prd.random(), prd.random(), prd.random()};
            if (!sampleMat(mat, dirIn, brdfSample, brdfDirOut)) {
                break;
            }

            // TODO: change `half` for refraction event
            const owl::vec3f half = normalize(dirIn + brdfDirOut);
            const float pdfBRDF = pdfMat(mat, dirIn, brdfDirOut, half);
            const owl::vec3f evalBRDF = evalMat(mat, dirIn, brdfDirOut, half);

            result += throughput * emit;
            throughput *= evalBRDF / pdfBRDF;
            ray = owl::Ray(prd.hitInfo.p, onb.toWorld(brdfDirOut), 1e-3f, 1e10f);
        } else if (prd.hitInfo.intersectEvent == RayMissed) {
            // hit background
            result += throughput * prd.hitInfo.emitted;
            break;
        } else {
            // debug color
            result = {1.f, 0.f, 1.f};
            break;
        }

        const float q = luminance(throughput);
        const float rrThreshold = 0.001f;
        if (q < rrThreshold) {
            if (prd.random() < 1.f - q) {
                break;
            }
            throughput /= q;
        }
    }

    return result;
}

__inline__ __device__ owl::vec3f LiRISDirect(owl::Ray &ray, RayInfo &prd, float power) {
    const int nBRDF = 2, nNEE = 2;
    prd.hitInfo.emitted = 0.f;
    owl::vec3f result(0.f);

    // Shoot primary ray
    traceRay(optixLaunchParams.world, ray, prd);

    if (prd.hitInfo.intersectEvent == RayScattered) {
        // Get material
        MaterialResult mat;
        getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd.hitInfo, mat);

        // Add emission
        result += mat.emission;

        // Create local shading frame
        ONB onb(prd.hitInfo.sn);
        const owl::vec3f dirIn = -normalize(onb.toLocal(ray.direction));

        // Generate 2 BRDF samples
        owl::vec3f brdfDirOut1, brdfDirOut2;
        float brdfPdf1, brdfPdf2, brdfPdf1NEE, brdfPdf2NEE;
        RayInfo brdfRecord1, brdfRecord2;
        sampleMat(mat, dirIn, {prd.random(), prd.random(), prd.random()}, brdfDirOut1);
        sampleMat(mat, dirIn, {prd.random(), prd.random(), prd.random()}, brdfDirOut2);

        owl::Ray brdfRay1(prd.hitInfo.p, onb.toWorld(brdfDirOut1), RAY_EPS, RAY_MAX);
        owl::Ray brdfRay2(prd.hitInfo.p, onb.toWorld(brdfDirOut2), RAY_EPS, RAY_MAX);

        brdfPdf1 = pdfMat(mat, dirIn, brdfDirOut1, normalize(dirIn + brdfDirOut1));
        brdfPdf2 = pdfMat(mat, dirIn, brdfDirOut2, normalize(dirIn + brdfDirOut2));
        brdfPdf1NEE = pdfLightExpensive(brdfRay1, brdfRecord1);
        brdfPdf2NEE = pdfLightExpensive(brdfRay2, brdfRecord2);

        brdfRecord1.hitInfo.emitted = (brdfRecord1.hitInfo.intersectEvent == RayScattered)
                                          ? optixLaunchParams.mats[brdfRecord1.hitInfo.mat].emission
                                          : 0.f;
        brdfRecord2.hitInfo.emitted = (brdfRecord2.hitInfo.intersectEvent == RayScattered)
                                          ? optixLaunchParams.mats[brdfRecord2.hitInfo.mat].emission
                                          : 0.f;

        // Generate 2 NEE samples
        LightSampleInfo lightSample1, lightSample2;
        float lightPdf1 = 0.f, lightPdf2 = 0.f, lightPdf1BRDF, lightPdf2BRDF;
        RayInfo lightRecord1, lightRecord2;
        sampleLight(prd.hitInfo.p, {prd.random(), prd.random(), prd.random()}, lightSample1);
        sampleLight(prd.hitInfo.p, {prd.random(), prd.random(), prd.random()}, lightSample2);

        const owl::vec3f lightDir1 = normalize(lightSample1.p - prd.hitInfo.p);
        const owl::vec3f lightDir2 = normalize(lightSample2.p - prd.hitInfo.p);

        owl::Ray lightRay1(prd.hitInfo.p, lightDir1, RAY_EPS, RAY_MAX);
        owl::Ray lightRay2(prd.hitInfo.p, lightDir2, RAY_EPS, RAY_MAX);

        bool visible1 = visiblityExpensive(lightRay1, lightSample1.primI, prd.hitInfo.sn, lightRecord1);
        bool visible2 = visiblityExpensive(lightRay2, lightSample2.primI, prd.hitInfo.sn, lightRecord2);
        lightPdf1 = lightSample1.pdf;
        lightPdf2 = lightSample2.pdf;

        lightPdf1BRDF = pdfMat(mat, dirIn, onb.toLocal(lightDir1), normalize(dirIn + onb.toLocal(lightDir1)));
        lightPdf2BRDF = pdfMat(mat, dirIn, onb.toLocal(lightDir2), normalize(dirIn + onb.toLocal(lightDir2)));

        lightRecord1.hitInfo.emitted = optixLaunchParams.mats[lightRecord1.hitInfo.mat].emission;
        lightRecord2.hitInfo.emitted = optixLaunchParams.mats[lightRecord2.hitInfo.mat].emission;

        // Compute MIS weights
        float m0 = multiPowerHeuristic(nBRDF, brdfPdf1, nNEE, brdfPdf1NEE, power);
        float m1 = multiPowerHeuristic(nBRDF, brdfPdf2, nNEE, brdfPdf2NEE, power);
        float m2 = multiPowerHeuristic(nNEE, lightPdf1, nBRDF, lightPdf1BRDF, power);
        float m3 = multiPowerHeuristic(nNEE, lightPdf2, nBRDF, lightPdf2BRDF, power);

        // Compute target function
        float p0 = luminance(evalMat(mat, dirIn, brdfDirOut1, normalize(dirIn + brdfDirOut1)) *
                             brdfRecord1.hitInfo.emitted);
        float p1 = luminance(evalMat(mat, dirIn, brdfDirOut2, normalize(dirIn + brdfDirOut2)) *
                             brdfRecord2.hitInfo.emitted);
        float p2 = (visible1) ? luminance(evalMat(mat, dirIn, onb.toLocal(lightDir1),
                                                  normalize(dirIn + onb.toLocal(lightDir1))) *
                                          lightRecord1.hitInfo.emitted)
                              : 0.f;
        float p3 = (visible2) ? luminance(evalMat(mat, dirIn, onb.toLocal(lightDir2),
                                                  normalize(dirIn + onb.toLocal(lightDir2))) *
                                          lightRecord2.hitInfo.emitted)
                              : 0.f;

        // Compute RIS weights
        // pHat is just f for now (cos-weighted BRDF * emitted)
        float w0 = m0 * p0 / brdfPdf1;
        float w1 = m1 * p1 / brdfPdf2;
        float w2 = m2 * p2 / lightPdf1;
        float w3 = m3 * p3 / lightPdf2;
        float sumWeights = w0 + w1 + w2 + w3;

        // Choose sample
        int sampleIndex = randomIndex(w0, w1, w2, w3, sumWeights, prd.random());
        float W = 0.f;
        if (sampleIndex == 0) {
            W = sumWeights / p0;
            result += W * evalMat(mat, dirIn, brdfDirOut1, normalize(dirIn + brdfDirOut1)) * Li(brdfRay1, prd);
        } else if (sampleIndex == 1) {
            W = sumWeights / p1;
            result += W * evalMat(mat, dirIn, brdfDirOut2, normalize(dirIn + brdfDirOut2)) * Li(brdfRay2, prd);
        } else if (sampleIndex == 2) {
            W = sumWeights / p2;
            result += W * evalMat(mat, dirIn, onb.toLocal(lightDir1), normalize(dirIn + onb.toLocal(lightDir1))) *
                      lightRecord1.hitInfo.emitted;
        } else if (sampleIndex == 3) {
            W = sumWeights / p3;
            result += W * evalMat(mat, dirIn, onb.toLocal(lightDir2), normalize(dirIn + onb.toLocal(lightDir2))) *
                      lightRecord2.hitInfo.emitted;
        }
    } else if (prd.hitInfo.intersectEvent == RayMissed) {
        result += prd.hitInfo.emitted;
    }
    return result;
}

/// Ray generation shader
OPTIX_RAYGEN_PROGRAM(RayGen)() {
    const auto &self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelId = owl::getLaunchIndex();
    const int pboOffset = pixelId.x + self.pboSize.x * pixelId.y;

    // Initialize record
    RayInfo prd;
    prd.random.init(pboOffset, optixLaunchParams.frame.id);

    // Color @ pixel
    owl::vec3f color(0.f);
    uint spp = SPP;
    if (optixLaunchParams.camera.integrator == 1) {
        spp = 24;
    }

    for (int sampleId = 0; sampleId < spp; ++sampleId) {
        const owl::vec2f pixelOffset(prd.random(), prd.random());
        const owl::vec2f uv = (owl::vec2f(pixelId) + pixelOffset);
        const owl::vec2f lensSample = {prd.random(), prd.random()};
        owl::Ray ray = generateRay(optixLaunchParams.camera, uv.x, uv.y, lensSample);
        if (optixLaunchParams.camera.integrator == 0) {
            color += Li(ray, prd);
        } else if (optixLaunchParams.camera.integrator == 1) {
            color += LiDirect(ray, prd);
        } else if (optixLaunchParams.camera.integrator == 2) {
            color += LiMIS(ray, prd, 2.f);
        } else if (optixLaunchParams.camera.integrator == 3) {
            color += LiMIS(ray, prd, 1.f);
        } else if (optixLaunchParams.camera.integrator == 4) {
            color += LiRISDirect(ray, prd, 2.f);
        } else {
            color += LiDebug(ray, prd);
        }
    }

    // Reject invalid samples
    if (isnan(color.x) || isnan(color.y) || isnan(color.z)) {
        color = owl::vec3f(0.f);
    }

    if (isinf(color.x) || isinf(color.y) || isinf(color.z)) {
        color = owl::vec3f(0.f);
    }

    if (optixLaunchParams.frame.dirty) {
        self.pboPtr[pboOffset] = owl::vec4f(color * (1.f / (spp)), 1.f);
    } else {
        const owl::vec4f cur = self.pboPtr[pboOffset];
        const owl::vec4f unnormalizedCur = ((float)optixLaunchParams.frame.accum - 1) * cur;
        const owl::vec4f unnormalizedNew = unnormalizedCur + owl::vec4f(color * (1.f / spp), 1.f);
        self.pboPtr[pboOffset] = unnormalizedNew / ((float)optixLaunchParams.frame.accum);
    }
}

OPTIX_MISS_PROGRAM(Miss)() {
    const MissProgData &self = owl::getProgramData<MissProgData>();
    RayInfo &prd = owl::getPRD<RayInfo>();

    // owl::vec3f dir = optixGetWorldRayDirection();

    // For now, grey background
    prd.hitInfo.emitted = 0.0f;
    prd.hitInfo.intersectEvent = RayMissed;
}