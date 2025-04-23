#include "geometry.cuh"
#include "integrator.cuh"
#include "material.cuh"
#include "ray.cuh"

#include <optix_device.h>

#define SPP 1
#define MAX_DEPTH 16
#define RAY_EPS (1e-4f)
#define RAY_MAX (1e10f)

__inline__ __device__ owl::Ray generateRay(float u, float v, const owl::vec2f &lensSample) {
    const DeviceCamera &camera = optixLaunchParams.camera;
    const owl::vec2f disk = camera.apertureRadius * randomInUnitDisk(lensSample);
    const owl::vec3f origin(disk.x, disk.y, 0.f);
    const owl::vec3f direction(camera.sensorSize.x * (0.5f - u / camera.resolution.x),
                               camera.sensorSize.y * (v / camera.resolution.y - 0.5f), camera.focalDist);
    const owl::vec3f tOrigin = xfmPt(camera.xform, origin);
    const owl::vec3f tDirection = xfmVec(camera.xform, direction - origin);
    return owl::Ray(tOrigin, tDirection, RAY_EPS, RAY_MAX);
}

// Samples a point on a light
__inline__ __device__ void sampleLight(const owl::vec3f &p, const owl::vec3f &sample, ScatterRecord &scatter) {
    const Lights &lights = optixLaunchParams.lights;
    uint lightI = static_cast<uint>(sample.x * lights.size);
    const owl::vec3ui vertI = lights.vertsI[lightI];
    const owl::vec3f v0 = lights.verts[vertI.x];
    const owl::vec3f v1 = lights.verts[vertI.y];
    const owl::vec3f v2 = lights.verts[vertI.z];

    float u = sample.y;
    float v = sample.z;
    if (u + v > 1.f) {
        u = 1.f - u;
        v = 1.f - v;
    }
    float w = 1.f - u - v;

    const owl::vec3f lightP = (u * v0) + (v * v1) + (w * v2);
    const owl::vec3f perp = cross(v1 - v0, v2 - v0);
    // const float lenPerp = length(perp);
    const owl::vec3f d = lightP - p;

    const float area = length(perp) * 0.5f;
    const owl::vec3f norm = normalize(perp);
    const float dist2 = dot(d, d);
    const owl::vec3f wo = normalize(d);
    const float cos = fabsf(dot(wo, norm));

    scatter.dir = wo;
    scatter.p = lightP;
    scatter.primI = lights.primsI[lightI];
    scatter.pdf = (dist2) / (lights.size * area * cos);
}

__inline__ __device__ float pdfLight(const owl::Ray &ray, const Record &prd) {
    float pdf = 0.f;
    if (luminance(optixLaunchParams.mats[prd.hitInfo.mat].emission) > 0.01) {
        // Compute pdf of light
        owl::vec3f d = prd.hitInfo.p - ray.origin;
        float dist2 = dot(d, d);
        float cos = abs(dot(normalize(d), prd.hitInfo.gn));

        pdf = dist2 / (optixLaunchParams.lights.size * prd.hitInfo.area * cos);
    }
    return pdf;
}

/// Extremely inefficient
__inline__ __device__ float pdfLightExpensive(const owl::Ray &ray, Record &prd) {
    traceRay(optixLaunchParams.world, ray, prd);
    float pdf = 0.f;
    if (luminance(optixLaunchParams.mats[prd.hitInfo.mat].emission) > 0.01) {
        // Compute pdf of light
        owl::vec3f d = prd.hitInfo.p - ray.origin;
        float dist2 = dot(d, d);
        float cos = abs(dot(normalize(d), prd.hitInfo.gn));

        pdf = dist2 / (optixLaunchParams.lights.size * prd.hitInfo.area * cos);
    }
    return pdf;
}

/// Extremely inefficient
__inline__ __device__ bool visiblityExpensive(const owl::Ray &ray, uint id, owl::vec3f &n, Record &prd) {
    traceRay(optixLaunchParams.world, ray, prd);
    return (id == prd.hitInfo.id);
}

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
__inline__ __device__ owl::vec3f LiMIS(owl::Ray &ray, Record &prd, float power) {
    owl::vec3f result(0.f);
    owl::vec3f throughput(1.f);
    float misWeight = 1.f;
    ONB onb({0.f, 0.f, 1.f});
    ScatterRecord scatterRecord;
    Record lightPrd;

    // First shoot primary ray
    prd.emitted = 0.f;
    traceRay(optixLaunchParams.world, ray, prd);
    MaterialResult mat;
    for (int depth = 0; depth <= MAX_DEPTH; ++depth) {
        // Intersect
        if (prd.intersectEvent == RayScattered) {

            // Obtain material at point
            getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd, ray, mat);

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
                sampleLight(prd.hitInfo.p, lightSample, scatterRecord);
                owl::Ray lightRay(prd.hitInfo.p, scatterRecord.dir, RAY_EPS, RAY_MAX);

                const owl::vec3f lightDirOut = onb.toLocal(scatterRecord.dir);
                const owl::vec3f half = normalize(dirIn + lightDirOut);

                float pdfLight1 = scatterRecord.pdf;
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
                    if (lightPrd.hitInfo.id != scatterRecord.primI) {
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
        } else if (prd.intersectEvent == RayMissed) {
            // hit background
            result += throughput * misWeight * prd.emitted;
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

__inline__ __device__ owl::vec3f LiDebug(owl::Ray &ray, Record &prd) {
    traceRay(optixLaunchParams.world, ray, prd);

    if (prd.intersectEvent == RayScattered) {
        if (dot(ray.direction, prd.hitInfo.gn) < 0.f) {
            return (prd.hitInfo.gn + 1.f) / 2.f;
        }
    }

    return {0.f, 0.f, 0.f};
}

/// Lights integrator
__inline__ __device__ owl::vec3f LiDirect(owl::Ray &ray, Record &prd) {
    owl::vec3f result(0.f);
    owl::vec3f throughput(1.f);
    ONB onb({0.f, 0.f, 1.f});
    ScatterRecord scatterRecord;
    Record lightPrd;
    int prevId = -1;

    prd.emitted = 0.f;
    for (int depth = 0; depth <= 1; ++depth) {
        traceRay(optixLaunchParams.world, ray, prd);
        // Intersect
        if (prd.intersectEvent == RayScattered) {

            // First check if material is emissive
            prd.emitted = optixLaunchParams.mats[prd.hitInfo.mat].emission;
            if (prevId != -1 && prevId != prd.hitInfo.id) {
                prd.emitted = 0.f;
            }

            // Terminate if depth is reached
            if (depth == 1 || luminance(prd.emitted) > 0.f) {
                result += throughput * prd.emitted;
                break;
            }

            // Obtain material at point
            MaterialResult mat;
            getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd, ray, mat);

            // Cutout material
            if (prd.random() < 1.f - mat.alpha) {
                ray = owl::Ray(prd.hitInfo.p, ray.direction, 1e-3f, 1e10f);
                traceRay(optixLaunchParams.world, ray, prd);
                continue;
            }

            // Create local shading frame
            onb = ONB(prd.hitInfo.sn);
            const owl::vec3f dirIn = -normalize(onb.toLocal(ray.direction));

            // Sample lights
            const owl::vec3f lightSample = {prd.random(), prd.random(), prd.random()};
            sampleLight(prd.hitInfo.p, lightSample, scatterRecord);

            const float pdf = scatterRecord.pdf;
            prevId = scatterRecord.primI;
            if (pdf > 0.f) {
                const owl::vec3f lightDirOut = onb.toLocal(scatterRecord.dir);
                const owl::vec3f half = normalize(dirIn + lightDirOut);
                const owl::vec3f eval = evalMat(mat, dirIn, lightDirOut, half);

                result += throughput * prd.emitted;
                throughput *= eval / pdf;
                ray = owl::Ray(prd.hitInfo.p, scatterRecord.dir, 1e-3f, 1e10f);
            }
        } else if (prd.intersectEvent == RayMissed) {
            // hit background
            result += throughput * prd.emitted;
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

/// Normal integrator
__inline__ __device__ owl::vec3f Li(owl::Ray &ray, Record &prd) {
    owl::vec3f result(0.f);
    owl::vec3f throughput(1.f);
    ONB onb({0.f, 0.f, 1.f});
    ScatterRecord scatterRecord;
    Record lightPrd;

    prd.emitted = 0.f;
    for (int depth = 0; depth <= MAX_DEPTH; ++depth) {
        traceRay(optixLaunchParams.world, ray, prd);
        // Intersect
        if (prd.intersectEvent == RayScattered) {

            // Obtain material at point
            MaterialResult mat;
            getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd, ray, mat);

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
        } else if (prd.intersectEvent == RayMissed) {
            // hit background
            result += throughput * prd.emitted;
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

__inline__ __device__ owl::vec3f LiRISDirect(owl::Ray &ray, Record &prd, float power) {
    const int nBRDF = 2, nNEE = 2;
    prd.emitted = 0.f;
    owl::vec3f result(0.f);

    // Shoot primary ray
    traceRay(optixLaunchParams.world, ray, prd);

    if (prd.intersectEvent == RayScattered) {
        // Get material
        MaterialResult mat;
        getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd, ray, mat);

        // Add emission
        result += mat.emission;

        // Create local shading frame
        ONB onb(prd.hitInfo.sn);
        const owl::vec3f dirIn = -normalize(onb.toLocal(ray.direction));

        // Generate 2 BRDF samples
        owl::vec3f brdfDirOut1, brdfDirOut2;
        float brdfPdf1, brdfPdf2, brdfPdf1NEE, brdfPdf2NEE;
        Record brdfRecord1, brdfRecord2;
        sampleMat(mat, dirIn, {prd.random(), prd.random(), prd.random()}, brdfDirOut1);
        sampleMat(mat, dirIn, {prd.random(), prd.random(), prd.random()}, brdfDirOut2);

        owl::Ray brdfRay1(prd.hitInfo.p, onb.toWorld(brdfDirOut1), RAY_EPS, RAY_MAX);
        owl::Ray brdfRay2(prd.hitInfo.p, onb.toWorld(brdfDirOut2), RAY_EPS, RAY_MAX);

        brdfPdf1 = pdfMat(mat, dirIn, brdfDirOut1, normalize(dirIn + brdfDirOut1));
        brdfPdf2 = pdfMat(mat, dirIn, brdfDirOut2, normalize(dirIn + brdfDirOut2));
        brdfPdf1NEE = pdfLightExpensive(brdfRay1, brdfRecord1);
        brdfPdf2NEE = pdfLightExpensive(brdfRay2, brdfRecord2);

        brdfRecord1.emitted = (brdfRecord1.intersectEvent == RayScattered)
                                  ? optixLaunchParams.mats[brdfRecord1.hitInfo.mat].emission
                                  : 0.f;
        brdfRecord2.emitted = (brdfRecord2.intersectEvent == RayScattered)
                                  ? optixLaunchParams.mats[brdfRecord2.hitInfo.mat].emission
                                  : 0.f;

        // Generate 2 NEE samples
        ScatterRecord lightSample1, lightSample2;
        float lightPdf1 = 0.f, lightPdf2 = 0.f, lightPdf1BRDF, lightPdf2BRDF;
        Record lightRecord1, lightRecord2;
        sampleLight(prd.hitInfo.p, {prd.random(), prd.random(), prd.random()}, lightSample1);
        sampleLight(prd.hitInfo.p, {prd.random(), prd.random(), prd.random()}, lightSample2);

        owl::Ray lightRay1(prd.hitInfo.p, lightSample1.dir, RAY_EPS, RAY_MAX);
        owl::Ray lightRay2(prd.hitInfo.p, lightSample2.dir, RAY_EPS, RAY_MAX);

        bool visible1 = visiblityExpensive(lightRay1, lightSample1.primI, prd.hitInfo.sn, lightRecord1);
        bool visible2 = visiblityExpensive(lightRay2, lightSample2.primI, prd.hitInfo.sn, lightRecord2);
        lightPdf1 = lightSample1.pdf;
        lightPdf2 = lightSample2.pdf;

        lightPdf1BRDF =
            pdfMat(mat, dirIn, onb.toLocal(lightSample1.dir), normalize(dirIn + onb.toLocal(lightSample1.dir)));
        lightPdf2BRDF =
            pdfMat(mat, dirIn, onb.toLocal(lightSample2.dir), normalize(dirIn + onb.toLocal(lightSample2.dir)));

        lightRecord1.emitted = optixLaunchParams.mats[lightRecord1.hitInfo.mat].emission;
        lightRecord2.emitted = optixLaunchParams.mats[lightRecord2.hitInfo.mat].emission;

        // Compute MIS weights
        float m0 = multiPowerHeuristic(nBRDF, brdfPdf1, nNEE, brdfPdf1NEE, power);
        float m1 = multiPowerHeuristic(nBRDF, brdfPdf2, nNEE, brdfPdf2NEE, power);
        float m2 = multiPowerHeuristic(nNEE, lightPdf1, nBRDF, lightPdf1BRDF, power);
        float m3 = multiPowerHeuristic(nNEE, lightPdf2, nBRDF, lightPdf2BRDF, power);

        // Compute target function
        float p0 =
            luminance(evalMat(mat, dirIn, brdfDirOut1, normalize(dirIn + brdfDirOut1)) * brdfRecord1.emitted);
        float p1 =
            luminance(evalMat(mat, dirIn, brdfDirOut2, normalize(dirIn + brdfDirOut2)) * brdfRecord2.emitted);
        float p2 = (visible1) ? luminance(evalMat(mat, dirIn, onb.toLocal(lightSample1.dir), normalize(dirIn + onb.toLocal(lightSample1.dir))) *
            lightRecord1.emitted) : 0.f;
        float p3 = (visible2) ? luminance(
            evalMat(mat, dirIn, onb.toLocal(lightSample2.dir), normalize(dirIn + onb.toLocal(lightSample2.dir))) *
            lightRecord2.emitted) : 0.f;

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
            result += W *
                      evalMat(mat, dirIn, onb.toLocal(lightSample1.dir),
                              normalize(dirIn + onb.toLocal(lightSample1.dir))) *
                      lightRecord1.emitted;
        } else if (sampleIndex == 3) {
            W = sumWeights / p3;
            result += W *
                      evalMat(mat, dirIn, onb.toLocal(lightSample2.dir),
                              normalize(dirIn + onb.toLocal(lightSample2.dir))) *
                      lightRecord2.emitted;
        }
    } else if (prd.intersectEvent == RayMissed) {
        result += prd.emitted;
    }
    return result;
}

/// Ray generation shader
OPTIX_RAYGEN_PROGRAM(RayGen)() {
    const RayGenData &self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelId = owl::getLaunchIndex();
    const int pboOffset = pixelId.x + self.pboSize.x * pixelId.y;

    // Initialize record
    Record prd;
    prd.random.init(pboOffset, optixLaunchParams.frame.id);

    // Color @ pixel
    owl::vec3f color(0.f);
    for (int sampleId = 0; sampleId < SPP; ++sampleId) {
        const owl::vec2f pixelOffset(prd.random(), prd.random());
        const owl::vec2f uv = (owl::vec2f(pixelId) + pixelOffset);
        const owl::vec2f lensSample = {prd.random(), prd.random()};
        owl::Ray ray = generateRay(uv.x, uv.y, lensSample);
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
        self.pboPtr[pboOffset] = owl::vec4f(color * (1.f / (SPP)), 1.f);
    } else {
        const owl::vec4f cur = self.pboPtr[pboOffset];
        const owl::vec4f unnormalizedCur = ((float)optixLaunchParams.frame.accum - 1) * cur;
        const owl::vec4f unnormalizedNew = unnormalizedCur + owl::vec4f(color * (1.f / SPP), 1.f);
        self.pboPtr[pboOffset] = unnormalizedNew / ((float)optixLaunchParams.frame.accum);
    }
}

OPTIX_MISS_PROGRAM(Miss)() {
    // const MissProgData &self = owl::getProgramData<MissProgData>();
    Record &prd = owl::getPRD<Record>();

    // owl::vec3f dir = optixGetWorldRayDirection();

    // For now, grey background
    prd.emitted = 0.f;
    prd.intersectEvent = RayMissed;
}