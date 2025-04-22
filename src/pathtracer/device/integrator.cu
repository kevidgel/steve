#include "ray.cuh"
#include "geometry.cuh"
#include "integrator.cuh"
#include "material.cuh"

#include <optix_device.h>

#define SPP 1
#define MAX_DEPTH 16

__inline__ __device__ owl::vec3f xfmPt(const affine3f_ &xform, const owl::vec3f &p) {
    return xform.vx * p.x + xform.vy * p.y + xform.vz * p.z + xform.p;
}

__inline__ __device__ owl::vec3f xfmVec(const affine3f_ &xform, const owl::vec3f &v) {
    return xform.vx * v.x + xform.vy * v.y + xform.vz * v.z;
}

__inline__ __device__ owl::Ray generateRay(float u, float v, const owl::vec2f &lensSample) {
    const DeviceCamera &camera = optixLaunchParams.camera;
    const owl::vec2f disk = camera.apertureRadius * randomInUnitDisk(lensSample);
    const owl::vec3f origin(disk.x, disk.y, 0.f);
    const owl::vec3f direction(camera.sensorSize.x * (0.5f - u / camera.resolution.x),
                               camera.sensorSize.y * (v / camera.resolution.y - 0.5f), camera.focalDist);
    const owl::vec3f tOrigin = xfmPt(camera.xform, origin);
    const owl::vec3f tDirection = xfmVec(camera.xform, direction - origin);
    return owl::Ray(tOrigin, tDirection, 1e-3f, 1e10f);
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

__inline__ __device__ float powerHeuristic(float pdf1, float pdf2, float power) {
    const float pow1 = powf(pdf1, power);
    const float pow2 = powf(pdf2, power);
    return (pow1) / (pow1 + pow2);
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
    for (int depth = 0; depth <= MAX_DEPTH; ++depth) {
        // Intersect
        if (prd.intersectEvent == RayScattered) {

            // Obtain material at point
            MaterialResult mat;
            getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd, mat);

            // First check if material is emissive
            owl::vec3f emit = misWeight * mat.emission;

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

            // Lights strategy
            {
                // Sample lights
                const owl::vec3f lightSample = {prd.random(), prd.random(), prd.random()};
                sampleLight(prd.hitInfo.p, lightSample, scatterRecord);
                owl::Ray lightRay(prd.hitInfo.p, scatterRecord.dir, 1e-3f, 1e10f);

                const owl::vec3f lightDirOut = onb.toLocal(scatterRecord.dir);
                const owl::vec3f half = normalize(dirIn + lightDirOut);

                float pdfLight1 = scatterRecord.pdf;
                if (pdfLight1 > 0.f) {
                    // Attempt to find emitted light
                    traceRay(optixLaunchParams.world, lightRay, lightPrd);
                    Material& localMat = optixLaunchParams.mats[lightPrd.hitInfo.mat];
                    owl::vec2f& uv = lightPrd.hitInfo.uv;
                    float alpha;
                    getTexResult1f(localMat.hasAlphaTex, localMat.alphaTex, uv.u, uv.v, alpha);
                    if (alpha < 0.5f) {
                        lightRay = owl::Ray(lightPrd.hitInfo.p, lightRay.direction, 1e-3f, 1e10f);
                        traceRay(optixLaunchParams.world, lightRay, lightPrd);
                    }

                    owl::vec3f emitLight = optixLaunchParams.mats[lightPrd.hitInfo.mat].emission;

                    // Check if occluded (hopefully triangles don't occlude themselves)
                    if (lightPrd.hitInfo.id != scatterRecord.primI) emitLight = 0.f;
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
                    ray = owl::Ray(prd.hitInfo.p, onb.toWorld(brdfDirOut), 1e-3f, 1e10f);
                    // We're gonna do something tricky and combine next intersection test with occulusion test
                    traceRay(optixLaunchParams.world, ray, prd);

                    float pdfBRDF2 = 0.f;
                    if (luminance(optixLaunchParams.mats[prd.hitInfo.mat].emission) > 0.01) {
                        // Compute pdf of light
                        owl::vec3f d = prd.hitInfo.p - ray.origin;
                        float dist2 = dot(d, d);
                        float cos = abs(dot(normalize(d), prd.hitInfo.gn));

                        pdfBRDF2 = dist2 / (optixLaunchParams.lights.size * prd.hitInfo.area * cos);
                    }
                    owl::vec3f evalBRDF = evalMat(mat, dirIn, brdfDirOut, half);

                    misWeight = powerHeuristic(pdfBRDF1, pdfBRDF2, power);
                    result += throughput * emit;
                    throughput *= evalBRDF / pdfBRDF1;
                } else {
                    break;
                }
            }

            // Update
        } else if (prd.intersectEvent == RayCancelled) {
            // TODO: hit emissive
            result += throughput * prd.emitted;
            break;
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

__inline__ __device__ owl::vec3f LiDebug(owl::Ray& ray, Record &prd) {
    traceRay(optixLaunchParams.world, ray, prd);
    // Obtain material at point
    if (prd.intersectEvent == RayScattered) {
        MaterialResult mat;
        getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd, mat);
        return mat.emission;
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
            if (prevId != -1 && prevId != prd.hitInfo.id) prd.emitted = 0.f;

            // Terminate if depth is reached
            if (depth == 1 || luminance(prd.emitted) > 0.f) {
                result += throughput * prd.emitted;
                break;
            }

            // Obtain material at point
            MaterialResult mat;
            getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd, mat);

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
        } else if (prd.intersectEvent == RayCancelled) {
            // TODO: hit emissive
            result += throughput * prd.emitted;
            break;
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
            getMatResult(optixLaunchParams.mats[prd.hitInfo.mat], prd, mat);

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

            const owl::vec3f half = normalize(dirIn + brdfDirOut);
            const float pdfBRDF = pdfMat(mat, dirIn, brdfDirOut, half);
            const owl::vec3f evalBRDF = evalMat(mat, dirIn, brdfDirOut, half);

            result += throughput * emit;
            throughput *= evalBRDF / pdfBRDF;
            ray = owl::Ray(prd.hitInfo.p, onb.toWorld(brdfDirOut), 1e-3f, 1e10f);
        } else if (prd.intersectEvent == RayCancelled) {
            // TODO: hit emissive
            result += throughput * prd.emitted;
            break;
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
    prd.emitted = 0.1f;
    prd.intersectEvent = RayMissed;
}