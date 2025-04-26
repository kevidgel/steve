/**
 * @file restir.cuh
 * @brief restir specific functions
 */

#pragma once

#include "integrator_defs.cuh"
#include "owl/common/math/vec.h"
#include "ray_defs.cuh"

#include <cuda_runtime.h>

__inline__ __device__ bool isVisible(const LightSampleInfo &to, const HitInfo &from) {
    owl::vec3f dir = normalize(to.p - from.p);
    owl::Ray ray(from.p, dir, RAY_EPS, RAY_MAX);
    RayInfo record;
    traceRay(optixLaunchParams.world, ray, record);
    return (length(record.hitInfo.p - to.p) < 0.001f && record.hitInfo.id == to.primI &&
            dot(dir, from.sn) * dot(-from.dirIn, from.sn) > 0.f);
}

__inline__ __device__ owl::vec3f evalTargetPdf(const LightSampleInfo &lightSample, const HitInfo &hit,
                                               const MaterialResult &mat, const ONB &onb, bool useVisibility) {
    const owl::vec3f d = lightSample.p - hit.p;
    const owl::vec3f dirOut = normalize(d);
    const owl::vec3f sn_dirIn = onb.toLocal(-hit.dirIn);
    const owl::vec3f sn_dirOut = onb.toLocal(dirOut);
    const owl::vec3f sn_half = normalize(sn_dirIn + sn_dirOut);
    const float cosThetaL = fmaxf(dot(lightSample.gn, -dirOut), 0.f);

    owl::vec3f f = evalMat(mat, sn_dirIn, sn_dirOut, sn_half);

    float V = 1.f;
    if (useVisibility) {
        V = isVisible(lightSample, hit) ? 1.f : 0.f;
    }
    const float dist2 = dot(d, d);
    const float G = cosThetaL / dist2;
    const owl::vec3f Le = lightSample.emission;

    return f * G * V * Le;
}

__inline__ __device__ owl::vec3f evalTargetPdfBRDF(const LightSampleInfo &lightSample, const HitInfo &hit,
                                                   const MaterialResult &mat, const ONB &onb, bool useVisibility) {
    const owl::vec3f d = lightSample.p - hit.p;
    const owl::vec3f dirOut = normalize(d);
    const owl::vec3f sn_dirIn = onb.toLocal(-hit.dirIn);
    const owl::vec3f sn_dirOut = onb.toLocal(dirOut);
    const owl::vec3f sn_half = normalize(sn_dirIn + sn_dirOut);
    const float cosThetaL = fmaxf(dot(lightSample.gn, -dirOut), 0.f);

    owl::vec3f f = evalMat(mat, sn_dirIn, sn_dirOut, sn_half);

    float V = 1.f;
    if (useVisibility) {
        V = isVisible(lightSample, hit) ? 1.f : 0.f;
    }
    const float dist2 = dot(d, d);
    const float G = cosThetaL / dist2;
    const owl::vec3f Le = lightSample.emission;

    return f * G * V * Le;
}
