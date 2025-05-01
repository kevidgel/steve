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

__inline__ __device__ float evalG(const LightSampleInfo &to, const HitInfo &from) {
    const owl::vec3f d = to.p - from.p;
    const float cosThetaL = fmaxf(dot(to.gn, -normalize(d)), 0.f);
    return cosThetaL / dot(d, d);
}

__inline__ __device__ owl::vec3f evalTargetPdf(const LightSampleInfo &lightSample, const HitInfo &hit,
                                               const MaterialResult &mat, const ONB &onb, bool useVisibility) {
    // Eval f
    const owl::vec3f d = lightSample.p - hit.p;
    const owl::vec3f dirOut = normalize(d);
    const owl::vec3f sn_dirIn = onb.toLocal(-hit.dirIn);
    const owl::vec3f sn_dirOut = onb.toLocal(dirOut);
    const owl::vec3f sn_half = normalize(sn_dirIn + sn_dirOut);
    owl::vec3f f = evalMat(mat, sn_dirIn, sn_dirOut, sn_half);

    // Eval V
    float V = 1.f;
    if (useVisibility) {
        V = isVisible(lightSample, hit) ? 1.f : 0.f;
    }

    // Eval G
    float G = evalG(lightSample, hit);

    // Eval Le
    const owl::vec3f Le = lightSample.emission;

    return f * G * V * Le;
}

__inline__ __device__ owl::vec3f evalTargetPdfBRDF(const owl::vec3f &sn_dirIn, const owl::vec3f &sn_dirOut,
                                                   const HitInfo &hit, const MaterialResult &mat, const ONB &onb, LightSampleInfo& lightSample, float& G) {
    owl::Ray ray(hit.p, onb.toWorld(sn_dirOut), RAY_EPS, RAY_MAX);
    RayInfo hitRecord;
    traceRay(optixLaunchParams.world, ray, hitRecord);
    if (hitRecord.hitInfo.intersectEvent == RayMissed) {
        // don't handle env map just yet
        return 0.f;
    }

    MaterialResult hitMat;
    getMatResult(optixLaunchParams.mats[hitRecord.hitInfo.mat], hitRecord.hitInfo, hitMat);
    if (luminance(hitMat.emission) <= 0.f) {
        return 0.f;
    }

    const owl::vec3f d = hitRecord.hitInfo.p - hit.p;
    G = fmaxf(dot(normalize(onb.toWorld(sn_dirOut)), hitRecord.hitInfo.gn), 0.f) / dot(d, d);

    // we get this for free;
    lightSample.p = hitRecord.hitInfo.p;
    lightSample.gn = hitRecord.hitInfo.gn;
    lightSample.primI = hitRecord.hitInfo.id;
    lightSample.areaPdf = 1 / (hitRecord.hitInfo.area * optixLaunchParams.lights.size);
    lightSample.pdf = 1 / (hitRecord.hitInfo.area * optixLaunchParams.lights.size * G);

    // Eval F
    const owl::vec3f sn_half = normalize(sn_dirIn + sn_dirOut);
    owl::vec3f f = evalMat(mat, sn_dirIn, sn_dirOut, sn_half);

    return f * G * hitMat.emission;
}

__inline__ __device__ void mergeReservoirs(Reservoir &to, const Reservoir &from, const HitInfo &hit,
                                           const MaterialResult &mat, const ONB &onb, const bool useVisibility,
                                           const float sample) {
    if (from.valid) {
        const LightSampleInfo &sampleToMerge = from.sample;
        float targetPdf = luminance(evalTargetPdf(sampleToMerge, hit, mat, onb, useVisibility));
        const float weight = targetPdf * from.W * from.count;
        updateReservoir(to, sampleToMerge, weight, from.count, sample);
    }
}

__inline__ __device__ float multiPowerHeuristic(float n1, float pdf1, float n2, float pdf2, float power) {
    const float pow1 = powf(pdf1, power);
    const float pow2 = powf(pdf2, power);
    return (pow1) / (n1 * pow1 + n2 * pow2);
}
