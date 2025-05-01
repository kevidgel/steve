/**
 * @file lights.cuh
 * @brief Definitions and helper functions for lights sampling
 */

#pragma once

#include "integrator.cuh"
#include "integrator_defs.cuh"
#include "owl/common/math/vec.h"
#include "ray.cuh"

__inline__ __device__ void sampleLight(const owl::vec3f &p, const owl::vec3f &sample, LightSampleInfo &scatter) {
    const LightsMesh &lights = optixLaunchParams.lights;
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

    scatter.gn = norm;
    scatter.p = lightP;
    scatter.primI = lights.primsI[lightI];
    scatter.emission = lights.emission[lightI];
    scatter.pdf = (dist2) / (lights.size * area * cos);
    scatter.areaPdf = 1.f / (lights.size * area);
}

__inline__ __device__ float pdfLight(const owl::Ray &ray, const RayInfo &prd) {
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

__inline__ __device__ owl::vec3f sampleEnvMap(const owl::vec2f &sample) {
    Alias *table = optixLaunchParams.alias;
    int W = optixLaunchParams.aliasSize.x;
    int H = optixLaunchParams.aliasSize.y;
    int N = W * H;
    int idx = min(int(sample.x * N), N - 1);
    idx = (sample.y < table[idx].prob) ? idx : table[idx].alias;

    int j = idx / W, i = idx % W;
    float u = (i + 0.5f) / float(W);
    float v = (j + 0.5f) / float(H);

    float phi = u * 2.0f * M_PI - M_PI;
    float theta = v * M_PI;
    float sinTheta = sinf(theta), cosTheta = cosf(theta);
    return (cosf(phi) * sinTheta, cosTheta, sinf(phi) * sinTheta);
}

__inline__ __device__ float pdfEnvMap(const owl::vec3f &dir) {
    Alias *table = optixLaunchParams.alias;
    int W = optixLaunchParams.aliasSize.x;
    int H = optixLaunchParams.aliasSize.y;

    const float theta = acosf(dir.y);
    const float phi = atan2f(dir.z, dir.x);

    const float u = (phi + M_PI) * 0.5f * (1.0f / M_PI);
    const float v = theta * (1.f / M_PI);

    int i = min(int(u * W), W - 1), j = min(int(v * H), H - 1), idx = j * W + i;

    float p_k = table[idx].pdf;
    float dPhi = 2*M_PI/float(W);
    float dTheta = M_PI/float(H);
    float solid = dPhi * dTheta * sin(theta);
    return (solid > 0.f) ? (p_k / solid) : 0.f;
}

/// Extremely inefficient
__inline__ __device__ float pdfLightExpensive(const owl::Ray &ray, RayInfo &prd) {
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
__inline__ __device__ bool visiblityExpensive(const owl::Ray &ray, uint id, owl::vec3f &n, RayInfo &prd) {
    traceRay(optixLaunchParams.world, ray, prd);
    return (id == prd.hitInfo.id);
}
