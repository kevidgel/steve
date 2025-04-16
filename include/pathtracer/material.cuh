/**
 * @file material.cuh
 * @brief Material definitions and functions. Header-only. Implements Disney's BRDF
 */

#pragma once

#include "owl/common/math/vec.h"
#include "ray.cuh"
#include "sampling.cuh"

#include <cuda_runtime.h>

#define INV_PI 0.31830988618379067154f

struct ONB {
    __device__ explicit ONB(const owl::vec3f &n) {
        w = normalize(n);
        float sign = copysignf(1.f, w.z);
        float a = -1.f / (sign + w.z);
        float b = w.x * w.y * a;
        v = {1.f + sign * w.x * w.x * a, sign * b, -sign * w.x};
        u = {b, sign + w.y * w.y * a, -w.y};
    }

    __device__ owl::vec3f toLocal(const owl::vec3f &world) const {
        return owl::vec3f(dot(world, u), dot(world, v), dot(world, w));
    }

    __device__ owl::vec3f toWorld(const owl::vec3f &local) const {
        return local.x * u + local.y * v + local.z * w;
    }

    owl::vec3f u;
    owl::vec3f v;
    owl::vec3f w;
};

struct Material {
    owl::vec3f baseColor;
    owl::vec3f emission;
    float roughness;
    float anisotropic;
    float subsurface;

    // Textures if available
    bool hasBaseColorTex;
    cudaTextureObject_t baseColorTex;
};

/// Evaluated material
struct MaterialResult {
    owl::vec3f baseColor;
    owl::vec3f emission;
    float specularTransmission;
    float metallic;
    float subsurface;
    float specular;
    float roughness;
    float specularTint;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    float ior;
};

__inline__ __device__ owl::vec3f reflect(const owl::vec3f &v, const owl::vec3f &n) {
    return (2 * dot(v, n) * n) - v;
}

/// Evaluate texture sampling
__inline__ __device__ void getMatResult(const Material &mat, const Record &record, MaterialResult &matResult) {
    if (mat.hasBaseColorTex) {
        const owl::vec4f &baseColorA = tex2D<float4>(mat.baseColorTex, record.hitInfo.uv.u, record.hitInfo.uv.v);
        matResult.baseColor = {baseColorA.x, baseColorA.y, baseColorA.z};
    } else {
        matResult.baseColor = mat.baseColor;
    }

    matResult.roughness = mat.roughness;
    matResult.anisotropic = mat.anisotropic;
    matResult.subsurface = mat.subsurface;
}

/// Lambertian material
__inline__ __device__ bool sampleLambertian(const MaterialResult &mat, const owl::vec3f &dirIn,
                                            const owl::vec2f &sample, owl::vec3f &dirOut) {
    dirOut = normalize(randomCosineHemisphere(sample));
    return (dirOut.z > 0.f);
};

__inline__ __device__ float pdfLambertian(const MaterialResult &mat, const owl::vec3f &dirIn,
                                          const owl::vec3f &dirOut) {
    return max(0.f, normalize(dirOut).z) * INV_PI;
}

__inline__ __device__ owl::vec3f evalLambertian(const MaterialResult &mat, const owl::vec3f &dirIn,
                                                const owl::vec3f &dirOut) {
    return mat.baseColor * pdfLambertian(mat, dirIn, dirOut);
}

/// Disney's diffuse BRDF
// Schlick approx
__inline__ __device__ float fresnelDiffuse(const owl::vec3f &dir, const float &fPerp) {
    return 1.f + (fPerp - 1.f) * powf(1.f - abs(dir.z), 5);
}

// Base diffuse factor
__inline__ __device__ float baseDiffuse(const MaterialResult &mat, const owl::vec3f &dirIn,
                                        const owl::vec3f &dirOut, const owl::vec3f &half) {
    const float cosHO = abs(dot(half, dirOut));
    const float fDPerp = 0.5f + 2.f * mat.roughness * cosHO * cosHO;
    return fresnelDiffuse(dirIn, fDPerp) * fresnelDiffuse(dirOut, fDPerp);
}

// Subsurface factor
__inline__ __device__ float subsurfDiffuse(const MaterialResult &mat, const owl::vec3f &dirIn,
                                           const owl::vec3f &dirOut, const owl::vec3f &half) {
    // NOTE: We factor out baseColor * cos / pi
    const float cosHO = abs(dot(half, dirOut));
    const float fSSPerp = mat.roughness * cosHO * cosHO;
    const float absorption = 1.f / (abs(dirIn.z) + abs(dirOut.z));
    return 1.25f * (fresnelDiffuse(dirIn, fSSPerp) * fresnelDiffuse(dirOut, fSSPerp) * (absorption - 0.5f) + 0.5f);
}

// Cos-weighted hemisphere
__inline__ __device__ bool sampleDiffuse(const MaterialResult &mat, const owl::vec3f &dirIn,
                                         const owl::vec2f &sample, owl::vec3f &dirOut) {
    dirOut = normalize(randomCosineHemisphere(sample));
    return (dirOut.z > 0.f);
};

__inline__ __device__ float pdfDiffuse(const MaterialResult &mat, const owl::vec3f &dirIn,
                                       const owl::vec3f &dirOut, const owl::vec3f &half) {
    return fmaxf(0.f, dirOut.z) * INV_PI;
}

__inline__ __device__ owl::vec3f evalDiffuse(const MaterialResult &mat, const owl::vec3f &dirIn,
                                             const owl::vec3f &dirOut, const owl::vec3f &half) {
    const float fBase = baseDiffuse(mat, dirIn, dirOut, half);
    const float fSubsurf = subsurfDiffuse(mat, dirIn, dirOut, half);
    const float fDiffuse = (1.f - mat.subsurface) * fBase + mat.subsurface * fSubsurf;
    return fDiffuse * mat.baseColor * pdfDiffuse(mat, dirIn, dirOut, half);
}

/// Disney's specular BRDF (Metal)
__inline__ __device__ owl::vec3f fresnelGGX(const MaterialResult &mat, const owl::vec3f &half,
                                            const owl::vec3f &dirOut) {
    return mat.baseColor + (1.f - mat.baseColor) * powf(1 - abs(dot(half, dirOut)), 5);
}

__inline__ __device__ float distribGGX(const MaterialResult &mat, const owl::vec3f &half) {
    const float aspect = sqrtf(1 - 0.9 * mat.anisotropic);
    const float aX = fmaxf(0.0001, mat.roughness * mat.roughness / aspect);
    const float aY = fmaxf(0.0001, mat.roughness * mat.roughness * aspect);
    const float ellipsoid = ((half.x * half.x) / (aX * aX) + (half.y * half.y) / (aY * aY) + half.z * half.z);
    const float invD = M_PI * aX * aY * ellipsoid * ellipsoid;
    return 1.f / invD;
}

__inline__ __device__ float g1GGX(const MaterialResult &mat, const owl::vec3f &dir) {
    const float aspect = sqrtf(1 - 0.9 * mat.anisotropic);
    const float aX = fmaxf(0.0001, mat.roughness * mat.roughness / aspect);
    const float aY = fmaxf(0.0001, mat.roughness * mat.roughness * aspect);
    const float ellipsoid = ((dir.x * aX) * (dir.x * aX) + (dir.y * aY) * (dir.y * aY)) / (dir.z * dir.z);
    const float Lambda = 0.5f * (sqrt(1.f + ellipsoid) - 1.f);
    return 1.f / (1.f + Lambda);
}

// VNDF normal sampling
__inline__ __device__ bool sampleGGX(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec2f &sample,
                                     owl::vec3f &dirOut) {
    const float aspect = sqrtf(1 - 0.9 * mat.anisotropic);
    const float aX = fmaxf(0.0001f, mat.roughness * mat.roughness / aspect);
    const float aY = fmaxf(0.0001f, mat.roughness * mat.roughness * aspect);

    const owl::vec3f dirInH = normalize(owl::vec3f(aX * dirIn.x, aY * dirIn.y, dirIn.z));

    // ONB again
    const float lensq = dirInH.x * dirInH.x + dirInH.y * dirInH.y;
    const owl::vec3f T1 = lensq > 0 ? normalize(owl::vec3f(-dirInH.y, dirInH.x, 0.f)) : owl::vec3f(1.f, 0.f, 0.f);
    const owl::vec3f T2 = cross(dirInH, T1);

    // Sample normal
    const float r = sqrt(sample.x);
    const float phi = 2.0 * M_PI * sample.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    const float s = 0.5f * (1.f + dirInH.z);
    t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1 * t1)) + s * t2;
    const owl::vec3f normH = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0, 1.0 - t1 * t1 - t2 * t2)) * dirInH;
    const owl::vec3f norm = normalize(owl::vec3f(aX * normH.x, aY * normH.y, fmaxf(0.f, normH.z)));

    // Reflect
    dirOut = normalize(reflect(dirIn, norm));
    return (dirOut.z > 0.f);
};

__inline__ __device__ float pdfGGX(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec3f &dirOut,
                                   const owl::vec3f &half) {
    if (dirIn.z <= 0.f || dirOut.z <= 0.f) {
        return 0.f;
    }
    const float g1 = g1GGX(mat, dirIn);
    const float distrib = distribGGX(mat, half);
    return g1 * distrib / (4.f * dirIn.z);
}

__inline__ __device__ owl::vec3f evalGGX(const MaterialResult &mat, const owl::vec3f &dirIn,
                                         const owl::vec3f &dirOut, const owl::vec3f &half) {
    const owl::vec3f fresnel = fresnelGGX(mat, half, dirOut);
    const float geom = g1GGX(mat, dirIn) * g1GGX(mat, dirOut);
    const float distrib = distribGGX(mat, half);
    return (fresnel * distrib * geom) / (4.f * dirIn.z);
}

/// Full material sampling in local space
__inline__ __device__ bool sampleMat(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec2f &sample,
                                     owl::vec3f &dirOut) {
    return sampleGGX(mat, dirIn, sample, dirOut);
};

__inline__ __device__ float pdfMat(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec3f &dirOut,
                                   const owl::vec3f &half) {
    return pdfGGX(mat, dirIn, dirOut, half);
}

__inline__ __device__ owl::vec3f evalMat(const MaterialResult &mat, const owl::vec3f &dirIn,
                                         const owl::vec3f &dirOut, const owl::vec3f &half) {
    if (dirOut.z < 0.f) return 0;
    return evalGGX(mat, dirIn, dirOut, half);
}
