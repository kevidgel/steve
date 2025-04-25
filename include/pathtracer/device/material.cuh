/**
 * @file material.cuh
 * @brief Material definitions and functions. Header-only. Implements Disney's BRDF
 */

#pragma once

#include "material_defs.cuh"
#include "owl/common/math/vec.h"
#include "ray.cuh"
#include "sampling.cuh"

#include <bits/algorithmfwd.h>
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
    float etaInOverOut;
    float alpha;

    float diffuseWeight;
    float sheenWeight;
    float clearcoatWeight;
    float metalWeight;
    float sumWeights;
};

__inline__ __device__ owl::vec3f reflect(const owl::vec3f &v, const owl::vec3f &n) {
    return (2 * dot(v, n) * n) - v;
}

/// Returns false if TIR. eta is ior(dirIn) / ior(dirOut)
__inline__ __device__ bool refract(const owl::vec3f &dirIn, const owl::vec3f &n, const float etaInOverOut,
                                   owl::vec3f &dirOut) {
    float cosThetaIn = dot(dirIn, n);
    float sinThetaIn = sqrtf(1.f - cosThetaIn * cosThetaIn);
    float sinThetaOut = sinThetaIn * etaInOverOut;
    if (sinThetaOut > 1.f) {
        dirOut = reflect(dirIn, n);
        return false;
    }
    float cosThetaOut = sqrtf(1.f - sinThetaOut * sinThetaOut);
    dirOut = normalize(etaInOverOut * (-dirIn + cosThetaIn * n) - cosThetaOut * n);
    return true;
}

__inline__ __device__ float R0(float ior) {
    return (ior - 1.f) * (ior - 1.f) / ((ior + 1.f) *  (ior + 1.f));
}

__inline__ __device__ void getTexResult1f(const bool hasTex, const cudaTextureObject_t &tex, const float u,
                                          const float v, float &val) {
    if (hasTex) {
        const owl::vec4f eval = tex2D<float4>(tex, u, v);
        val = eval.x;
    }
}

__inline__ __device__ void getTexResult3f(const bool hasTex, const cudaTextureObject_t &tex, const float u,
                                          const float v, owl::vec3f &vec) {
    if (hasTex) {
        const owl::vec4f eval = tex2D<float4>(tex, u, v);
        vec = {eval.x, eval.y, eval.z};
    }
}

__inline__ __device__ void getTexResult4f(const bool hasTex, const cudaTextureObject_t &tex, const float u,
                                          const float v, owl::vec4f &vec) {
    if (hasTex) {
        const owl::vec4f eval = tex2D<float4>(tex, u, v);
        vec = {eval.x, eval.y, eval.z, eval.w};
    }
}

/// Evaluate texture sampling
__inline__ __device__ void getMatResult(const Material &mat, const HitInfo &hit, MaterialResult &matResult) {
    matResult.metallic = mat.metallic;
    matResult.subsurface = mat.subsurface;
    matResult.specular = mat.specular;
    matResult.roughness = mat.roughness;
    matResult.specularTint = mat.specularTint;
    matResult.anisotropic = mat.anisotropic;
    matResult.sheen = mat.sheen;
    matResult.sheenTint = mat.sheenTint;
    matResult.clearcoat = mat.clearcoat;
    matResult.clearcoatGloss = mat.clearcoatGloss;
    matResult.baseColor = mat.baseColor;
    matResult.emission = mat.emission;
    matResult.alpha = 1.f;
    matResult.etaInOverOut = dot(hit.dirIn,  hit.gn) < 0.f ? 1.0f / mat.ior : mat.ior;

    const owl::vec2f &uv = hit.uv;
    getTexResult1f(mat.hasMetallicTex, mat.metallicTex, uv.u, uv.v, matResult.metallic);
    getTexResult1f(mat.hasSpecularTex, mat.specularTex, uv.u, uv.v, matResult.specular);
    getTexResult1f(mat.hasRoughnessTex, mat.roughnessTex, uv.u, uv.v, matResult.roughness);
    getTexResult1f(mat.hasSheenTex, mat.sheenTex, uv.u, uv.v, matResult.sheen);
    owl::vec4f baseColorWithAlpha;
    getTexResult4f(mat.hasBaseColorTex, mat.baseColorTex, uv.u, uv.v, baseColorWithAlpha);
    if (mat.hasBaseColorTex) {
        matResult.baseColor = {baseColorWithAlpha.x, baseColorWithAlpha.y, baseColorWithAlpha.z};
        matResult.alpha = baseColorWithAlpha.w;
    }
    getTexResult3f(mat.hasEmissiveTex, mat.emissiveTex, uv.u, uv.v, matResult.emission);
    getTexResult1f(mat.hasAlphaTex, mat.alphaTex, uv.u, uv.v, matResult.alpha);
    if (mat.hasEmissiveTex) {
        owl::vec3f emit;
        getTexResult3f(mat.hasEmissiveTex, mat.emissiveTex, uv.u, uv.v, emit);
        matResult.emission *= emit;
    }
    if (mat.hasMetallicRoughnessTex) {
        owl::vec3f metallicRoughness;
        getTexResult3f(mat.hasMetallicRoughnessTex, mat.metallicRoughnessTex, uv.u, uv.v, metallicRoughness);
        matResult.roughness *= metallicRoughness.y;
        matResult.metallic *= metallicRoughness.z;
    }

    // Bump mapping
    // if (mat.hasBumpTex) {
    //     float h, hU, hD, hL, hR;
    //     const float scale = 0.01f;
    //     const float delta = 0.001f;
    //     const float delta2 = 0.002f;
    //     getTexResult1f(mat.hasBumpTex, mat.bumpTex, uv.u, uv.v, h);
    //     getTexResult1f(mat.hasBumpTex, mat.bumpTex, uv.u + delta, uv.v, hU);
    //     getTexResult1f(mat.hasBumpTex, mat.bumpTex, uv.u - delta, uv.v, hD);
    //     getTexResult1f(mat.hasBumpTex, mat.bumpTex, uv.u, uv.v + delta, hL);
    //     getTexResult1f(mat.hasBumpTex, mat.bumpTex, uv.u, uv.v - delta, hR);
    //
    //     float dHdU = scale * (hU - hD) / delta2;
    //     float dHdV = scale * (hL - hR) / delta2;
    //     const owl::vec3f bn = normalize(owl::vec3f(-dHdU, -dHdV, 1.f));
    //     ONB onb(hit.sn);
    //     hit.sn = normalize(onb.toWorld(bn));
    // }

    matResult.clearcoatGloss = owl::clamp(matResult.clearcoatGloss, 0.f, 1.f);
    matResult.clearcoat = owl::clamp(matResult.clearcoat, 0.f, 1.f);
    matResult.sheen = owl::clamp(matResult.sheen, 0.f, 1.f);
    matResult.sheenTint = owl::clamp(matResult.sheenTint, 0.f, 1.f);
    matResult.specular = owl::clamp(matResult.specular, 0.f, 1.f);
    matResult.metallic = owl::clamp(matResult.metallic, 0.f, 1.f);
    matResult.roughness = owl::clamp(matResult.roughness, 0.f, 1.f);
    matResult.baseColor = clamp(matResult.baseColor, {0.f}, {1.f});

    // Sampling weights
    matResult.diffuseWeight = (1.f - mat.specular) * (1.f - mat.metallic);
    matResult.metalWeight = 1.f - mat.specular * (1.f - mat.metallic);
    matResult.clearcoatWeight = 0.25f * mat.clearcoat;
    matResult.sheenWeight = (1.f - mat.metallic)  * mat.sheen;
    matResult.sumWeights = matResult.diffuseWeight + matResult.metalWeight + matResult.clearcoatWeight + matResult.sheenWeight;
    matResult.diffuseWeight /= matResult.sumWeights;
    matResult.metalWeight /= matResult.sumWeights;
    matResult.clearcoatWeight /= matResult.sumWeights;
    matResult.sheenWeight /= matResult.sumWeights;
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
__inline__ __device__ owl::vec3f fresnelMetal(const owl::vec3f &color, const owl::vec3f &half,
                                              const owl::vec3f &dirOut) {
    return color + (1.f - color) * powf(1 - abs(dot(half, dirOut)), 5);
}

__inline__ __device__ float distribGGX(const MaterialResult &mat, const owl::vec3f &half) {
    const float aspect = sqrtf(1 - 0.9 * mat.anisotropic);
    const float aX = fmaxf(0.0001, mat.roughness * mat.roughness / aspect);
    const float aY = fmaxf(0.0001, mat.roughness * mat.roughness * aspect);
    const float ellipsoid = ((half.x * half.x) / (aX * aX) + (half.y * half.y) / (aY * aY) + half.z * half.z);
    const float invD = M_PI * aX * aY * ellipsoid * ellipsoid;
    return 1.f / invD;
}

__inline__ __device__ float g1SmithMaskingShadowing(const MaterialResult &mat, const owl::vec3f &dir) {
    const float aspect = sqrtf(1 - 0.9 * mat.anisotropic);
    const float aX = fmaxf(0.0001, mat.roughness * mat.roughness / aspect);
    const float aY = fmaxf(0.0001, mat.roughness * mat.roughness * aspect);
    const float ellipsoid = ((dir.x * aX) * (dir.x * aX) + (dir.y * aY) * (dir.y * aY)) / (dir.z * dir.z);
    const float Lambda = 0.5f * (sqrt(1.f + ellipsoid) - 1.f);
    return 1.f / (1.f + Lambda);
}

__inline__ __device__ owl::vec3f sampleVNDF(const MaterialResult &mat, const owl::vec3f &dirIn,
                                            const owl::vec2f &sample) {
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

    return norm;
}

// VNDF normal sampling
__inline__ __device__ bool sampleMetal(const MaterialResult &mat, const owl::vec3f &dirIn,
                                       const owl::vec2f &sample, owl::vec3f &dirOut) {
    owl::vec3f sampledNormal = sampleVNDF(mat, dirIn, sample);

    // Reflect
    dirOut = normalize(reflect(dirIn, sampledNormal));

    // Reject tail samples of GGX
    return (dirOut.z > 0.f);
};

__inline__ __device__ float pdfMetal(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec3f &dirOut,
                                     const owl::vec3f &half) {
    if (dirIn.z <= 0.f || dirOut.z <= 0.f) {
        return 0.f;
    }
    const float g1 = g1SmithMaskingShadowing(mat, dirIn);
    const float distrib = distribGGX(mat, half);
    return g1 * distrib / (4.f * dirIn.z);
}

__inline__ __device__ owl::vec3f evalMetal(const MaterialResult &mat, const owl::vec3f &dirIn,
                                           const owl::vec3f &dirOut, const owl::vec3f &half) {
    const float lum = luminance(mat.baseColor);
    const owl::vec3f cTint = lum > 0 ? mat.baseColor / lum : 1.f;
    const owl::vec3f kS = (1.f - mat.specularTint) + mat.specularTint * cTint;
    const owl::vec3f c0 = mat.specular * R0(mat.etaInOverOut) * (1.f - mat.metallic) * kS + mat.metallic * mat.baseColor;
    const owl::vec3f fresnel = fresnelMetal(c0, half, dirOut);
    const float geom = g1SmithMaskingShadowing(mat, dirIn) * g1SmithMaskingShadowing(mat, dirOut);
    const float distrib = distribGGX(mat, half);
    return (fresnel * distrib * geom) / (4.f * dirIn.z);
}

__inline__ __device__ owl::vec3f fresnelClearcoat(const MaterialResult &mat, const owl::vec3f &half,
                                                  const owl::vec3f &dirOut) {
    return 0.04f + 0.96f * powf(1 - abs(dot(half, dirOut)), 5);
}

__inline__ __device__ float distribClearcoat(const MaterialResult &mat, const owl::vec3f &half) {
    const float aG = (1.f - mat.clearcoatGloss) * 0.1 + mat.clearcoatGloss * 0.001;
    const float aG2 = aG * aG;
    float D = aG2 - 1.f;
    D /= M_PI * log(aG2) * (1 + (aG2 - 1.f) * (half.z * half.z));
    return D;
}

__inline__ __device__ float g1Clearcoat(const MaterialResult &mat, const owl::vec3f &dir) {
    const float aX = 0.25f;
    const float aY = 0.25f;
    const float ellipsoid = ((dir.x * aX) * (dir.x * aX) + (dir.y * aY) * (dir.y * aY)) / (dir.z * dir.z);
    const float Lambda = 0.5f * (sqrt(1.f + ellipsoid) - 1.f);
    return 1.f / (1.f + Lambda);
}

__inline__ __device__ bool sampleClearcoat(const MaterialResult &mat, const owl::vec3f &dirIn,
                                           const owl::vec2f &sample, owl::vec3f &dirOut) {
    const float aG = (1.f - mat.clearcoatGloss) * 0.1 + mat.clearcoatGloss * 0.001;
    const float aG2 = aG * aG;
    const float cosTheta2 = (1.f - powf(aG2, sample.x)) / (1.f - aG2);
    const float cosTheta = sqrtf(cosTheta2);
    const float sinTheta = sqrtf(1.f - cosTheta2);
    const float phi = 2.f * M_PI * sample.y;
    owl::vec3f norm(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    dirOut = normalize(reflect(dirIn, norm));
    return (dirOut.z > 0.f);
};

__inline__ __device__ float pdfClearcoat(const MaterialResult &mat, const owl::vec3f &dirIn,
                                         const owl::vec3f &dirOut, const owl::vec3f &half) {
    const float distrib = distribClearcoat(mat, half);
    return abs(half.z) * distrib / (4.f * abs(dot(half, dirOut)));
}

__inline__ __device__ owl::vec3f evalClearcoat(const MaterialResult &mat, const owl::vec3f &dirIn,
                                               const owl::vec3f &dirOut, const owl::vec3f &half) {
    const owl::vec3f fresnel = fresnelClearcoat(mat, half, dirOut);
    const float geom = g1Clearcoat(mat, dirIn) * g1Clearcoat(mat, dirOut);
    const float distrib = distribClearcoat(mat, half);
    return (fresnel * distrib * geom) / (4.f * dirIn.z);
}

__inline__ __device__ float fresnelGlass(const MaterialResult &mat, const owl::vec3f &dirIn,
                                         const owl::vec3f &dirOut, const owl::vec3f &half) {
    float rPara = dot(half, dirIn) - mat.etaInOverOut * dot(half, dirOut);
    rPara /= dot(half, dirIn) + mat.etaInOverOut * dot(half, dirOut);
    float rPerp = mat.etaInOverOut * dot(half, dirIn) - dot(half, dirOut);
    rPerp /= mat.etaInOverOut * dot(half, dirIn) + dot(half, dirOut);
    return 0.5f * (rPara * rPara + rPerp * rPerp);
}

/// Assume dirIn in (+) hemisphere
__inline__ __device__ bool sampleGlass(const MaterialResult &mat, const owl::vec3f &dirIn,
                                       const owl::vec3f &sample, owl::vec3f &dirOut) {
    // Sample microfacet normal (GGX in this case)
    owl::vec3f sampledNormal = sampleVNDF(mat, dirIn, {sample.x, sample.y});

    bool TIR = !refract(dirIn, sampledNormal, mat.etaInOverOut, dirOut);
    float F = TIR ? 1.0f : fresnelGlass(mat, dirIn, dirOut, sampledNormal);
    // Reflect with probability F
    if (sample.z < F) {
        dirOut = reflect(dirIn, sampledNormal);
    }
    return true;
};

/// Assume dirIn in (+) hemisphere
__inline__ __device__ float pdfGlass(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec3f &dirOut,
                                     const owl::vec3f &half, bool reflect) {
    float F = fresnelGlass(mat, dirIn, dirOut, half);
    float D = distribGGX(mat, half);
    float g1 = g1SmithMaskingShadowing(mat, dirIn);
    if (reflect) {
        return (F * D * g1) / (4.f * dirIn.z);
    } else {
        float cosHO = dot(half, dirOut);
        float cosHI = dot(half, dirIn);
        float sqrtDen = cosHI + mat.etaInOverOut * cosHO;
        float dHdO = mat.etaInOverOut * mat.etaInOverOut * cosHO / (sqrtDen * sqrtDen);
        return (1.f - F) * D * g1 * fabsf(dHdO * cosHI / dirIn.z);
    }
}

/// Assume dirIn in (+) hemisphere
__inline__ __device__ owl::vec3f evalGlass(const MaterialResult &mat, const owl::vec3f &dirIn,
                                           const owl::vec3f &dirOut, const owl::vec3f &half, bool reflect) {
    float F = fresnelGlass(mat, dirIn, dirOut, half);
    float D = distribGGX(mat, half);                                                      // Same as metal
    float G = g1SmithMaskingShadowing(mat, dirIn) * g1SmithMaskingShadowing(mat, dirOut); // Same as metal
    if (reflect) {
        // Reflection
        return (mat.baseColor * F * D * G) / (4.f * fabsf(dirIn.z));
    } else {
        // Transmission
        float cosHO = dot(half, dirOut);
        float cosHI = dot(half, dirIn);
        float sqrtDen = cosHI + mat.etaInOverOut * cosHO;

        owl::vec3f eval = sqrt(mat.baseColor) * (1.f - F) * D * G * fabsf(cosHO * cosHI);
        eval /= dirIn.z * (sqrtDen * sqrtDen);
        return eval;
    }
}

__inline__ __device__ bool sampleSheen(const MaterialResult &mat, const owl::vec3f &dirIn,
                                       const owl::vec2f &sample, owl::vec3f &dirOut) {
    dirOut = normalize(randomCosineHemisphere(sample));
    return (dirOut.z > 0.f);
};

__inline__ __device__ float pdfSheen(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec3f &dirOut,
                                     const owl::vec3f &half) {
    return fmaxf(0.f, normalize(dirOut).z) * INV_PI;
}

__inline__ __device__ owl::vec3f evalSheen(const MaterialResult &mat, const owl::vec3f &dirIn,
                                           const owl::vec3f &dirOut, const owl::vec3f &half) {
    const float lum = luminance(mat.baseColor);
    const owl::vec3f cTint = lum > 0.f ? mat.baseColor / lum : 1.f;
    const owl::vec3f cSheen = (1.f - mat.sheenTint) + mat.sheenTint * cTint;
    return cSheen * powf(1 - abs(dot(half, dirOut)), 5) * abs(dirOut.z);
}

/// Full material sampling in local space
__inline__ __device__ bool sampleMat(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec3f &sample,
                                     owl::vec3f &dirOut) {
    if (sample.z < mat.diffuseWeight) {
        return sampleDiffuse(mat, dirIn, {sample.x, sample.y}, dirOut);
    } else if (sample.z < mat.diffuseWeight + mat.metalWeight) {
        return sampleMetal(mat, dirIn, {sample.x, sample.y}, dirOut);
    } else if (sample.z < mat.diffuseWeight + mat.metalWeight + mat.sheenWeight) {
        return sampleSheen(mat, dirIn, {sample.x, sample.y}, dirOut);
    } else {
        return sampleClearcoat(mat, dirIn, {sample.x, sample.y}, dirOut);
    }
    // return sampleMetal(mat, dirIn, {sample.x, sample.y}, dirOut);
    // return sampleLambertian(mat, dirIn, {sample.x, sample.y}, dirOut);
};

__inline__ __device__ float pdfMat(const MaterialResult &mat, const owl::vec3f &dirIn, const owl::vec3f &dirOut,
                                   const owl::vec3f &half) {
    if (dirIn.z <= 0.f || dirOut.z <= 0.f) {
        return 0.f;
    }
    float pdf = 0.f;
    pdf += pdfDiffuse(mat, dirIn, dirOut, half) * mat.diffuseWeight;
    pdf += pdfMetal(mat, dirIn, dirOut, half) * mat.metalWeight;
    pdf += pdfClearcoat(mat, dirIn, dirOut, half) * mat.clearcoatWeight;
    pdf += pdfSheen(mat, dirIn, dirOut, half) * mat.sheenWeight;
    return pdf;
    // return pdfMetal(mat, dirIn, dirOut, half);
    // return pdfLambertian(mat, dirIn, dirOut);
}

__inline__ __device__ owl::vec3f evalMat(const MaterialResult &mat, const owl::vec3f &dirIn,
                                         const owl::vec3f &dirOut, const owl::vec3f &half) {
    const owl::vec3f diffuse = mat.diffuseWeight * evalDiffuse(mat, dirIn, dirOut, half);
    const owl::vec3f sheen = mat.sheenWeight * evalSheen(mat, dirIn, dirOut, half);
    const owl::vec3f metal = mat.metalWeight * evalMetal(mat, dirIn, dirOut, half);
    const owl::vec3f clearcoat = mat.clearcoatWeight * evalClearcoat(mat, dirIn, dirOut, half);
    const owl::vec3f sum = diffuse + sheen + metal + clearcoat;
    return sum;
    // return evalMetal(mat, dirIn, dirOut, half);
    // return evalLambertian(mat, dirIn, dirOut);
}
