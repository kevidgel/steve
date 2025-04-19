/**
 * @file material_defs.cuh
 * @brief Shared material definitions between host and device code
 */

#pragma once

#include "owl/common/math/vec.h"
#include "cuda_runtime.h"

__inline__ __host__ __device__ float luminance(const owl::vec3f &c) {
    return c.x * 0.212671f + c.y * 0.715160f + c.z * 0.072169f;
}

struct Material {
    bool hasMetallicTex;
    // no subsurfaceTex
    bool hasSpecularTex;
    bool hasRoughnessTex;
    // no specularTintTex
    // no anisotropicTex
    bool hasSheenTex;
    // no sheenTintTex
    // no clearcoatTex;
    // no clearcoatGlossTex;
    bool hasBaseColorTex;
    bool hasEmissiveTex;
    bool hasAlphaTex;
    bool hasBumpTex;
    bool hasNormalTex;

    cudaTextureObject_t metallicTex;
    // no subsurfaceTex
    cudaTextureObject_t specularTex;
    cudaTextureObject_t roughnessTex;
    // no specularTintTex
    // no anisotropicTex
    cudaTextureObject_t sheenTex;
    // no sheenTintTex
    // no clearcoatTex;
    // no clearcoatGlossTex;
    cudaTextureObject_t baseColorTex;
    cudaTextureObject_t emissiveTex;
    cudaTextureObject_t alphaTex;
    cudaTextureObject_t bumpTex;
    cudaTextureObject_t normalTex;

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
    owl::vec3f baseColor;
    owl::vec3f emission;
};
