/**
* @file sampling.cuh
* @brief Sampling routines
*/

#pragma once
#include "owl/common/math/vec.h"

#include <cuda_runtime.h>

__inline__ __device__ owl::vec3f randomCosineHemisphere(const owl::vec2f &sample) {
    const float phi = 2.f * M_PI * sample.x;
    const float cosTheta = sqrtf(sample.y);
    const float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
    const float x = cosf(phi) * sinTheta;
    const float y = sinf(phi) * sinTheta;
    const float z = cosTheta;
    return {x, y, z};
}

__inline__ __device__ owl::vec3f randomCosinePowerHemisphere(float exponent, const owl::vec2f &sample) {
    const float phi = 2.f * M_PI * sample.x;
    const float cosTheta = powf(sample.y, 1.f / (exponent + 1.f));
    const float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
    const float x = cosf(phi) * sinTheta;
    const float y = sinf(phi) * sinTheta;
    const float z = cosTheta;
    return {x, y, z};
}

__inline__ __device__ owl::vec2f randomInUnitDisk(const owl::vec2f &sample) {
    const float r = sqrtf(sample.x);
    const float phi = 2.f * M_PI * sample.y;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    return {x, y};
}