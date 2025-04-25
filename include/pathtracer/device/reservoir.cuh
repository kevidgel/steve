/**
 * @file reservoir.cuh
 * @brief Reservoir definitions and functions. Header-only.
 */

#pragma once

#include "pathtracer/shared/reservoir_defs.cuh"

#include <cuda_runtime.h>

__inline__ __device__ void updateReservoir(Reservoir &resr, const LightSampleInfo &lightSample, const float w, const float sample) {
    if (w <= 0) return;

    resr.sumWeights += w;
    resr.count += 1;

    if (sample * resr.sumWeights < w) {
        resr.sample = lightSample;
    }
}