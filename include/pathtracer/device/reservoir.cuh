/**
 * @file reservoir.cuh
 * @brief Reservoir definitions and functions. Header-only.
 */

#pragma once

#include "pathtracer/shared/reservoir_defs.cuh"

#include <cuda_runtime.h>

__inline__ __device__ void initReservoir(Reservoir &resr) {
    resr.sumWeights = 0.f;
    resr.count = 0;
}

__inline__ __device__ void updateReservoir(Reservoir &resr, const Candidate &lightSample, float sample) {
    float w = lightSample.radiance.x + lightSample.radiance.y + lightSample.radiance.z;
    if (w <= 0) return;

    resr.sumWeights += w;
    resr.count += 1;

    if (sample * resr.sumWeights < w) {
        resr.chosen = lightSample;
    }
}

__inline__ __device__ void fetchTemporal(int x, int y) {

}
