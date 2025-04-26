/**
 * @file reservoir.cuh
 * @brief Reservoir definitions and functions. Header-only.
 */

#pragma once

#include "reservoir_defs.cuh"

#include <cuda_runtime.h>

__inline__ __device__ void mergeReservoirs(Reservoir &to, const Reservoir &from, const float sample) {
    if (from.valid) {
        to.wSum += from.wSum;
        to.count += from.count;
        if (sample < (from.wSum / to.wSum)) {
            to.sample = from.sample;
        }
    }
}

__inline__ __device__ void updateReservoir(Reservoir &resr, const LightSampleInfo &lightSample, const float weight,
                                           const uint count, const float sample) {
    resr.count += count;
    resr.wSum += weight;
    if (sample < (weight / resr.wSum)) {
        resr.sample = lightSample;
    }
}

__inline__ __device__ void initReservoir(Reservoir &resr) {
    resr.valid = true;
    resr.wSum = 0;
    resr.count = 0;
}
