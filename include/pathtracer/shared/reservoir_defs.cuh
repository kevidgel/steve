/**
 * @file reservoir_defs.cuh
 * @brief Shared defintions for reservoir buffers
 */

#pragma once

#include "light_defs.cuh"

struct Reservoir {
    bool valid;
    LightSampleInfo sample;
    float wSum;
    float W;
    uint count;
};
