/**
 * @file reservoir_defs.cuh
 * @brief Shared defintions for reservoir buffers
 */

#pragma once

#include "light_defs.cuh"

struct Reservoir {
    bool valid;
    LightSampleInfo sample;
    float sumWeights;
    uint count;
};
