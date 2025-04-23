/**
 * @file reservoir_defs.cuh
 * @brief Shared defintions for reservoir buffers
 */

#pragma once

#include "owl/common/math/vec.h"

struct Candidate {

    float pdf;
};

struct Reservoir {
    Candidate chosen;
    float sumWeights;
    uint count;
};