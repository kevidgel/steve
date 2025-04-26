/**
 * @file light_defs.cuh
 * @brief Shared definitions for lights sampling
 */

#pragma once

#include "owl/common/math/vec.h"

struct LightSampleInfo {
    owl::vec3f p;
    owl::vec3f gn;
    owl::vec3f emission; // NOTE!!! empty until visibility test
    float pdf; // NOTE!!! w.r.t solid angle!!!! We may have to change this.
    float areaPdf;
    uint primI; // ID in global buffer (NOT in lights buffer)
};
