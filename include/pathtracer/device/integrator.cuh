/**
 * @file integrator.cuh
 * @brief Definitions for integrator
 */
#pragma once

#include "owl/common/math/vec.h"
#include "integrator_defs.cuh"

struct ScatterRecord {
    owl::vec3f dir;
    owl::vec3f p;
    float pdf; // w.r.t solid angle

    // light sample
    uint primI;
};

__constant__ LaunchParams optixLaunchParams;
