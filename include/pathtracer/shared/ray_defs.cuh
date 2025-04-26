/**
 * @file gbuffer_defs.cuh
 * @brief GBuffer definitions for deferred shading
 */

#pragma once

#include "material_defs.cuh"
#include "owl/common/math/vec.h"

typedef enum {
    RayScattered,
    RayCancelled,
    RayMissed,
} IntersectEvent;

struct HitInfo {
    IntersectEvent intersectEvent;
    float t;
    owl::vec3f dirIn;
    owl::vec3f p;
    owl::vec3f gn;
    owl::vec3f sn;
    owl::vec2f uv;
    int mat;
    uint id;
    float area;
    owl::vec3f emitted;
};

struct GBufferInfo {
    HitInfo hitInfo;
    MaterialResult mat;
    owl::vec2f motion;
};
