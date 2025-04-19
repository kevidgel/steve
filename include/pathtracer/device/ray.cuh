#pragma once

#include "owl/common/math/random.h"

typedef enum {
    RayScattered,
    RayCancelled,
    RayMissed,
} IntersectEvent;

/// Global record per primary ray
struct Record {
    owl::LCG<4> random;
    IntersectEvent intersectEvent;
    struct {
        float t;
        owl::vec3f p;
        owl::vec3f gn;
        owl::vec3f sn;
        owl::vec2f uv;
        int mat;
        uint id;
        float area;
    } hitInfo /** This is only populated when RayScattered */;
    owl::vec3f emitted /** This is only populated when RayCancelled or RayMissed */;
};
