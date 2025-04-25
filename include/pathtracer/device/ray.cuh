#pragma once

#include "ray_defs.cuh"
#include "owl/common/math/random.h"

/// Global record per primary ray
struct RayInfo {
    owl::LCG<4> random;
    HitInfo hitInfo;
};
