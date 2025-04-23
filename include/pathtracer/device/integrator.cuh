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

__inline__ __device__ owl::vec3f xfmPt(const affine3f_ &xform, const owl::vec3f &p) {
    return xform.vx * p.x + xform.vy * p.y + xform.vz * p.z + xform.p;
}

__inline__ __device__ owl::vec3f xfmVec(const affine3f_ &xform, const owl::vec3f &v) {
    return xform.vx * v.x + xform.vy * v.y + xform.vz * v.z;
}

__constant__ LaunchParams optixLaunchParams;
