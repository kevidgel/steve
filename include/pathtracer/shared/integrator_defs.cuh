/**
 * @file integrator_defs.cuh
 * @brief Shared definitions for integrator between host and device code
 */

#pragma once

#include "material_defs.cuh"
#include "geometry_defs.cuh"
#include "owl/owl.h"
#include "owl/common/math/vec.h"

#include <cuda_runtime.h>

struct affine3f_ {
    owl::vec3f p;
    owl::vec3f vx;
    owl::vec3f vy;
    owl::vec3f vz;
};

struct DeviceCamera {
    affine3f_ xform;
    owl::vec2f sensorSize;
    owl::vec2i resolution;
    float focalDist;
    float apertureRadius;
    int integrator;
};

struct Frame {
    bool dirty;
    int id;
    int accum;
};

struct LaunchParams {
    Frame frame;
    DeviceCamera camera;
    OptixTraversableHandle world;
    Material *mats;
    Lights lights;
};

struct RayGenData {
    owl::vec4f *pboPtr;
    owl::vec2i pboSize;
};

struct MissProgData {
    owl::vec3f envColor;
    bool hasEnvMap;
    cudaTextureObject_t envMap;
};
