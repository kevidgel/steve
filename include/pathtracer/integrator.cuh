/**
 * @file integrator.cuh
 * @brief Definitions for integrator
 */
#pragma once

#include "material.cuh"
#include "owl/common/math/vec.h"

#include <cuda_runtime.h>

struct ScatterRecord {
    owl::vec3f dir;
    owl::vec3f p;
    float pdf; // w.r.t solid angle

    // light sample
    uint primI;
};

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

__constant__ LaunchParams optixLaunchParams;
