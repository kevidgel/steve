/**
 * @file host_defs.hpp
 * @brief The purpose of this file is to prevent mixing .cuh and .hpp headers
 * .cuh headers should only be used by device code, .hpp with host code. This
 * comes with the drawback of needing to maintain two versions of our interf-
 * ace. The alternative is using header-guards in .cuh files, which is very
 * annoying.
 */

#pragma once

#include "camera.hpp"
#include "owl/common/math/vec.h"
#include "owl/owl.h"

#include <cuda_runtime.h>

/// From geometry.cuh
struct Lights {
    owl::vec3f *verts;
    owl::vec3ui *vertsI;
    uint *primsI; // Global primitive ids
    uint size;
};

struct TriangleMesh {
    owl::vec3f *verts;
    owl::vec3f *norms;
    owl::vec2f *texCoords;

    // Indices
    owl::vec3ui *vertsI;
    owl::vec3ui *normsI;
    owl::vec3ui *texCoordsI;

    // Per-face materialId
    int *matsI;
};

/// From material.cuh
struct Material {
    owl::vec3f baseColor;
    owl::vec3f emission;
    float roughness;
    float anisotropic;
    float subsurface;

    // Textures if available
    bool hasBaseColorTex = false;
    cudaTextureObject_t baseColorTex;
};

/// From integrator.cuh
struct affine3f_ {
    owl::vec3f p;
    owl::vec3f vx;
    owl::vec3f vy;
    owl::vec3f vz;
};

struct DeviceCamera {
    affine3f_ xform = {};
    owl::vec2f sensorSize = {1.f, 1.f};
    owl::vec2i resolution = {512, 512};
    float focalDist = 1.f;
    float apertureRadius = 0.f;
};

struct Frame {
    bool dirty = true;
    int id = 0;
    int accum = 1;
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
    Camera *camera;
};

struct MissProgData {
    owl::vec3f envColor;
    bool hasEnvMap;
    cudaTextureObject_t envMap;
};
