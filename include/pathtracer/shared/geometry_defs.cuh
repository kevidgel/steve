/**
 * @file geometry_defs.cuh
 * @brief Shared geometry definitions between host and device
 */

#pragma once

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
