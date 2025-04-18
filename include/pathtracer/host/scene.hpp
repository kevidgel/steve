/**
 * @file scene.hpp
 * @brief Scene loader + parser
 */
#pragma once

#include "material_buffer.hpp"
#include "owl/common/math/vec.h"
#include "pathtracer/shared/material_defs.cuh"

#include <filesystem>
#include <vector>

/// A little redundant with RapidObj's Result object, but abstracts away some code
class SceneBuffer {
  public:
    explicit SceneBuffer(const std::filesystem::path& filename);

    // Attributes
    std::vector<owl::vec3f> verts;
    std::vector<owl::vec3ui> vertsI;
    std::vector<owl::vec3f> norms;
    std::vector<owl::vec3ui> normsI;
    std::vector<owl::vec2f> texCoords;
    std::vector<owl::vec3ui> texCoordsI;
    std::vector<int> matsI;

    // Lights
    std::vector<owl::vec3f> lightVerts;
    std::vector<owl::vec3ui> lightVertsI;
    std::vector<uint> lightPrimsI;

    // Materials
    MaterialBuffer materialBuffer;

    // Keep a copy of filename just in case
    const std::filesystem::path filename;
};

