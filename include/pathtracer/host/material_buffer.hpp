/**
 * @file material_buffer.hpp
 * @brief Material buffer definitions
 */

#pragma once

#include "nlohmann/json.hpp"
#include "pathtracer/shared/material_defs.cuh"
#include "rapidobj/rapidobj.hpp"

#include <tiny_gltf.h>
#include <vector>

namespace robj = rapidobj;

class MaterialBuffer {
  public:
    MaterialBuffer() = default;
    void loadMtl(const robj::Result &scene, const std::filesystem::path &filename);
    void loadGltf(const tinygltf::Model &model, const std::filesystem::path &filename);

    /// Load material texture into buffer
    static bool loadTexMtl(const std::filesystem::path &filename, std::string relTexPath, cudaTextureObject_t &tex,
                           bool &hasTex);
    static bool loadTexGltf(const tinygltf::Model &model, int texIdx, cudaTextureObject_t &tex, bool &hasTex);
    /// Render material props
    void renderProperties();

    std::vector<Material> mats{};
    std::vector<std::string> names{};
    bool dirty = false;
};

namespace nlohmann {
/// Serializers for vec2i
template <> struct adl_serializer<MaterialBuffer> {
    static void from_json(const json &j, MaterialBuffer &buffer){

    };

    static void to_json(json &j, const MaterialBuffer &buffer) {}
};
} // namespace nlohmann
