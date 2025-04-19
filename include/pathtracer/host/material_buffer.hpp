/**
 * @file material_buffer.hpp
 * @brief Material buffer definitions
 */

#pragma once

#include "pathtracer/shared/material_defs.cuh"
#include "nlohmann/json.hpp"
#include "rapidobj/rapidobj.hpp"

#include <vector>

namespace robj = rapidobj;

struct MaterialDesc {
    std::string metallicTex;
    // no subsurfaceTex
    std::string specularTex;
    std::string roughnessTex;
    // no specularTintTex
    // no anisotropicTex
    std::string sheenTex;
    // no sheenTintTex
    // no clearcoatTex;
    // no clearcoatGlossTex;
    std::string baseColorTex;
    std::string emissiveTex;
    std::string alphaTex;
    std::string bumpTex;
    std::string normalTex;

    float metallic;
    float subsurface;
    float specular;
    float roughness;
    float specularTint;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    owl::vec3f baseColor;
    owl::vec3f emission;
};

class MaterialBuffer {
public:
    MaterialBuffer() = default;
    MaterialBuffer(const robj::Result &scene, const std::filesystem::path &filename);

    /// Load material texture into buffer
    static bool loadTex(const std::filesystem::path& filename, std::string relTexPath, cudaTextureObject_t& tex, bool& hasTex);
    /// Upload buffer to LaunchParams
    void uploadBuffer();

    std::vector<Material> mats {};
    std::vector<MaterialDesc> matDescs {};
};

namespace nlohmann {
/// Serializers for vec2i
template <> struct adl_serializer<MaterialBuffer> {
    static void from_json(const json &j, MaterialBuffer &buffer) {

    };

    static void to_json(json &j, const MaterialBuffer &buffer) {
    }
};
} // namespace nlohmann
