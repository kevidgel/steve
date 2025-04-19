#include "pathtracer/host/material_buffer.hpp"
#include "fmt/format.h"
#include "pathtracer/host/texture.hpp"

#include <spdlog/spdlog.h>

owl::vec3f toOwlVec3f(const robj::Float3& val) {
    return {val[0], val[1], val[2]};
}

bool MaterialBuffer::loadTex(const std::filesystem::path& filename, std::string relTexPath, cudaTextureObject_t& tex, bool& hasTex) {
    std::replace(relTexPath.begin(), relTexPath.end(), '\\', '/');
    std::filesystem::path texPath = fmt::format("{}/{}", filename.parent_path().string(), relTexPath);
    hasTex = false;
    if (exists(texPath) && is_regular_file(texPath)) {
        auto texResult = loadImageCuda(texPath);
        if (texResult.has_value()) {
            tex = texResult.value();
            hasTex = true;
        }
    }
    return hasTex;
}

MaterialBuffer::MaterialBuffer(const robj::Result &scene, const std::filesystem::path& filename) {
    mats.resize(scene.materials.size());
    matDescs.resize(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const auto& objMat = scene.materials[i];
        Material mat;
        MaterialDesc matDesc;
        matDesc.metallic = mat.metallic = objMat.metallic;
        matDesc.subsurface = mat.subsurface = 0.0f; // no equiv
        matDesc.specular = mat.specular = luminance(toOwlVec3f(objMat.specular));
        matDesc.roughness = mat.roughness = objMat.roughness < 0.001 ? sqrt(2 / (objMat.shininess + 2)) : objMat.roughness;
        matDesc.specularTint = mat.specularTint = 0.f; // no equiv
        matDesc.anisotropic = mat.anisotropic = objMat.anisotropy;
        matDesc.sheen = mat.sheen = objMat.sheen;
        matDesc.sheenTint = mat.sheenTint = 0.f; // no equiv
        matDesc.clearcoat = mat.clearcoat = objMat.clearcoat_thickness;
        matDesc.clearcoatGloss = mat.clearcoatGloss = objMat.clearcoat_roughness;
        matDesc.baseColor = mat.baseColor = toOwlVec3f(objMat.diffuse);
        matDesc.emission = mat.emission = toOwlVec3f(objMat.emission);

        mat.metallic = std::clamp(mat.metallic, 0.f, 1.f);
        mat.subsurface = std::clamp(mat.subsurface, 0.f, 1.f);
        mat.specular = std::clamp(mat.specular, 0.f, 1.f);
        mat.roughness = std::clamp(mat.roughness, 0.f, 1.f);
        mat.specularTint = std::clamp(mat.specularTint, 0.f, 1.f);
        mat.anisotropic = std::clamp(mat.anisotropic, 0.f, 1.f);
        mat.sheen = std::clamp(mat.sheen, 0.f, 1.f);
        mat.sheenTint = std::clamp(mat.sheenTint, 0.f, 1.f);
        mat.clearcoat = std::clamp(mat.clearcoat, 0.f, 1.f);
        mat.clearcoatGloss = std::clamp(mat.clearcoatGloss, 1.f, 1.f);

        loadTex(filename, objMat.metallic_texname, mat.metallicTex, mat.hasMetallicTex);
        loadTex(filename, objMat.specular_texname, mat.specularTex, mat.hasSpecularTex);
        loadTex(filename, objMat.roughness_texname, mat.roughnessTex, mat.hasRoughnessTex);
        loadTex(filename, objMat.sheen_texname, mat.sheenTex, mat.hasSheenTex);
        loadTex(filename, objMat.diffuse_texname, mat.baseColorTex, mat.hasBaseColorTex);
        if (loadTex(filename, objMat.emissive_texname, mat.emissiveTex, mat.hasEmissiveTex)) {
            mat.emission = 0.01;
        }
        loadTex(filename, objMat.alpha_texname, mat.alphaTex, mat.hasAlphaTex);
        loadTex(filename, objMat.bump_texname, mat.bumpTex, mat.hasBumpTex);
        loadTex(filename, objMat.normal_texname, mat.normalTex, mat.hasNormalTex);

        matDesc.metallicTex = objMat.metallic_texname;
        matDesc.specularTex = objMat.specular_texname;
        matDesc.roughnessTex = objMat.roughness_texname;
        matDesc.sheenTex = objMat.sheen_texname;
        matDesc.baseColorTex = objMat.diffuse_texname;
        matDesc.emissiveTex = objMat.emissive_texname;
        matDesc.alphaTex = objMat.alpha_texname;
        matDesc.bumpTex = objMat.bump_texname;
        matDesc.normalTex = objMat.normal_texname;

        mats[i] = mat;
        matDescs[i] = matDesc;
    }
}

void MaterialBuffer::uploadBuffer() {
    spdlog::info("I do nothing!");
}
