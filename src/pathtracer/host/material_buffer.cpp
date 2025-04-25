#include "pathtracer/host/material_buffer.hpp"
#include "fmt/format.h"
#include "pathtracer/host/texture.hpp"

#include <imgui.h>
#include <spdlog/spdlog.h>

owl::vec3f toOwlVec3f(const robj::Float3 &val) { return {val[0], val[1], val[2]}; }

bool MaterialBuffer::loadTexMtl(const std::filesystem::path &filename, std::string relTexPath,
                                cudaTextureObject_t &tex, bool &hasTex) {
    hasTex = false;
    if (relTexPath.empty()) {
        return false;
    }
    std::replace(relTexPath.begin(), relTexPath.end(), '\\', '/');
    std::filesystem::path texPath = fmt::format("{}/{}", filename.parent_path().string(), relTexPath);
    if (exists(texPath) && is_regular_file(texPath)) {
        auto texResult = loadImageCuda(texPath);
        if (texResult.has_value()) {
            tex = texResult.value();
            hasTex = true;
        }
    } else {
        spdlog::error("Texture {} not found!", relTexPath);
    }
    return hasTex;
}

std::vector<uchar4> packImageRGBA(const tinygltf::Image &img) {
    if (!(img.component == 3 || img.component == 4)) {
        throw std::runtime_error("Only RGB/RGBA images supported");
    }
    int w = img.width;
    int h = img.height;
    std::vector<uchar4> hostPixels(w * h);

    const unsigned char *src = img.image.data();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            int srcOff = idx * img.component;
            uchar4 px;
            px.x = src[srcOff + 0];
            px.y = src[srcOff + 1];
            px.z = src[srcOff + 2];
            px.w = (img.component == 4 ? src[srcOff + 3] : 255);
            hostPixels[idx] = px;
        }
    }
    return hostPixels;
}

bool MaterialBuffer::loadTexGltf(const tinygltf::Model &model, int texIdx, cudaTextureObject_t &tex,
                                 bool &hasTex) {
    if (texIdx < 0) {
        hasTex = false;
        return false;
    };

    hasTex = true;
    const tinygltf::Texture &sourceTex = model.textures[texIdx];
    const tinygltf::Image &img = model.images[sourceTex.source];
    int w = img.width, h = img.height;
    auto hostPixels = packImageRGBA(img);

    cudaPitchedPtr dPitched = {};
    cudaMallocPitch(&dPitched.ptr, &dPitched.pitch, w * sizeof(uchar4), h);
    cudaMemcpy2D(dPitched.ptr, dPitched.pitch, hostPixels.data(), w * sizeof(uchar4), w * sizeof(uchar4), h,
                 cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dPitched.ptr;
    resDesc.res.pitch2D.width = w;
    resDesc.res.pitch2D.height = h;
    resDesc.res.pitch2D.pitchInBytes = dPitched.pitch;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    if (cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr) != cudaSuccess) {
        spdlog::error("Texture object creation of {} failed!", sourceTex.name);
    }
    return true;
}

void MaterialBuffer::loadMtl(const robj::Result &scene, const std::filesystem::path &filename) {
    mats.resize(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const auto &objMat = scene.materials[i];
        names.push_back(objMat.name);
        Material mat;
        mat.metallic = objMat.metallic;
        mat.subsurface = 0.0f; // no equiv
        mat.specular = luminance(toOwlVec3f(objMat.specular));
        mat.roughness = objMat.roughness < 0.001 ? sqrt(2 / (objMat.shininess + 2)) : objMat.roughness;
        mat.specularTint = 0.f; // no equiv
        mat.anisotropic = objMat.anisotropy;
        mat.sheen = objMat.sheen;
        mat.sheenTint = 0.f; // no equiv
        mat.clearcoat = objMat.clearcoat_thickness;
        mat.clearcoatGloss = objMat.clearcoat_roughness;
        mat.baseColor = toOwlVec3f(objMat.diffuse);
        mat.emission = toOwlVec3f(objMat.emission);
        mat.ior = 1.f;

        mat.metallic = std::clamp(mat.metallic, 0.f, 1.f);
        mat.subsurface = std::clamp(mat.subsurface, 0.f, 1.f);
        mat.specular = std::clamp(mat.specular, 0.f, 1.f);
        mat.roughness = std::clamp(mat.roughness, 0.f, 1.f);
        mat.specularTint = std::clamp(mat.specularTint, 0.f, 1.f);
        mat.anisotropic = std::clamp(mat.anisotropic, 0.f, 1.f);
        mat.sheen = std::clamp(mat.sheen, 0.f, 1.f);
        mat.sheenTint = std::clamp(mat.sheenTint, 0.f, 1.f);
        mat.clearcoat = std::clamp(mat.clearcoat, 0.f, 1.f);
        mat.clearcoatGloss = std::clamp(mat.clearcoatGloss, 0.f, 1.f);

        loadTexMtl(filename, objMat.metallic_texname, mat.metallicTex, mat.hasMetallicTex);
        loadTexMtl(filename, objMat.specular_texname, mat.specularTex, mat.hasSpecularTex);
        loadTexMtl(filename, objMat.roughness_texname, mat.roughnessTex, mat.hasRoughnessTex);
        loadTexMtl(filename, objMat.sheen_texname, mat.sheenTex, mat.hasSheenTex);
        loadTexMtl(filename, objMat.diffuse_texname, mat.baseColorTex, mat.hasBaseColorTex);
        if (loadTexMtl(filename, objMat.emissive_texname, mat.emissiveTex, mat.hasEmissiveTex)) {
            mat.emission = 0.01;
        }
        loadTexMtl(filename, objMat.alpha_texname, mat.alphaTex, mat.hasAlphaTex);
        loadTexMtl(filename, objMat.bump_texname, mat.bumpTex, mat.hasBumpTex);
        loadTexMtl(filename, objMat.normal_texname, mat.normalTex, mat.hasNormalTex);
        mat.hasMetallicRoughnessTex = false; // only in GLTF

        // matDesc.metallicTex = objMat.metallic_texname;
        // matDesc.specularTex = objMat.specular_texname;
        // matDesc.roughnessTex = objMat.roughness_texname;
        // matDesc.sheenTex = objMat.sheen_texname;
        // matDesc.baseColorTex = objMat.diffuse_texname;
        // matDesc.emissiveTex = objMat.emissive_texname;
        // matDesc.alphaTex = objMat.alpha_texname;
        // matDesc.bumpTex = objMat.bump_texname;
        // matDesc.normalTex = objMat.normal_texname;

        mats[i] = mat;
    }
}

owl::vec3f vectorToVec3f(const std::vector<double> &vec) {
    return {
        static_cast<float>(vec[0]),
        static_cast<float>(vec[1]),
        static_cast<float>(vec[2]),
    };
}

static bool getColorFactor(const tinygltf::Value &value, owl::vec3f &result) {
    int valid = 0;
    float color[3] = {0.0f, 0.0f, 0.0f};

    if (value.IsArray()) {
        if (value.Size() == 3) {
            for (int i = 0; i < 3; i++) {
                if (value.Get(i).IsNumber()) {
                    color[i] = float(value.Get(i).Get<double>());
                    valid++;
                } else if (value.Get(i).IsInt()) {
                    // probably 0 or 1.
                    color[i] = float(value.Get(i).Get<int>());
                    valid++;
                }
            }
        }
    }
    if (valid == 3) {
        result[0] = color[0];
        result[1] = color[1];
        result[2] = color[2];

        return true;
    }

    return false;
}

void MaterialBuffer::loadGltf(const tinygltf::Model &model, const std::filesystem::path &filename) {
    mats.resize(model.materials.size());
    size_t i = 0;
    for (const auto &sourceMat : model.materials) {
        names.push_back(sourceMat.name);
        const auto &pbr = sourceMat.pbrMetallicRoughness;
        Material mat;
        // MaterialDesc matDesc;

        mat.hasMetallicTex = false;
        mat.hasSpecularTex = false;
        mat.hasRoughnessTex = false;
        mat.hasSheenTex = false;
        mat.hasBaseColorTex = false;
        mat.hasEmissiveTex = false;
        mat.hasAlphaTex = false;
        mat.hasBumpTex = false;
        mat.hasNormalTex = false;
        mat.hasMetallicRoughnessTex = false;

        // Basecolor
        mat.baseColor = vectorToVec3f(pbr.baseColorFactor);
        // loadTexMtl(filename, "../test_tex.png", mat.baseColorTex, mat.hasBaseColorTex);
        loadTexGltf(model, pbr.baseColorTexture.index, mat.baseColorTex, mat.hasBaseColorTex);

        // Metalness
        mat.metallic = static_cast<float>(pbr.metallicFactor);
        // Roughness
        mat.roughness = static_cast<float>(pbr.roughnessFactor);
        // Weirdly gltf packs both parameters into one texture
        loadTexGltf(model, pbr.metallicRoughnessTexture.index, mat.metallicRoughnessTex,
                    mat.hasMetallicRoughnessTex);

        // Specular
        mat.specular = 0.04f * (1.f - mat.metallic); // Defaults if not found
        mat.specularTint = 0.f;
        if (auto it = sourceMat.extensions.find("KHR_materials_specular"); it != sourceMat.extensions.end()) {
            const tinygltf::Value &ext = it->second;
            if (ext.Has("specularFactor")) {
                mat.specular = (float)ext.Get("specularFactor").Get<double>();
            }
            if (ext.Has("specularColorFactor")) {
                owl::vec3f specularColorFactor;
                getColorFactor(ext.Get("specularColorFactor"), specularColorFactor);
                float l = (specularColorFactor.x + specularColorFactor.y + specularColorFactor.z) / 3.f;
                mat.specular = length(specularColorFactor - l);
            }
            if (ext.Has("specularTexture")) {
                int texIdx = ext.Get("specularTexture").Get("index").Get<int>();
                loadTexGltf(model, texIdx, mat.specularTex, mat.hasSpecularTex);
            }
            // Tint texture avail?
        }

        // Anisotropy
        mat.anisotropic = 0.f; // Defaults if not found
        if (auto it = sourceMat.extensions.find("KHR_materials_anisotropy"); it != sourceMat.extensions.end()) {
            const tinygltf::Value &ext = it->second;
            if (ext.Has("anisotropyStrength")) {
                mat.anisotropic = (float)ext.Get("anisotropyStrength").Get<double>();
            }
        }

        // Sheen
        mat.sheen = 0.f; // Defaults if not found
        mat.sheenTint = 0.f;
        if (auto it = sourceMat.extensions.find("KHR_materials_sheen"); it != sourceMat.extensions.end()) {
            const tinygltf::Value &ext = it->second;
            if (ext.Has("sheenRoughnessFactor")) {
                mat.sheen = (float)ext.Get("sheenRoughnessFactor").Get<double>();
            }
            if (ext.Has("sheenColorFactor")) {
                owl::vec3f sheenColorFactor;
                getColorFactor(ext.Get("sheenColorFactor"), sheenColorFactor);
                float l = (sheenColorFactor.x + sheenColorFactor.y + sheenColorFactor.z) / 3.f;
                mat.sheenTint = length(sheenColorFactor - l);
            }
            if (ext.Has("sheenColorTexture")) {
                int texIdx = ext.Get("sheenColorTexture").Get("index").Get<int>();
                loadTexGltf(model, texIdx, mat.sheenTex, mat.hasSheenTex);
            }
        }

        // Clearcoat
        mat.clearcoat = 0;
        mat.clearcoatGloss = 0;
        if (auto it = sourceMat.extensions.find("KHR_materials_clearcoat"); it != sourceMat.extensions.end()) {
            const tinygltf::Value &ext = it->second;
            if (ext.Has("clearcoatFactor")) {
                mat.clearcoat = (float)ext.Get("clearcoatFactor").Get<double>();
            }
            if (ext.Has("clearcoatRoughnessFactor")) {
                mat.clearcoat = (float)ext.Get("clearcoatRoughnessFactor").Get<double>();
            }
        }

        // Emission
        mat.emission = vectorToVec3f(sourceMat.emissiveFactor);
        loadTexGltf(model, sourceMat.emissiveTexture.index, mat.emissiveTex, mat.hasEmissiveTex);

        // Subsurface
        mat.subsurface = 0.0f; // no equiv

        // IOR
        mat.ior = 1.f;

        mats[i] = mat;
        i++;
    }
}

static bool vectorGetter(void *data, int idx, const char **out) {
    auto &vec = *static_cast<std::vector<std::string> *>(data);
    if (idx < 0 || idx >= (int)vec.size()) {
        return false;
    }
    *out = vec[idx].c_str();
    return true;
}

void MaterialBuffer::renderProperties() {
    static int current = 0;
    ImGui::Combo("Name", &current, vectorGetter, (void *)&names, (int)names.size());

    auto matCopy = mats.at(current);
    auto &mat = mats.at(current);

    ImGui::Indent(10.f);
    if (ImGui::CollapsingHeader("Base Color")) {
        ImGui::Checkbox("Enable Texture:##basecolor_tex", &mat.hasBaseColorTex);
        float baseColor[3] = {mat.baseColor.x, mat.baseColor.y, mat.baseColor.z};
        ImGui::PushItemWidth(290);
        ImGuiColorEditFlags flags = ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_NoSidePreview |
                                    ImGuiColorEditFlags_NoSmallPreview;
        ImGui::ColorPicker3("##colorwheel", baseColor, flags);
        mat.baseColor = {baseColor[0], baseColor[1], baseColor[2]};
        ImGui::PopItemWidth();
    }
    if (ImGui::CollapsingHeader("Metallic-Roughness")) {
        ImGui::Checkbox("Enable Roughness Texture:", &mat.hasMetallicTex);
        ImGui::Checkbox("Enable Metallic Texture:", &mat.hasRoughnessTex);
        ImGui::SliderFloat("Metallic##metallic", &mat.metallic, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Roughness##roughness", &mat.roughness, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Anisotropy##anisotropic", &mat.anisotropic, 0.f, 1.f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Specular")) {
        ImGui::Checkbox("Enable Texture:##specular_tex", &mat.hasSpecularTex);
        ImGui::SliderFloat("Specular##specular", &mat.specular, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Tint##specular_tint", &mat.specularTint, 0.f, 1.f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Glass")) {
        ImGui::SliderFloat("IOR##ior", &mat.ior, 1.0, 1.8f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Sheen")) {
        ImGui::Checkbox("Enable Texture:##sheen_tex", &mat.hasSheenTex);
        ImGui::SliderFloat("Sheen##sheen", &mat.sheen, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Tint##sheen_tint", &mat.sheenTint, 0.f, 1.f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Clearcoat")) {
        ImGui::SliderFloat("Clearcoat##clearcoat", &mat.clearcoat, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Gloss##clearcoat_gloss", &mat.clearcoatGloss, 0.f, 1.f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Fake Subsurface")) {
        ImGui::SliderFloat("Subsurface##subsurface", &mat.subsurface, 0.f, 1.f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Emissive")) {
        float emit[3] = {mat.emission.x, mat.emission.y, mat.emission.z};
        ImGui::InputFloat3("Emission##emission", emit);
        mat.emission = {emit[0], emit[1], emit[2]};
    }
    ImGui::Unindent(10.f);

    if (matCopy != mat) {
        dirty = true;
    }
}