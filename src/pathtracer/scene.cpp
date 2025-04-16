#include "pathtracer/scene.hpp"
#include "pathtracer/texture.hpp"

#include <glm/detail/type_mat4x2.hpp>
#include <rapidobj/rapidobj.hpp>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace robj = rapidobj;

owl::vec3f toOwlVec3f(const robj::Float3& val) {
    return {val[0], val[1], val[2]};
}

SceneBuffer::SceneBuffer(const std::filesystem::path &filename) : filename(filename) {
    // Check existence
    if (!exists(filename)) {
        throw std::runtime_error(fmt::format("Scene file {} not found!", filename.string()));
    }

    // Eager load the scene
    robj::Result scene = robj::ParseFile(filename);
    if (scene.error) {
        throw std::runtime_error(fmt::format("Scene file {} parsing error! {}, {}", filename.string(), scene.error.line, scene.error.line_num));
    }

    // Triangulate
    if (!Triangulate(scene)) {
        throw std::runtime_error(fmt::format("Error triangulating {}!", filename.string()));
    }
    // First resize vectors to desired size
    size_t nI = 0;
    size_t nMats = 0;
    for (const auto &shape : scene.shapes) {
        nI += shape.mesh.indices.size();
        nMats += shape.mesh.material_ids.size();
    }
    vertsI.resize(nI);
    normsI.resize(nI);
    texCoordsI.resize(nI);
    matsI.resize(nMats);

    // Fill the indices
    size_t offset = 0;
    for (const auto &shape : scene.shapes) {
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            auto i1 = shape.mesh.indices[i];
            auto i2 = shape.mesh.indices[i + 1];
            auto i3 = shape.mesh.indices[i + 2];

            vertsI[offset] = owl::vec3ui(i1.position_index, i2.position_index, i3.position_index);
            normsI[offset] = owl::vec3ui(i1.normal_index, i2.normal_index, i3.normal_index);
            texCoordsI[offset] = owl::vec3ui(i1.texcoord_index, i2.texcoord_index, i3.texcoord_index);

            // Next prim
            ++offset;
        }
    }

    /// Fill in actual data
    size_t nVerts = scene.attributes.positions.size();
    verts.resize(nVerts / 3);
    for (size_t i = 0; i < nVerts; i += 3) {
        verts[i / 3] = owl::vec3f(scene.attributes.positions[i], scene.attributes.positions[i + 1],
                                  scene.attributes.positions[i + 2]);
    }

    size_t nNorms = scene.attributes.normals.size();
    norms.resize(nNorms / 3);
    for (size_t i = 0; i < nNorms; i += 3) {
        norms[i / 3] = owl::vec3f(scene.attributes.normals[i], scene.attributes.normals[i + 1],
                                  scene.attributes.normals[i + 2]);
    }

    size_t nTexCoords = scene.attributes.texcoords.size();
    texCoords.resize(nTexCoords / 2);
    for (size_t i = 0; i < nTexCoords; i += 2) {
        texCoords[i / 2] = owl::vec2f(scene.attributes.texcoords[i], scene.attributes.texcoords[i + 1]);
    }

    // Parse materials
    mats.resize(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const auto& objMat = scene.materials[i];
        Material mat;
        mat.baseColor = toOwlVec3f(objMat.diffuse);
        mat.emission = toOwlVec3f(objMat.emission);
        mat.roughness = objMat.roughness < 0.001 ? 1.f : objMat.roughness;
        mat.roughness = i % 3 == 1 ? 0.38f : mat.roughness;
        mat.anisotropic = objMat.anisotropy;
        mat.subsurface = 0.1f;

        // Attempt to load texture
        // TODO: Abstract this
        std::string relTexPath = objMat.diffuse_texname;
        std::replace(relTexPath.begin(), relTexPath.end(), '\\', '/');
        std::filesystem::path texPath = fmt::format("{}/{}", filename.parent_path().string(), relTexPath);
        if (exists(texPath) && is_regular_file(texPath)) {
            auto texResult = loadImageCuda(texPath);
            if (texResult.has_value()) {
                mat.baseColorTex = texResult.value();
                mat.hasBaseColorTex = true;
            } else {
                mat.hasBaseColorTex = false;
            }
        } else {
            mat.hasBaseColorTex = false;
        }
        mats[i] = mat;
    }

    size_t offsetMat = 0;
    for (const auto &shape : scene.shapes) {
        for (int material_id : shape.mesh.material_ids) {
            matsI[offsetMat] = material_id;
            if (length(mats[material_id].emission) > 0.01f) {
                lightPrimsI.push_back((int)offsetMat);
            }
            ++offsetMat;
        }
    }

    std::unordered_map<uint, uint> indexMap; // Map old to new index
    size_t offsetLight = 0;
    for (const auto &lightI : lightPrimsI) {
        const auto& vertI = vertsI[lightI];

        for (size_t i = 0; i < 3; i++) {
            if (indexMap.find(vertI[i]) == indexMap.end()) {
                lightVerts.push_back(verts[vertI[i]]);
                indexMap[vertI[i]] = offsetLight;
                offsetLight++;
            }
        }

        lightVertsI.emplace_back(
            indexMap[vertI.x],
            indexMap[vertI.y],
            indexMap[vertI.z]
        );
    }

    for (size_t i = 0; i < lightPrimsI.size(); i++) {
        bool xMatch = lightVerts[lightVertsI[i].x] == verts[vertsI[lightPrimsI[i]].x];
        bool yMatch = lightVerts[lightVertsI[i].y] == verts[vertsI[lightPrimsI[i]].y];
        bool zMatch = lightVerts[lightVertsI[i].z] == verts[vertsI[lightPrimsI[i]].z];
        bool isEmit = length(mats[matsI[lightPrimsI[i]]].emission) > 0.f;
        if (!(xMatch && yMatch && zMatch && isEmit)) {
            spdlog::error("Check failed!");
        }
    }

    spdlog::info("Loaded file {}", filename.string());
    spdlog::info("Shapes: {}", scene.shapes.size());
    spdlog::info("Materials: {}", mats.size());
    spdlog::info("Num Triangles: {}", vertsI.size());
    spdlog::info("Num Emissive Triangles: {}", lightVertsI.size());
}
