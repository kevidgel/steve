#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "pathtracer/host/scene.hpp"
#include "pathtracer/host/texture.hpp"

#include <rapidobj/rapidobj.hpp>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace robj = rapidobj;

SceneBuffer::SceneBuffer(const std::filesystem::path &filename) : filename(filename) {
    // Check existence
    if (!exists(filename)) {
        throw std::runtime_error(fmt::format("Scene file {} not found!", filename.string()));
    }

    auto ext = filename.extension();
    if (ext == ".obj") {
        loadObj(filename);
    } else if (ext == ".gltf") {
        loadGltf(filename, false);
    } else if (ext == ".glb") {
        loadGltf(filename, true);
    } else {
        throw std::runtime_error("Scene file has unrecognized extension. Must be either .obj/.glb/.gltf");
    }

    spdlog::info("Loaded file {}", filename.string());
    spdlog::info("Materials: {}", materialBuffer.mats.size());
    spdlog::info("Num Triangles: {}", vertsI.size());
    spdlog::info("Num Emissive Triangles: {}", lightVertsI.size());
}

void SceneBuffer::loadGltf(const std::filesystem::path &filename, bool isBinary) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;

    // Load model
    bool ok;
    if (!isBinary) {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    } else {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    }

    if (!warn.empty()) {
        spdlog::warn("{}", warn);
    }
    if (!err.empty()) {
        spdlog::error("{}", err);
    }
    if (!ok) {
        throw std::runtime_error(fmt::format("Scene file {} failed to load!", filename.string()));
    }

    for (const auto &mesh : model.meshes) {
        for (const auto &prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                throw std::runtime_error(
                    fmt::format("Scene file {} not triangulated! Offending mesh: {}", filename.string(), mesh.name));
            }

            // Material index
            unsigned matIndex = (prim.material >= 0 ? prim.material : 0);

            // Load positions
            auto itP = prim.attributes.find("POSITION");
            if (itP == prim.attributes.end()) {
                continue;
            }
            auto const &pAcc = model.accessors[itP->second];
            auto const &pView = model.bufferViews[pAcc.bufferView];
            auto const &pBuf = model.buffers[pView.buffer];
            size_t pOff = pView.byteOffset + pAcc.byteOffset;
            auto posPtr = reinterpret_cast<const float *>(pBuf.data.data() + pOff);
            size_t pCount = pAcc.count;

            auto basePos = static_cast<unsigned>(verts.size());
            verts.reserve(basePos + pCount);
            for (size_t i = 0; i < pCount; ++i) {
                verts.emplace_back(posPtr[3 * i + 0], posPtr[3 * i + 1], posPtr[3 * i + 2]);
            }

            // Load norms
            bool haveAttrNorm = false;
            const float *normPtr = nullptr;
            size_t nCount = 0;
            auto baseNorm = static_cast<unsigned>(norms.size());

            auto itN = prim.attributes.find("NORMAL");
            if (itN != prim.attributes.end()) {
                haveAttrNorm = true;
                auto const &nAcc = model.accessors[itN->second];
                auto const &nView = model.bufferViews[nAcc.bufferView];
                auto const &nBuf = model.buffers[nView.buffer];
                size_t nOff = nView.byteOffset + nAcc.byteOffset;
                normPtr = reinterpret_cast<const float *>(nBuf.data.data() + nOff);
                nCount = nAcc.count;

                norms.reserve(baseNorm + nCount);
                for (size_t i = 0; i < nCount; ++i) {
                    norms.emplace_back(normPtr[3 * i + 0], normPtr[3 * i + 1], normPtr[3 * i + 2]);
                }
            }

            // Load texcoords
            bool haveAttrUV = false;
            const float *uvPtr = nullptr;
            size_t uCount = 0;
            auto baseUV = (unsigned)texCoords.size();

            auto itU = prim.attributes.find("TEXCOORD_0");
            if (itU != prim.attributes.end()) {
                haveAttrUV = true;
                auto const &uAcc = model.accessors[itU->second];
                auto const &uView = model.bufferViews[uAcc.bufferView];
                auto const &uBuf = model.buffers[uView.buffer];
                size_t uOff = uView.byteOffset + uAcc.byteOffset;
                uvPtr = reinterpret_cast<const float *>(uBuf.data.data() + uOff);
                uCount = uAcc.count;

                texCoords.reserve(baseUV + uCount);
                for (size_t i = 0; i < uCount; ++i) {
                    texCoords.emplace_back(uvPtr[2 * i + 0], uvPtr[2 * i + 1]);
                }
            }

            // Load indices
            if (prim.indices >= 0) {
                auto const &iAcc = model.accessors[prim.indices];
                auto const &iView = model.bufferViews[iAcc.bufferView];
                auto const &iBuf = model.buffers[iView.buffer];
                size_t iOff = iView.byteOffset + iAcc.byteOffset;
                auto idxPtr = iBuf.data.data() + iOff;
                size_t iCount = iAcc.count;

                size_t triCnt = iCount / 3;
                vertsI.reserve(vertsI.size() + triCnt);
                normsI.reserve(normsI.size() + triCnt);
                texCoordsI.reserve(texCoordsI.size() + triCnt);

                auto handleTriangle = [&](unsigned i0, unsigned i1, unsigned i2) {
                    // matsI
                    matsI.push_back(static_cast<int>(matIndex));

                    // vertsI
                    vertsI.emplace_back(i0 + basePos, i1 + basePos, i2 + basePos);

                    // normsI
                    if (haveAttrNorm) {
                        normsI.emplace_back(i0 + baseNorm, i1 + baseNorm, i2 + baseNorm);
                    } else {
                        // compute gn
                        owl::vec3f A = verts[i0 + basePos];
                        owl::vec3f B = verts[i1 + basePos];
                        owl::vec3f C = verts[i2 + basePos];
                        owl::vec3f fn = normalize(cross(B - A, C - A));
                        // push three copies
                        auto fnBase = static_cast<unsigned>(norms.size());
                        norms.push_back(fn);
                        norms.push_back(fn);
                        norms.push_back(fn);
                        normsI.emplace_back(fnBase, fnBase + 1, fnBase + 2);
                    }

                    // texCoordsI
                    if (haveAttrUV) {
                        texCoordsI.emplace_back(i0 + baseUV, i1 + baseUV, i2 + baseUV);
                    } else {
                        auto uvBase = static_cast<unsigned>(texCoords.size());
                        // three (0,0)
                        texCoords.emplace_back(0.f, 0.f);
                        texCoords.emplace_back(0.f, 0.f);
                        texCoords.emplace_back(0.f, 0.f);
                        texCoordsI.emplace_back(uvBase, uvBase + 1, uvBase + 2);
                    }
                };

                // unpack index buffer
                switch (iAcc.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    auto idx16 = reinterpret_cast<const uint16_t *>(idxPtr);
                    for (size_t i = 0; i < iCount; i += 3) {
                        handleTriangle(idx16[i + 0], idx16[i + 1], idx16[i + 2]);
                    }
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    auto idx32 = reinterpret_cast<const uint32_t *>(idxPtr);
                    for (size_t i = 0; i < iCount; i += 3) {
                        handleTriangle(idx32[i + 0], idx32[i + 1], idx32[i + 2]);
                    }
                    break;
                }
                default:
                    throw std::runtime_error(fmt::format(
                        "Scene loading failed! Unsupported index format. Offending mesh: {}", mesh.name));
                }
            } else {
                throw std::runtime_error(
                    fmt::format("Scene loading failed! Please indexify your mesh! Offending mesh: {}", mesh.name));
            }
        }
    }

    // Load materials
    materialBuffer.loadGltf(model, filename);

    // Load lights
    for (size_t offsetMat = 0; offsetMat < matsI.size(); ++offsetMat) {
        int material_id = matsI[offsetMat];
        if (luminance(materialBuffer.mats[material_id].emission) > 0.01f) {
            lightPrimsI.push_back((int)offsetMat);
            lightEmission.push_back(materialBuffer.mats[material_id].emission);
        }
    }

    std::unordered_map<uint, uint> indexMap; // Map old to new index
    size_t offsetLight = 0;
    for (const auto &lightI : lightPrimsI) {
        const auto &vertI = vertsI[lightI];

        for (size_t i = 0; i < 3; i++) {
            if (indexMap.find(vertI[i]) == indexMap.end()) {
                lightVerts.push_back(verts[vertI[i]]);
                indexMap[vertI[i]] = offsetLight;
                offsetLight++;
            }
        }

        lightVertsI.emplace_back(indexMap[vertI.x], indexMap[vertI.y], indexMap[vertI.z]);
    }

    // Check if lights mesh is valid
    for (size_t i = 0; i < lightPrimsI.size(); i++) {
        bool xMatch = lightVerts[lightVertsI[i].x] == verts[vertsI[lightPrimsI[i]].x];
        bool yMatch = lightVerts[lightVertsI[i].y] == verts[vertsI[lightPrimsI[i]].y];
        bool zMatch = lightVerts[lightVertsI[i].z] == verts[vertsI[lightPrimsI[i]].z];
        bool isEmit = length(materialBuffer.mats[matsI[lightPrimsI[i]]].emission) > 0.f;
        if (!(xMatch && yMatch && zMatch && isEmit)) {
            spdlog::error("Check failed!");
        }
    }
}

void SceneBuffer::loadObj(const std::filesystem::path &filename) {
    // Eager load the scene
    robj::Result scene = robj::ParseFile(filename);
    if (scene.error) {
        throw std::runtime_error(fmt::format("Scene file {} parsing error! {}, {}", filename.string(),
                                             scene.error.line, scene.error.line_num));
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
    materialBuffer.loadMtl(scene, filename);

    size_t offsetMat = 0;
    for (const auto &shape : scene.shapes) {
        for (int material_id : shape.mesh.material_ids) {
            matsI[offsetMat] = material_id;
            if (luminance(materialBuffer.mats[material_id].emission) > 0.f) {
                lightPrimsI.push_back((int)offsetMat);
                lightEmission.push_back(materialBuffer.mats[material_id].emission);
            }
            ++offsetMat;
        }
    }

    std::unordered_map<uint, uint> indexMap; // Map old to new index
    size_t offsetLight = 0;
    for (const auto &lightI : lightPrimsI) {
        const auto &vertI = vertsI[lightI];

        for (size_t i = 0; i < 3; i++) {
            if (indexMap.find(vertI[i]) == indexMap.end()) {
                lightVerts.push_back(verts[vertI[i]]);
                indexMap[vertI[i]] = offsetLight;
                offsetLight++;
            }
        }

        lightVertsI.emplace_back(indexMap[vertI.x], indexMap[vertI.y], indexMap[vertI.z]);
    }

    // Check if lights mesh is valid
    for (size_t i = 0; i < lightPrimsI.size(); i++) {
        bool xMatch = lightVerts[lightVertsI[i].x] == verts[vertsI[lightPrimsI[i]].x];
        bool yMatch = lightVerts[lightVertsI[i].y] == verts[vertsI[lightPrimsI[i]].y];
        bool zMatch = lightVerts[lightVertsI[i].z] == verts[vertsI[lightPrimsI[i]].z];
        bool isEmit = length(materialBuffer.mats[matsI[lightPrimsI[i]]].emission) > 0.f;
        if (!(xMatch && yMatch && zMatch && isEmit)) {
            spdlog::error("Check failed!");
        }
    }
}
