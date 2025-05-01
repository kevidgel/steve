#include "pathtracer/host/texture.hpp"
#include "owl/Object.h"
#include "owl/common/math/vec.h"
#include "owl/owl.h"
#include "pathtracer/shared/material_defs.cuh"
#include "spdlog/spdlog.h"
#include "stb_image.h"

#include <filesystem>
#include <numeric>
#include <optional>

float *getImageData(const std::filesystem::path &filename, int &width, int &height) {
    int channels;
    stbi_set_flip_vertically_on_load(1);
    float *data = stbi_loadf(filename.c_str(), &width, &height, &channels, 4);
    if (!data) {
        spdlog::error("Failed to load texture at {}", filename.string());
        return nullptr;
    }

    return data;
}

OWLTexture loadImageOwl(const std::filesystem::path &filename, const OWLContext &ctx) {
    int width, height;
    const float *data = getImageData(filename, width, height);
    if (!data) {
        return nullptr;
    }

    auto texData = reinterpret_cast<const owl::vec4f *>(data);

    OWLTexture texture = owlTexture2DCreate(ctx, OWL_TEXEL_FORMAT_RGBA32F, width, height, texData,
                                            OWL_TEXTURE_NEAREST, OWL_TEXTURE_CLAMP);
    stbi_image_free((void *)data);
    return texture;
}

std::optional<cudaTextureObject_t> loadImageCuda(const std::filesystem::path &filename) {
    int width, height;
    float *data = getImageData(filename, width, height);
    if (!data) {
        return std::nullopt;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cudaArray;
    cudaMallocArray(&cudaArray, &channelDesc, width, height);

    cudaMemcpy2DToArray(cudaArray, 0, 0, data, width * sizeof(float4), width * sizeof(float4), height,
                        cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    if (cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr) != cudaSuccess) {
        spdlog::error("Texture object creation of {} failed!", filename.string());
    }

    stbi_image_free(data);
    return texObj;
}

std::vector<Alias> buildEnvMapAlias(const std::filesystem::path &filename, int &width, int &height) {
    float *data = stbi_loadf(filename.string().c_str(), &width, &height, nullptr, 4);
    if (!data) {
        throw std::runtime_error("Failed to load env map: " + filename.string());
    }

    const int N = width * height;
    std::vector<float> weights(N);

    for (int y = 0; y < height; ++y) {
        float theta = M_PI * (y + 0.5f) / float(height);
        float sinTheta = std::sin(theta);
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            owl::vec3f c{data[4 * idx + 0], data[4 * idx + 1], data[4 * idx + 2]};
            weights[idx] = luminance(c) * sinTheta;
        }
    }
    stbi_image_free(data);

    float sumW = std::accumulate(weights.begin(), weights.end(), 0.0f);
    std::vector<Alias> table(N);
    if (sumW <= 0.0f) {
        // uniform fallback
        for (int i = 0; i < N; ++i) {
            table[i].pdf = 1.0f / float(N);
        }
    } else {
        for (int i = 0; i < N; ++i) {
            table[i].pdf = weights[i] / sumW;
        }
    }

    std::vector<float> P(N);
    std::deque<int> small, large;
    for (int i = 0; i < N; ++i) {
        P[i] = table[i].pdf * float(N);
        if (P[i] < 1.0f) {
            small.push_back(i);
        } else {
            large.push_back(i);
        }
    }

    while (!small.empty() && !large.empty()) {
        int s = small.back();
        small.pop_back();
        int l = large.back();
        large.pop_back();

        table[s].prob = P[s];
        table[s].alias = l;

        P[l] = P[l] - (1.0f - P[s]);
        if (P[l] < 1.0f) {
            small.push_back(l);
        } else {
            large.push_back(l);
        }
    }

    for (int i : large) {
        table[i].prob = 1.0f;
        table[i].alias = i;
    }
    for (int i : small) {
        table[i].prob = 1.0f;
        table[i].alias = i;
    }

    return table;
}