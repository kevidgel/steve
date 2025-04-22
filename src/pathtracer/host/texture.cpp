#include "pathtracer/host/texture.hpp"
#include "owl/Object.h"
#include "owl/common/math/vec.h"
#include "owl/owl.h"
#include "spdlog/spdlog.h"
#include "stb_image.h"

#include <filesystem>
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
