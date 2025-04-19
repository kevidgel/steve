/**
 * @file texture.hpp
 * @brief Texture loading functions
 */

#pragma once

#include "../../../ext/owl/owl/Object.h"
#include "../../../ext/owl/owl/include/owl/owl.h"

#include <filesystem>
#include <optional>

float *getImageData(const std::filesystem::path &filename, int &width, int &height);
OWLTexture loadImageOwl(const std::filesystem::path &filename, const OWLContext &ctx);
std::optional<cudaTextureObject_t> loadImageCuda(const std::filesystem::path &filename);
