/**
 * @file texture.hpp
 * @brief Texture loading functions
 */

#pragma once

#include "owl/Object.h"
#include "owl/owl.h"
#include "pathtracer/shared/integrator_defs.cuh"

#include <filesystem>
#include <optional>

float *getImageData(const std::filesystem::path &filename, int &width, int &height);
OWLTexture loadImageOwl(const std::filesystem::path &filename, const OWLContext &ctx);
std::optional<cudaTextureObject_t> loadImageCuda(const std::filesystem::path &filename);
std::vector<Alias> buildEnvMapAlias(const std::filesystem::path &filename, int &width, int &height);
