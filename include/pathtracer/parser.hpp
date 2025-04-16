// /**
//  * @file parser.hpp
//  * @brief Parses scene description
//  */
//
#pragma once

#include "pathtracer/camera.hpp"
#include "pathtracer/scene.hpp"
#include "nlohmann/json.hpp"
#include "owl/common/math/AffineSpace.h"

using json = nlohmann::json;

struct Result {
    std::unique_ptr<SceneBuffer> scene;
    std::unique_ptr<Camera> camera;
};

/// Parses a file into a usable scene format
/// Currently this only supports one .obj file at a time
Result parseFile(const std::filesystem::path &filename);
