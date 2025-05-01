#include "pathtracer/host/parser.hpp"
#include "fstream"

#include <spdlog/spdlog.h>

using json = nlohmann::json;
std::unique_ptr<Camera> parseCamera(const json &j);
std::unique_ptr<SceneBuffer> parseScene(const json &j);
std::string parseEnv(const json &j);

Result parseFile(const std::filesystem::path &filename) {
    spdlog::info("Parsing {}", filename.string());
    std::ifstream f(filename);
    json j = json::parse(f);

    std::unique_ptr<Camera> camera = parseCamera(j);
    std::unique_ptr<SceneBuffer> scene = parseScene(j);
    std::string env = parseEnv(j);

    return {std::move(scene), std::move(camera), env};
}

std::unique_ptr<Camera> parseCamera(const json &j) {
    if (j.contains("camera")) {
        auto camera = std::make_unique<Camera>();
        const auto &cameraParams = j["camera"];

        camera->transform = cameraParams.value("transform", camera->transform);
        camera->yaw = cameraParams.value("yaw", camera->yaw);
        camera->pitch = cameraParams.value("pitch", camera->pitch);
        camera->sensorSize = cameraParams.value("sensor_size", camera->sensorSize);
        camera->resolution = cameraParams.value("resolution", camera->resolution);
        camera->focalDist = cameraParams.value("focal_dist", camera->focalDist);
        camera->apertureRadius = cameraParams.value("aperture_radius", camera->apertureRadius);
        return camera;
    }
    throw std::runtime_error("Camera not found!");
}

std::unique_ptr<SceneBuffer> parseScene(const json &j) {
    if (j.contains("scene")) {
        auto sceneDesc = j["scene"];
        if (sceneDesc.contains("path")) {
            std::filesystem::path path = sceneDesc["path"];
            return std::make_unique<SceneBuffer>(path);
        }
    }
    throw std::runtime_error("Scene not found!");
}
std::string parseEnv(const json &j) {
    if (j.contains("env")) {
        return j["env"];
    }
    return "";
}
