#include "pathtracer/parser.hpp"
#include "fstream"

#include <spdlog/spdlog.h>

using json = nlohmann::json;
std::unique_ptr<Camera> parseCamera(const json &j);
std::unique_ptr<SceneBuffer> parseScene(const json &j);

Result parseFile(const std::filesystem::path &filename) {
    spdlog::info("Parsing {}", filename.string());
    std::ifstream f(filename);
    json j = json::parse(f);

    std::unique_ptr<Camera> camera = parseCamera(j);
    std::unique_ptr<SceneBuffer> scene = parseScene(j);

    return {std::move(scene), std::move(camera)};
}

namespace nlohmann {
/// Serializers for vec2i
template <> struct adl_serializer<owl::vec2i> {
    static void from_json(const json &j, owl::vec2i &vec) {
        if (j.is_object()) {
            throw std::runtime_error("Error parsing vector, received json object");
        }

        const size_t n = j.size();
        if (n == 1) {
            vec = owl::vec2i(j.get<int>());
        } else if (n != 2) {
            throw std::runtime_error("Error parsing vector, incorrect size");
        }

        for (size_t i = 0; i < n; ++i) {
            j.at(i).get_to(vec[i]);
        }
    };

    static void to_json(json &j, owl::vec2i &vec) { j = json{vec.x, vec.y}; }
};

/// Serializers for vec2f
template <> struct adl_serializer<owl::vec2f> {
    static void from_json(const json &j, owl::vec2f &vec) {
        if (j.is_object()) {
            throw std::runtime_error("Error parsing vector, received json object");
        }

        const size_t n = j.size();
        if (n == 1) {
            vec = owl::vec2f(j.get<float>());
        } else if (n != 2) {
            throw std::runtime_error("Error parsing vector, incorrect size");
        }

        for (size_t i = 0; i < n; ++i) {
            j.at(i).get_to(vec[i]);
        }
    };

    static void to_json(json &j, owl::vec2f &vec) { j = json{vec.x, vec.y}; }
};

/// Serializers for vec3f
template <> struct adl_serializer<owl::vec3f> {
    static void from_json(const json &j, owl::vec3f &vec) {
        if (j.is_object()) {
            throw std::runtime_error("Error parsing vector, received json object");
        }

        const size_t n = j.size();
        if (n == 1) {
            vec = owl::vec3f(j.get<float>());
        } else if (n != 3) {
            throw std::runtime_error("Error parsing vector, incorrect size");
        }

        for (size_t i = 0; i < n; ++i) {
            j.at(i).get_to(vec[i]);
        }
    };

    static void to_json(json &j, owl::vec3f &vec) { j = json{vec.x, vec.y, vec.z}; }
};

/// Serializers for affine3f
template <> struct adl_serializer<owl::affine3f> {
    static void from_json(const json &j, owl::affine3f &mat) {
        if (j.count("o") || j.count("x") || j.count("y") || j.count("z")) {
            owl::vec3f o(0, 0, 0), x(1, 0, 0), y(0, 1, 0), z(0, 0, 1);
            o = j.value("o", o);
            x = j.value("x", x);
            y = j.value("y", y);
            z = j.value("z", z);
            mat = owl::affine3f({x, y, z}, o);
        } else if (j.is_array() && j.size() == 4) {
            for (const auto &element : j) {
                if (!element.is_array() || element.size() != 3) {
                    throw std::runtime_error("Error parsing affine transformation");
                }
            }

            owl::linear3f l(1.f);
            j.at(0).get_to(l.vx);
            j.at(1).get_to(l.vy);
            j.at(2).get_to(l.vz);
            owl::vec3f p(0.f);
            j.at(3).get_to(p);
            mat = owl::affine3f(l, p);
        } else {
            throw std::runtime_error("Error parsing affine transformation");
        }
    }

    static void to_json(json &j, const owl::affine3f &mat) {
        j = json{
            {mat.l.vx.x, mat.l.vx.y, mat.l.vx.z},
            {mat.l.vy.x, mat.l.vy.y, mat.l.vy.z},
            {mat.l.vz.x, mat.l.vz.y, mat.l.vz.z},
            {   mat.p.x,    mat.p.y,    mat.p.z},
        };
    }
};
} // namespace nlohmann

std::unique_ptr<Camera> parseCamera(const json &j) {
    if (j.contains("camera")) {
        auto camera = std::make_unique<Camera>();
        const auto &cameraParams = j["camera"];

        camera->transform = cameraParams.value("transform", camera->transform);
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
