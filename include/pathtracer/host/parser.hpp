// /**
//  * @file parser.hpp
//  * @brief Parses scene description
//  */
//
#pragma once

#include "nlohmann/json.hpp"
#include "pathtracer/host/camera.hpp"
#include "scene.hpp"

using json = nlohmann::json;

struct Result {
    std::unique_ptr<SceneBuffer> scene;
    std::unique_ptr<Camera> camera;
    std::string env;
};

/// Parses a file into a usable scene format
/// Currently this only supports one .obj file at a time
Result parseFile(const std::filesystem::path &filename);

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
