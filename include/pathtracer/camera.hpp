/**
 * @file camera.hpp
 * @brief Camera definition and helper functions
 */

#pragma once

#include "owl/common/math/AffineSpace.h"

struct Camera {
    owl::affine3f transform = {};
    owl::affine3f yaw = {};
    owl::affine3f pitch = {};
    owl::vec2f sensorSize = {1.f, 1.f};
    owl::vec2i resolution = {512, 512};
    float focalDist = 1.f;
    float apertureRadius = 0.f;

    [[nodiscard]] owl::affine3f xform() const {
        return transform * yaw * pitch;
    }
};

inline bool operator==(const Camera &lhs, const Camera &rhs) {
    return (lhs.transform == rhs.transform) && (lhs.yaw == rhs.yaw) && (lhs.pitch == rhs.pitch) &&
           (lhs.sensorSize == rhs.sensorSize) && (lhs.resolution == rhs.resolution) &&
           (lhs.focalDist == rhs.focalDist) && (lhs.apertureRadius == rhs.apertureRadius);
}