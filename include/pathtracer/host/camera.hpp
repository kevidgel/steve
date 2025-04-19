/**
 * @file camera.hpp
 * @brief Camera definition and helper functions
 */

#pragma once

#include "imgui.h"
#include "ImGuizmo.h"
#include "owl/common/math/AffineSpace.h"

struct Camera {
    owl::affine3f transform = {};
    float yaw = 0.f;
    float pitch = 0.f;
    owl::vec2f sensorSize = {1.f, 1.f};
    owl::vec2i resolution = {512, 512};
    float focalDist = 1.f;
    float apertureRadius = 0.f;
    int integrator = 0;

    [[nodiscard]] owl::affine3f xyaw() const { return owl::affine3f::rotate(transform.l.vy, yaw * M_PIf / 180.f); }

    [[nodiscard]] owl::affine3f xpitch() const {
        return owl::affine3f::rotate(transform.l.vx, pitch * M_PIf / 180.f);
    }

    [[nodiscard]] owl::affine3f xform() const { return transform * xyaw() * xpitch(); }

    void renderProperties() {
        {
            static int current = 0;
            const char* integrators[] = {"BRDF", "Direct Lighting", "BRDF + NEE with MIS (power)", "BRDF + NEE with MIS (balance)", "Debug"};
            constexpr int count = IM_ARRAYSIZE(integrators);
            if (ImGui::Combo("Integrator", &current, integrators, count)) {
                integrator = current;
            }
        }

        {
            float translation[3], rotation[3], scale[3];
            float tmp[16] = {transform.l.vx.x, transform.l.vx.y, transform.l.vx.z, 0.f,
                             transform.l.vy.x, transform.l.vy.y, transform.l.vy.z, 0.f,
                             transform.l.vz.x, transform.l.vz.y, transform.l.vz.z, 0.f,
                             transform.p.x,    transform.p.y,    transform.p.z,    0.f};
            ImGuizmo::DecomposeMatrixToComponents(tmp, translation, rotation, scale);
            ImGui::InputFloat3("Translation", translation);
            // ImGui::InputFloat3("Rotation", rotation);
            // ImGui::InputFloat3("Scale", scale);
            ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, tmp);
            const owl::vec3f vx = {tmp[0], tmp[1], tmp[2]};
            const owl::vec3f vy = {tmp[4], tmp[5], tmp[6]};
            const owl::vec3f vz = {tmp[8], tmp[9], tmp[10]};
            const owl::vec3f p = {tmp[12], tmp[13], tmp[14]};
            transform = owl::affine3f({vx, vy, vz}, p);
        }

        {
            ImGui::InputFloat("Yaw", &yaw);
            ImGui::InputFloat("Pitch", &pitch);
        }

        {
            float tmp[2] = {sensorSize.x, sensorSize.y};
            ImGui::InputFloat2("Sensor Size", tmp);
            sensorSize = {tmp[0], tmp[1]};
        }

        ImGui::InputFloat("Focal Dist.", &focalDist);
        ImGui::InputFloat("Apert. Radius", &apertureRadius);
    }
};

inline bool operator==(const Camera &lhs, const Camera &rhs) {
    return (lhs.transform == rhs.transform) && (lhs.yaw == rhs.yaw) && (lhs.pitch == rhs.pitch) &&
           (lhs.sensorSize == rhs.sensorSize) && (lhs.resolution == rhs.resolution) &&
           (lhs.focalDist == rhs.focalDist) && (lhs.apertureRadius == rhs.apertureRadius) &&
           (lhs.integrator == rhs.integrator);
}