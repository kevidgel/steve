/**
 * @file geometry.cuh
 * @brief Geometry definitions and helper functions
 */

#pragma once

#include "geometry_defs.cuh"
#include "owl/common/math/vec.h"
#include "owl/owl_device.h"
#include "ray.cuh"

/// Hit program
OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)() {
    const auto &self = owl::getProgramData<TriangleMesh>();
    auto &prd = owl::getPRD<Record>();

    const owl::vec3f rayOrigin = optixGetWorldRayOrigin();
    const owl::vec3f rayDir = optixGetWorldRayDirection();
    const int primId = optixGetPrimitiveIndex();
    const owl::vec2f bary = optixGetTriangleBarycentrics();

    // Compute gn
    const owl::vec3ui &vertI = self.vertsI[primId];
    const owl::vec3f &v0 = self.verts[vertI.x];
    const owl::vec3f &v1 = self.verts[vertI.y];
    const owl::vec3f &v2 = self.verts[vertI.z];

    owl::vec3f gn = cross(v1 - v0, v2 - v0);
    prd.hitInfo.area = length(gn) * 0.5f;
    gn = normalize(gn);

    const owl::vec3ui &normI = self.normsI[primId];
    const owl::vec3f &n0 = self.norms[normI.x];
    const owl::vec3f &n1 = self.norms[normI.y];
    const owl::vec3f &n2 = self.norms[normI.z];
    const owl::vec3f sn = normalize((1.f - bary.x - bary.y) * n0 + bary.x * n1 + bary.y * n2);

    const owl::vec3ui &texCoordI = self.texCoordsI[primId];
    const owl::vec2f &t0 = self.texCoords[texCoordI.x];
    const owl::vec2f &t1 = self.texCoords[texCoordI.y];
    const owl::vec2f &t2 = self.texCoords[texCoordI.z];
    const owl::vec2f uv = ((1.f - bary.x - bary.y) * t0) + bary.x * t1 + bary.y * t2;

    // For now, return the normal and cancel the ray
    prd.hitInfo.t = optixGetRayTmax();
    prd.hitInfo.p = rayOrigin + prd.hitInfo.t * rayDir;
    prd.hitInfo.gn = gn;
    prd.hitInfo.sn = sn;
    prd.hitInfo.uv = uv;
    prd.hitInfo.mat = self.matsI[primId];
    prd.hitInfo.id = primId;
    prd.emitted = 0.0f;
    prd.intersectEvent = RayScattered;
}
