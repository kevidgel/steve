/**
 * @file integrator.cuh
 * @brief Definitions for integrator
 */
#pragma once

#include "integrator_defs.cuh"
#include "owl/common/math/vec.h"
#include "sampling.cuh"

__constant__ LaunchParams optixLaunchParams;

#define RAY_EPS (1e-4f)
#define RAY_MAX (1e10f)

__inline__ __device__ owl::vec3f xfmPt(const affine3f_ &xform, const owl::vec3f &p) {
    return xform.vx * p.x + xform.vy * p.y + xform.vz * p.z + xform.p;
}

__inline__ __device__ owl::vec3f invXfmPt(const affine3f_ &xform, const owl::vec3f &p) {
    owl::vec3f d = p - xform.p;
    return owl::vec3f(dot(d, xform.vx), dot(d, xform.vy), dot(d, xform.vz));
}

__inline__ __device__ owl::vec3f xfmVec(const affine3f_ &xform, const owl::vec3f &v) {
    return xform.vx * v.x + xform.vy * v.y + xform.vz * v.z;
}

__inline__ __device__ owl::vec2f projectToScreen(const DeviceCamera &cam, const owl::vec3f &p) {
    owl::vec3f d = invXfmPt(cam.xform, p);

    float x = (d.x / d.z) * cam.focalDist;
    float y = (d.y / d.z) * cam.focalDist;

    float u = cam.resolution.x * (0.5f - x / cam.sensorSize.x);
    float v = cam.resolution.y * (y / cam.sensorSize.y + 0.5f);

    return {u, v};
}

__inline__ __device__ owl::Ray generateRay(const DeviceCamera& camera, float u, float v, const owl::vec2f &lensSample) {
    const owl::vec2f disk = camera.apertureRadius * randomInUnitDisk(lensSample);
    const owl::vec3f origin(disk.x, disk.y, 0.f);
    const owl::vec3f direction(camera.sensorSize.x * (0.5f - u / camera.resolution.x),
                               camera.sensorSize.y * (v / camera.resolution.y - 0.5f), camera.focalDist);
    const owl::vec3f tOrigin = xfmPt(camera.xform, origin);
    const owl::vec3f tDirection = xfmVec(camera.xform, direction - origin);
    return owl::Ray(tOrigin, tDirection, RAY_EPS, RAY_MAX);
}
