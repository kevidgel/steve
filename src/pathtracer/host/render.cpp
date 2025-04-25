#include "pathtracer/host/render.hpp"
#include "glm/glm.hpp"
#include "imgui.h"
#include "ImGuizmo.h"
#include "integrator.ptx.hpp"
#include "restir_integrator.ptx.hpp"
#include "gbuffer.ptx.hpp"
#include "pathtracer/host/texture.hpp"

#include <cuda_gl_interop.h>
#include <spdlog/spdlog.h>

namespace Screen {
const int NUM_VERTICES = 8;
owl::vec3f vertices[NUM_VERTICES] = {
    {-1.f, -1.f, -1.f},
    {+1.f, -1.f, -1.f},
    {-1.f, +1.f, -1.f},
    {+1.f, +1.f, -1.f},
    {-1.f, -1.f, +1.f},
    {+1.f, -1.f, +1.f},
    {-1.f, +1.f, +1.f},
    {+1.f, +1.f, +1.f}
};

const int NUM_INDICES = 12;
owl::vec3i indices[NUM_INDICES] = {
    {0, 1, 3},
    {2, 3, 0},
    {5, 7, 6},
    {5, 6, 4},
    {0, 4, 5},
    {0, 5, 1},
    {2, 3, 7},
    {2, 7, 6},
    {1, 5, 7},
    {1, 7, 3},
    {4, 0, 2},
    {4, 2, 6}
};

const char *vertex_shader_source = R"(
    #version 460 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;
    layout (location = 2) in vec2 aTexCoord;

    out vec3 ourColor;
    out vec2 TexCoord;

    void main()
    {
        gl_Position = vec4(aPos, 1.0);
        ourColor = aColor;
        TexCoord = aTexCoord;
    }
    )";

const char *fragment_shader_source = R"(
    #version 460 core
    out vec4 FragColor;

    in vec3 ourColor;
    in vec2 TexCoord;

    uniform sampler2D texture1;
    uniform float num_samples;

    void main()
    {
        vec4 texColor = texture(texture1, TexCoord);
        FragColor = texColor / num_samples;
    }
    )";

float screen_vertices[] = {
    // positions          // colors           // texture coords
    1.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top right
    1.0f,  -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom right
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
    -1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f  // top left
};

unsigned int screen_indices[] = {
    0, 1, 2, // first triangle
    0, 2, 3  // second triangle
};
} // namespace Screen

void Render::logInfo(const std::string &log) { spdlog::info("Renderer: {}", log); }

Render::Render(std::unique_ptr<SceneBuffer> scene, std::unique_ptr<Camera> camera_,
               const std::filesystem::path &envFilename)
    : camera(*camera_) {
    // First initialize GL stuff
    gl.shader = new Shader(Screen::vertex_shader_source, Screen::fragment_shader_source);

    glGenVertexArrays(1, &gl.vao);
    glGenBuffers(1, &gl.vbo);
    glGenBuffers(1, &gl.ebo);

    glBindVertexArray(gl.vao);
    glBindBuffer(GL_ARRAY_BUFFER, gl.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Screen::screen_vertices), Screen::screen_vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Screen::screen_indices), Screen::screen_indices, GL_STATIC_DRAW);

    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // Color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Texcoords
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Create pbo
    glGenBuffers(1, &gl.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.pbo);

    // Initialize buffer data
    glBufferData(GL_PIXEL_UNPACK_BUFFER, camera.resolution.x * camera.resolution.y * sizeof(float4), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Generate textures
    glGenTextures(1, &gl.dispTex);
    glBindTexture(GL_TEXTURE_2D, gl.dispTex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, camera.resolution.x, camera.resolution.y, 0, GL_RGBA, GL_FLOAT,
                 nullptr);

    gl.shader->use();
    gl.shader->set_int("texture1", 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register PBO with CUDA
    if (cudaGraphicsGLRegisterBuffer(&pbo, gl.pbo, cudaGraphicsMapFlagsNone) != cudaSuccess) {
        throw std::runtime_error("Failed to register PBO with CUDA");
    }

    void *devPboPtr;
    size_t devPboSize;
    cudaGraphicsMapResources(1, &pbo, nullptr);
    cudaGraphicsResourceGetMappedPointer(&devPboPtr, &devPboSize, pbo);

    // Initialize OWL
    owl.ctx = owlContextCreate(nullptr, 1);
    owl.lightingModule = owlModuleCreate(owl.ctx, reinterpret_cast<const char *>(ShaderSources::integrator_ptx_source));
    owl.gBufferModule = owlModuleCreate(owl.ctx, reinterpret_cast<const char *>(ShaderSources::gbuffer_ptx_source));
    owl.reSTIRModule = owlModuleCreate(owl.ctx, reinterpret_cast<const char *>(ShaderSources::restir_integrator_ptx_source));

    // Initialize geometries (Only triangles for now)
    OWLVarDecl triMeshVars[] = {
        {"verts", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh, verts)},
        {"norms", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh, norms)},
        {"texCoords", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh, texCoords)},
        {"vertsI", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh, vertsI)},
        {"normsI", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh, normsI)},
        {"texCoordsI", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh, texCoordsI)},
        {"matsI", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh, matsI)},
        {nullptr}
    };

    // Create the mesh type
    owl.mesh = owlGeomTypeCreate(owl.ctx, OWL_TRIANGLES, sizeof(TriangleMesh), triMeshVars, -1);

    // Set closest hit program for triangle mesh
    owlGeomTypeSetClosestHit(owl.mesh, 0, owl.lightingModule, "TriangleMesh");
    owlGeomTypeSetClosestHit(owl.mesh, 0, owl.gBufferModule, "TriangleMesh");
    owlGeomTypeSetClosestHit(owl.mesh, 0, owl.reSTIRModule, "TriangleMesh");

    // Build shader programs for geometries
    owlBuildPrograms(owl.ctx);

    // Build scene (only verts + norms for now)
    owl.vertsBuffer = owlDeviceBufferCreate(owl.ctx, OWL_FLOAT3, scene->verts.size(), scene->verts.data());
    owl.vertsIBuffer = owlDeviceBufferCreate(owl.ctx, OWL_UINT3, scene->vertsI.size(), scene->vertsI.data());
    owl.normsBuffer = owlDeviceBufferCreate(owl.ctx, OWL_FLOAT3, scene->norms.size(), scene->norms.data());
    owl.normsIBuffer = owlDeviceBufferCreate(owl.ctx, OWL_UINT3, scene->normsI.size(), scene->normsI.data());
    owl.texCoordsBuffer =
        owlDeviceBufferCreate(owl.ctx, OWL_FLOAT2, scene->texCoords.size(), scene->texCoords.data());
    owl.texCoordsIBuffer =
        owlDeviceBufferCreate(owl.ctx, OWL_UINT3, scene->texCoordsI.size(), scene->texCoordsI.data());
    owl.matsIBuffer = owlDeviceBufferCreate(owl.ctx, OWL_INT, scene->matsI.size(), scene->matsI.data());

    owl.triMeshGeom = owlGeomCreate(owl.ctx, owl.mesh);
    owlTrianglesSetVertices(owl.triMeshGeom, owl.vertsBuffer, scene->verts.size(), sizeof(owl::vec3f), 0);
    owlTrianglesSetIndices(owl.triMeshGeom, owl.vertsIBuffer, scene->vertsI.size(), sizeof(owl::vec3ui), 0);

    owlGeomSetBuffer(owl.triMeshGeom, "verts", owl.vertsBuffer);
    owlGeomSetBuffer(owl.triMeshGeom, "vertsI", owl.vertsIBuffer);
    owlGeomSetBuffer(owl.triMeshGeom, "norms", owl.normsBuffer);
    owlGeomSetBuffer(owl.triMeshGeom, "normsI", owl.normsIBuffer);
    owlGeomSetBuffer(owl.triMeshGeom, "texCoords", owl.texCoordsBuffer);
    owlGeomSetBuffer(owl.triMeshGeom, "texCoordsI", owl.texCoordsIBuffer);
    owlGeomSetBuffer(owl.triMeshGeom, "matsI", owl.matsIBuffer);

    // Two level IAS
    owl.triMeshGroup = owlTrianglesGeomGroupCreate(owl.ctx, 1, &owl.triMeshGeom);
    owlGroupBuildAccel(owl.triMeshGroup);
    owl.world = owlInstanceGroupCreate(owl.ctx, 1, &owl.triMeshGroup);
    owlGroupBuildAccel(owl.world);

    // Get env map
    owl.envMap = loadImageOwl(envFilename, owl.ctx);

    // Create materials buffer
    owl.matsBuffer = owlDeviceBufferCreate(owl.ctx, OWL_USER_TYPE(Material), scene->materialBuffer.mats.size(),
                                           scene->materialBuffer.mats.data());

    // Create lights buffers
    owl.lightsVertsBuffer =
        owlDeviceBufferCreate(owl.ctx, OWL_FLOAT3, scene->lightVerts.size(), scene->lightVerts.data());
    owl.lightsVertsIBuffer =
        owlDeviceBufferCreate(owl.ctx, OWL_UINT3, scene->lightVertsI.size(), scene->lightVertsI.data());
    owl.lightsPrimsIBuffer =
        owlDeviceBufferCreate(owl.ctx, OWL_UINT, scene->lightPrimsI.size(), scene->lightPrimsI.data());

    // Create gBuffer + reservoirs
    owl.gBuffer[0] = owlDeviceBufferCreate(owl.ctx, OWL_USER_TYPE(GBufferInfo),
                                             camera.resolution.x * camera.resolution.y, nullptr);
    owl.gBuffer[1] = owlDeviceBufferCreate(owl.ctx, OWL_USER_TYPE(GBufferInfo),
                                             camera.resolution.x * camera.resolution.y, nullptr);
    owl.reservoir[0] = owlDeviceBufferCreate(owl.ctx, OWL_USER_TYPE(Reservoir),
                                             camera.resolution.x * camera.resolution.y, nullptr);
    owl.reservoir[1] = owlDeviceBufferCreate(owl.ctx, OWL_USER_TYPE(Reservoir),
                                             camera.resolution.x * camera.resolution.y, nullptr);

    // Create launch params
    OWLVarDecl launchParamsVars[] = {
        {"frame.dirty", OWL_BOOL, OWL_OFFSETOF(LaunchParams, frame.dirty)},
        {"frame.id", OWL_INT, OWL_OFFSETOF(LaunchParams, frame.id)},
        {"frame.accum", OWL_INT, OWL_OFFSETOF(LaunchParams, frame.accum)},
        {"prevCamera.xform.p", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, prevCamera.xform.p)},
        {"prevCamera.xform.vx", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, prevCamera.xform.vx)},
        {"prevCamera.xform.vy", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, prevCamera.xform.vy)},
        {"prevCamera.xform.vz", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, prevCamera.xform.vz)},
        {"prevCamera.sensorSize", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, prevCamera.sensorSize)},
        {"prevCamera.resolution", OWL_INT2, OWL_OFFSETOF(LaunchParams, prevCamera.resolution)},
        {"prevCamera.focalDist", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, prevCamera.focalDist)},
        {"prevCamera.apertureRadius", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, prevCamera.apertureRadius)},
        {"prevCamera.integrator", OWL_INT, OWL_OFFSETOF(LaunchParams, prevCamera.integrator)},
        {"camera.xform.p", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.xform.p)},
        {"camera.xform.vx", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.xform.vx)},
        {"camera.xform.vy", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.xform.vy)},
        {"camera.xform.vz", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.xform.vz)},
        {"camera.sensorSize", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, camera.sensorSize)},
        {"camera.resolution", OWL_INT2, OWL_OFFSETOF(LaunchParams, camera.resolution)},
        {"camera.focalDist", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, camera.focalDist)},
        {"camera.apertureRadius", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, camera.apertureRadius)},
        {"camera.integrator", OWL_INT, OWL_OFFSETOF(LaunchParams, camera.integrator)},
        {"world", OWL_GROUP, OWL_OFFSETOF(LaunchParams, world)},
        {"mats", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, mats)},
        {"lights.verts", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, lights.verts)},
        {"lights.vertsI", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, lights.vertsI)},
        {"lights.primsI", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, lights.primsI)},
        {"lights.size", OWL_UINT, OWL_OFFSETOF(LaunchParams, lights.size)},
        {"curReservoir", OWL_UINT, OWL_OFFSETOF(LaunchParams, curReservoir)},
        {"gBuffer0", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, gBuffer0)},
        {"gBuffer1", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, gBuffer1)},
        {"reservoir0", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, reservoir0)},
        {"reservoir1", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, reservoir1)},
        {nullptr},
    };

    owl::affine3f xform = camera.xform();
    owl.launchParams = owlParamsCreate(owl.ctx, sizeof(LaunchParams), launchParamsVars, -1);
    owlParamsSet1b(owl.launchParams, "frame.dirty", frame.dirty);
    owlParamsSet1i(owl.launchParams, "frame.id", frame.id);
    owlParamsSet1i(owl.launchParams, "frame.accum", frame.accum);
    owlParamsSet3f(owl.launchParams, "prevCamera.xform.p", xform.p.x, xform.p.y, xform.p.z);
    owlParamsSet3f(owl.launchParams, "prevCamera.xform.vx", xform.l.vx.x, xform.l.vx.y, xform.l.vx.z);
    owlParamsSet3f(owl.launchParams, "prevCamera.xform.vy", xform.l.vy.x, xform.l.vy.y, xform.l.vy.z);
    owlParamsSet3f(owl.launchParams, "prevCamera.xform.vz", xform.l.vz.x, xform.l.vz.y, xform.l.vz.z);
    owlParamsSet2f(owl.launchParams, "prevCamera.sensorSize", camera.sensorSize.x, camera.sensorSize.y);
    owlParamsSet2i(owl.launchParams, "prevCamera.resolution", camera.resolution.x, camera.resolution.y);
    owlParamsSet1f(owl.launchParams, "prevCamera.focalDist", camera.focalDist);
    owlParamsSet1f(owl.launchParams, "prevCamera.apertureRadius", camera.apertureRadius);
    owlParamsSet1i(owl.launchParams, "prevCamera.integrator", camera.integrator);
    owlParamsSet3f(owl.launchParams, "camera.xform.p", xform.p.x, xform.p.y, xform.p.z);
    owlParamsSet3f(owl.launchParams, "camera.xform.vx", xform.l.vx.x, xform.l.vx.y, xform.l.vx.z);
    owlParamsSet3f(owl.launchParams, "camera.xform.vy", xform.l.vy.x, xform.l.vy.y, xform.l.vy.z);
    owlParamsSet3f(owl.launchParams, "camera.xform.vz", xform.l.vz.x, xform.l.vz.y, xform.l.vz.z);
    owlParamsSet2f(owl.launchParams, "camera.sensorSize", camera.sensorSize.x, camera.sensorSize.y);
    owlParamsSet2i(owl.launchParams, "camera.resolution", camera.resolution.x, camera.resolution.y);
    owlParamsSet1f(owl.launchParams, "camera.focalDist", camera.focalDist);
    owlParamsSet1f(owl.launchParams, "camera.apertureRadius", camera.apertureRadius);
    owlParamsSet1i(owl.launchParams, "camera.integrator", camera.integrator);
    owlParamsSetGroup(owl.launchParams, "world", owl.world);
    owlParamsSetBuffer(owl.launchParams, "mats", owl.matsBuffer);
    owlParamsSetBuffer(owl.launchParams, "lights.verts", owl.lightsVertsBuffer);
    owlParamsSetBuffer(owl.launchParams, "lights.vertsI", owl.lightsVertsIBuffer);
    owlParamsSetBuffer(owl.launchParams, "lights.primsI", owl.lightsPrimsIBuffer);
    owlParamsSet1ui(owl.launchParams, "lights.size", scene->lightVertsI.size());
    owlParamsSet1ui(owl.launchParams, "curReservoir", 0);
    owlParamsSetBuffer(owl.launchParams, "gBuffer0", owl.gBuffer[0]);
    owlParamsSetBuffer(owl.launchParams, "gBuffer1", owl.gBuffer[1]);
    owlParamsSetBuffer(owl.launchParams, "reservoir0", owl.reservoir[0]);
    owlParamsSetBuffer(owl.launchParams, "reservoir1", owl.reservoir[1]);

    // Create miss program
    OWLVarDecl missProgVars[] = {
        {"envColor", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, envColor)},
        {"hasEnvMap", OWL_BOOL, OWL_OFFSETOF(MissProgData, hasEnvMap)},
        {"envMap", OWL_TEXTURE, OWL_OFFSETOF(MissProgData, envMap)},
        {nullptr},
    };

    owl.lightingMissProg = owlMissProgCreate(owl.ctx, owl.lightingModule, "Miss", sizeof(MissProgData), missProgVars, -1);
    owlMissProgSet3f(owl.lightingMissProg, "envColor", {0.8f, 0.8f, 0.8f});
    owlMissProgSet1b(owl.lightingMissProg, "hasEnvMap", (owl.envMap != nullptr));
    owlMissProgSetTexture(owl.lightingMissProg, "envMap", owl.envMap);

    owl.reSTIRMissProg = owlMissProgCreate(owl.ctx, owl.reSTIRModule, "Miss2", sizeof(MissProgData), missProgVars, -1);
    owlMissProgSet3f(owl.reSTIRMissProg, "envColor", {0.8f, 0.8f, 0.8f});
    owlMissProgSet1b(owl.reSTIRMissProg, "hasEnvMap", (owl.envMap != nullptr));
    owlMissProgSetTexture(owl.reSTIRMissProg, "envMap", owl.envMap);

    owl.gBufferMissProg = owlMissProgCreate(owl.ctx, owl.gBufferModule, "Miss", sizeof(MissProgData), missProgVars, -1);
    owlMissProgSet3f(owl.gBufferMissProg, "envColor", {0.8f, 0.8f, 0.8f});
    owlMissProgSet1b(owl.gBufferMissProg, "hasEnvMap", (owl.envMap != nullptr));
    owlMissProgSetTexture(owl.gBufferMissProg, "envMap", owl.envMap);

    OWLVarDecl geometryPassVars[] = {
        {"pboSize", OWL_INT2, OWL_OFFSETOF(GeometryPassData, pboSize)},
        {nullptr}
    };

    owl.gBufferRayGen = owlRayGenCreate(owl.ctx, owl.gBufferModule, "GeometryPass", sizeof(GeometryPassData), geometryPassVars, -1);
    owlRayGenSet2i(owl.gBufferRayGen, "pboSize", camera.resolution.x, camera.resolution.y);

    OWLVarDecl rayGenVars[] = {
        {"pboPtr", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, pboPtr)},
        {"pboSize", OWL_INT2, OWL_OFFSETOF(RayGenData, pboSize)},
        {nullptr}
    };

    owl.lightingRayGen = owlRayGenCreate(owl.ctx, owl.lightingModule, "RayGen", sizeof(RayGenData), rayGenVars, -1);
    owlRayGenSetPointer(owl.lightingRayGen, "pboPtr", devPboPtr);
    owlRayGenSet2i(owl.lightingRayGen, "pboSize", camera.resolution.x, camera.resolution.y);

    owl.reSTIRRayGen = owlRayGenCreate(owl.ctx, owl.reSTIRModule, "RayGen", sizeof(RayGenData), rayGenVars, -1);
    owlRayGenSetPointer(owl.reSTIRRayGen, "pboPtr", devPboPtr);
    owlRayGenSet2i(owl.reSTIRRayGen, "pboSize", camera.resolution.x, camera.resolution.y);

    spdlog::info("Building programs, pipeline, and SBT...");
    owlBuildPrograms(owl.ctx);
    owlBuildPipeline(owl.ctx);
    owlBuildSBT(owl.ctx);

    cudaGraphicsUnmapResources(1, &pbo, nullptr);

    materialBuffer = scene->materialBuffer;
    oldCamera = camera;
}

Render::~Render() {
    owlGroupRelease(owl.world);
    owlGroupRelease(owl.triMeshGroup);
    owlGeomRelease(owl.triMeshGeom);
    owlBufferRelease(owl.matsBuffer);
    owlBufferRelease(owl.matsIBuffer);
    owlBufferRelease(owl.launchParamsBuffer);
    owlBufferRelease(owl.vertsBuffer);
    owlBufferRelease(owl.vertsIBuffer);
    owlBufferRelease(owl.normsBuffer);
    owlBufferRelease(owl.normsIBuffer);
    owlBufferRelease(owl.texCoordsBuffer);
    owlBufferRelease(owl.texCoordsIBuffer);
    owlModuleRelease(owl.lightingModule);
    owlRayGenRelease(owl.lightingRayGen);
    owlContextDestroy(owl.ctx);

    /********** Cleanup GL **********/
    glDeleteVertexArrays(1, &gl.vao);
    glDeleteBuffers(1, &gl.vbo);
    glDeleteBuffers(1, &gl.ebo);
    glDeleteBuffers(1, &gl.pbo);
    glDeleteTextures(1, &gl.dispTex);
    delete gl.shader;
}

/// Control camera
void Render::moveCamera(const CameraAction &action, float speed) {
    ImGuiIO &io = ImGui::GetIO();
    float delta = io.DeltaTime;
    float inc = speed * delta;
    owl::affine3f &transform = camera.transform;

    switch (action) {
    case MoveUp:
        transform.p += transform.l.vy * inc;
        break;
    case MoveDown:
        transform.p -= transform.l.vy * inc;
        break;
    case MoveRight:
        transform.p -= xfmVector(camera.xyaw(), transform.l.vx) * inc;
        break;
    case MoveLeft:
        transform.p += xfmVector(camera.xyaw(), transform.l.vx) * inc;
        break;
    case MoveForward:
        transform.p += xfmVector(camera.xyaw(), transform.l.vz) * inc;
        break;
    case MoveBackward:
        transform.p -= xfmVector(camera.xyaw(), transform.l.vz) * inc;
        break;
    case RotateLeft:
        camera.yaw += 10 * inc;
        break;
    case RotateRight:
        camera.yaw -= 10 * inc;
        break;
    case RotateUp:
        camera.pitch -= 10 * inc;
        break;
    case RotateDown:
        camera.pitch += 10 * inc;
        break;
    case FocalInc: {
        const float oldFocalDist = camera.focalDist;
        camera.focalDist += inc / 4;
        const float change = camera.focalDist / oldFocalDist;
        camera.sensorSize *= change;
    } break;
    case FocalDec: {
        const float oldFocalDist = camera.focalDist;
        camera.focalDist -= inc / 4;
        const float change = camera.focalDist / oldFocalDist;
        camera.sensorSize *= change;
    } break;
    case AptInc:
        camera.apertureRadius += inc / 1000;
        break;
    case AptDec:
        camera.apertureRadius -= inc / 1000;
        break;
    }
}

/// Update state
void Render::update() {
    if (accumFrames) {
        frame.dirty = (oldCamera != camera);
    } else {
        frame.dirty = true;
    }

    if (materialBuffer.dirty) {
        owlBufferUpload(owl.matsBuffer, materialBuffer.mats.data(), 0, materialBuffer.mats.size());
        frame.dirty = true;
        materialBuffer.dirty = false;
    }

    frame.accum = frame.dirty ? 1 : frame.accum + 1;
    frame.id++;

    owlParamsSet1b(owl.launchParams, "frame.dirty", frame.dirty);
    owlParamsSet1i(owl.launchParams, "frame.id", frame.id);
    owlParamsSet1i(owl.launchParams, "frame.accum", frame.accum);

    // Flip reservoirs
    curReservoir = 1 - curReservoir;
    owlParamsSet1ui(owl.launchParams, "curReservoir", curReservoir);

    const owl::affine3f oldXform = oldCamera.xform();
    owlParamsSet3f(owl.launchParams, "prevCamera.xform.p", oldXform.p.x, oldXform.p.y, oldXform.p.z);
    owlParamsSet3f(owl.launchParams, "prevCamera.xform.vx", oldXform.l.vx.x, oldXform.l.vx.y, oldXform.l.vx.z);
    owlParamsSet3f(owl.launchParams, "prevCamera.xform.vy", oldXform.l.vy.x, oldXform.l.vy.y, oldXform.l.vy.z);
    owlParamsSet3f(owl.launchParams, "prevCamera.xform.vz", oldXform.l.vz.x, oldXform.l.vz.y, oldXform.l.vz.z);
    owlParamsSet2f(owl.launchParams, "prevCamera.sensorSize", oldCamera.sensorSize.x, oldCamera.sensorSize.y);
    owlParamsSet2i(owl.launchParams, "prevCamera.resolution", oldCamera.resolution.x, oldCamera.resolution.y);
    owlParamsSet1f(owl.launchParams, "prevCamera.focalDist", oldCamera.focalDist);
    owlParamsSet1f(owl.launchParams, "prevCamera.apertureRadius", oldCamera.apertureRadius);
    owlParamsSet1i(owl.launchParams, "prevCamera.integrator", oldCamera.integrator);

    const owl::affine3f xform = camera.xform();
    owlParamsSet3f(owl.launchParams, "camera.xform.p", xform.p.x, xform.p.y, xform.p.z);
    owlParamsSet3f(owl.launchParams, "camera.xform.vx", xform.l.vx.x, xform.l.vx.y, xform.l.vx.z);
    owlParamsSet3f(owl.launchParams, "camera.xform.vy", xform.l.vy.x, xform.l.vy.y, xform.l.vy.z);
    owlParamsSet3f(owl.launchParams, "camera.xform.vz", xform.l.vz.x, xform.l.vz.y, xform.l.vz.z);
    owlParamsSet2f(owl.launchParams, "camera.sensorSize", camera.sensorSize.x, camera.sensorSize.y);
    owlParamsSet2i(owl.launchParams, "camera.resolution", camera.resolution.x, camera.resolution.y);
    owlParamsSet1f(owl.launchParams, "camera.focalDist", camera.focalDist);
    owlParamsSet1f(owl.launchParams, "camera.apertureRadius", camera.apertureRadius);
    owlParamsSet1i(owl.launchParams, "camera.integrator", camera.integrator);
}

/// Launch kernel and display results per frame
void Render::render() {
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGuiIO &io = ImGui::GetIO();
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();

    const float toolbarWidth = 300.f;
    const ImVec2 toolbarPos =
        ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - toolbarWidth, viewport->WorkPos.y);
    ImGui::SetNextWindowPos(toolbarPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(toolbarWidth, viewport->WorkPos.y + viewport->WorkSize.y - toolbarPos.y));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;

    ImGui::Begin("Properties", nullptr, flags);
    if (ImGui::CollapsingHeader("Render")) {
        ImGui::Text("Total samples: %d", frame.accum);
        ImGui::Checkbox("Enable frame accum.", &accumFrames);
    }
    if (ImGui::CollapsingHeader("Camera")) {
        camera.renderProperties();
    }
    if (ImGui::CollapsingHeader("Materials")) {
        materialBuffer.renderProperties();
    }
    ImGui::End();

    const ImVec2 renderPos = ImVec2(0, viewport->WorkPos.y);
    ImGui::SetNextWindowPos(renderPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - toolbarWidth,
                                    viewport->WorkPos.y + viewport->WorkSize.y - renderPos.y));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
    flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;

    update(); // Update state on GPU

    ImGui::Begin("Render", nullptr, flags);

    // The most important lines
    if (camera.integrator == 6) {
        owlLaunch2D(owl.gBufferRayGen, camera.resolution.x, camera.resolution.y, owl.launchParams);
        cudaDeviceSynchronize();
        owlLaunch2D(owl.reSTIRRayGen, camera.resolution.x, camera.resolution.y, owl.launchParams);
        cudaDeviceSynchronize();
    } else {
        owlLaunch2D(owl.lightingRayGen, camera.resolution.x, camera.resolution.y, owl.launchParams);
        cudaDeviceSynchronize();
    }

    ImVec2 availRegion = ImGui::GetContentRegionAvail();

    float cursorX = (availRegion.x - static_cast<float>(camera.resolution.x)) * 0.5f;
    if (cursorX > 0.f) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + cursorX);
    }

    float cursorY = (availRegion.y - static_cast<float>(camera.resolution.y)) * 0.5f;
    if (cursorY > 0.f) {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + cursorY);
    }

    // Bind gl buffers, textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gl.dispTex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.pbo);

    // Transfer from pbo to texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camera.resolution.x, camera.resolution.y, GL_RGBA, GL_FLOAT, nullptr);

    ImGui::Image((ImTextureID)(intptr_t)gl.dispTex, ImVec2(camera.resolution.x, camera.resolution.y), ImVec2(0, 1),
                 ImVec2(1, 0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    ImGui::End();
    ImGui::PopStyleColor();

    oldCamera = camera;
}
