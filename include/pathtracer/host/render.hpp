/**
 * @file render.hpp
 * @brief Launches CUDA kernel per frame
 */

#pragma once

#include "owl/owl.h"
#include "pathtracer/host/camera.hpp"
#include "pathtracer/shared/integrator_defs.cuh"
#include "utils/shader.hpp"
#include "scene.hpp"

class Render {
  public:
    explicit Render(std::unique_ptr<SceneBuffer> scene, std::unique_ptr<Camera> camera,
                    const std::filesystem::path &envFilename);
    ~Render();

    enum CameraAction {
        MoveUp,
        MoveDown,
        MoveLeft,
        MoveRight,
        MoveForward,
        MoveBackward,
        RotateLeft,
        RotateRight,
        RotateUp,
        RotateDown,
        FocalInc,
        FocalDec,
        AptInc,
        AptDec,
    };

    void moveCamera(const CameraAction &action, float speed);
    void render();

  private:

    static void logInfo(const std::string &log);
    void update();

    struct {
        Shader *shader;
        GLuint vbo;
        GLuint vao;
        GLuint ebo;
        GLuint pbo;
        GLuint dispTex;
    } gl{};

    struct {
        OWLContext ctx;
        // ReSTIR Lighting shader
        OWLModule reSTIRModule;
        OWLRayGen reSTIRRayGen;
        OWLMissProg reSTIRMissProg;
        // Lighting shader
        OWLModule lightingModule;
        OWLRayGen lightingRayGen;
        OWLMissProg lightingMissProg;
        // GeometryPass Shader
        OWLModule gBufferModule;
        OWLRayGen gBufferRayGen;
        OWLMissProg gBufferMissProg;
        OWLBuffer gBuffer[2];
        // Launch Params
        OWLParams launchParams;
        OWLBuffer launchParamsBuffer;
        OWLBuffer reservoir[2];
        // Geometry
        OWLGeomType mesh;
        OWLGeom triMeshGeom;
        OWLGroup triMeshGroup;
        OWLGroup world;
        OWLBuffer vertsBuffer;
        OWLBuffer vertsIBuffer;
        OWLBuffer normsBuffer;
        OWLBuffer normsIBuffer;
        OWLBuffer texCoordsBuffer;
        OWLBuffer texCoordsIBuffer;
        OWLBuffer matsBuffer;
        OWLBuffer lightsVertsBuffer;
        OWLBuffer lightsVertsIBuffer;
        OWLBuffer lightsPrimsIBuffer;
        OWLBuffer matsIBuffer;
        OWLBuffer texturesBuffer;
        // Env map
        OWLTexture envMap;
    } owl{};

    Frame frame{};
    Camera camera{};
    Camera oldCamera{};
    int curReservoir = 0;
    MaterialBuffer materialBuffer{};
    bool accumFrames = false;

    cudaGraphicsResource *pbo = nullptr;
};