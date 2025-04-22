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
        OWLModule module; // Eventually we want more modules
        OWLRayGen rayGen;
        OWLMissProg missProg;
        OWLParams launchParams;
        OWLGeomType mesh;
        OWLBuffer launchParamsBuffer;
        OWLGeom triMeshGeom;
        OWLGroup triMeshGroup;
        OWLGroup world;
        OWLBuffer cameraBuffer;
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
        OWLTexture envMap;
    } owl{};

    Frame frame{};
    Camera camera{};
    Camera oldCamera{};
    MaterialBuffer materialBuffer{};
    bool accumFrames = false;

    cudaGraphicsResource *pbo = nullptr;
};