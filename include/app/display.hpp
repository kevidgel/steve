/**
 * @file display.hpp
 * @brief Class for handling viewport
 */

#pragma once

#include "pathtracer/host/render.hpp"

#include <GLFW/glfw3.h>
#include <thread>

class Display {
  public:
    explicit Display();
    ~Display();

    /// Attach renderer
    void attachRenderer(const std::shared_ptr<Render>& renderer);
    /// Start in current thread
    void start();
    /// Start in separate display thread
    void startThread();

  private:
    GLFWwindow *window = nullptr;
    std::shared_ptr<Render> renderer;
    std::thread renderThread;
    std::atomic<bool> running;

    void processInputs();
};
