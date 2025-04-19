#define GLFW_INCLUDE_NONE

#include "app/display.hpp"

#include <GLFW/glfw3.h>
#include <fontconfig/fontconfig.h>
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <optional>
#include <spdlog/spdlog.h>

std::optional<std::string> get_default_font() {
    FcConfig *config = FcInitLoadConfigAndFonts();
    FcPattern *pattern = FcPatternCreate();
    FcObjectSet *object_set = FcObjectSetBuild(FC_FILE, nullptr);
    FcFontSet *font_set = FcFontList(config, pattern, object_set);

    std::string font_path;
    if (font_set && font_set->nfont > 0) {
        FcChar8 *file = nullptr;
        if (FcPatternGetString(font_set->fonts[0], FC_FILE, 0, &file) == FcResultMatch) {
            font_path = reinterpret_cast<const char *>(file);
        } else {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }

    FcFontSetDestroy(font_set);
    FcObjectSetDestroy(object_set);
    FcPatternDestroy(pattern);
    FcConfigDestroy(config);

    return font_path;
}

Display::Display() : renderer(nullptr), running(false) {
    if (!glfwInit()) {
        spdlog::error("GLFW init failed");
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(1920, 1080, "steve", nullptr, nullptr);
    if (!window) {
        spdlog::error("GLFW window creation failed");
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        spdlog::error("GLAD init failed");
        return;
    }

    const GLubyte *glVersion = glGetString(GL_VERSION);
    const GLubyte *glRenderer = glGetString(GL_RENDERER);
    spdlog::info("GL_VERSION: {}", reinterpret_cast<const char *>(glVersion));
    spdlog::info("GL_RENDERER: {}", reinterpret_cast<const char *>(glRenderer));

    glEnable(GL_FRAMEBUFFER_SRGB);
    glfwSwapInterval(0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    // Setup font
    auto font_path_res = get_default_font();
    if (font_path_res.has_value()) {
        std::string font_path = font_path_res.value();
        spdlog::debug("Using font: {}", font_path);
        ImFontConfig font_config;
        io.Fonts->AddFontFromFileTTF(font_path_res.value().c_str(), 16.0f, &font_config);
    } else {
        spdlog::warn("Could not find a default font, using the ImGui default font.");
    }
}

Display::~Display() {
    running = false;
    if (renderThread.joinable()) {
        renderThread.join();
    }
}

void Display::attachRenderer(const std::shared_ptr<Render> &renderer) { this->renderer = renderer; }

void Display::startThread() { renderThread = std::thread(&Display::start, this); }

void Display::processInputs() {
    const float move_speed = 5.f;
    const float rotate_speed = 5.0f;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    } else if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        renderer->moveCamera(Render::MoveUp, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        renderer->moveCamera(Render::MoveDown, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        renderer->moveCamera(Render::MoveLeft, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        renderer->moveCamera(Render::MoveRight, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        renderer->moveCamera(Render::MoveForward, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        renderer->moveCamera(Render::MoveBackward, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        renderer->moveCamera(Render::RotateUp, rotate_speed);
    } else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        renderer->moveCamera(Render::RotateDown, rotate_speed);
    } else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        renderer->moveCamera(Render::RotateLeft, rotate_speed);
    } else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        renderer->moveCamera(Render::RotateRight, rotate_speed);
    } else if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
        renderer->moveCamera(Render::FocalDec, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        renderer->moveCamera(Render::FocalInc, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
        renderer->moveCamera(Render::AptInc, move_speed);
    } else if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        renderer->moveCamera(Render::AptDec, move_speed);
    }
}

void Display::start() {
    // Render loop
    running = true;
    using clock = std::chrono::steady_clock;
    constexpr auto FRAMETIME = std::chrono::duration<double, std::milli>(1000.0 / 60.0);

    auto nextFrame = clock::now() + FRAMETIME;
    while (running && !glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit")) {
                glfwSetWindowShouldClose(window, true);
            }
            ImGui::EndMenu();
        }

        auto fps = fmt::format("{:.1f} FPS", ImGui::GetIO().Framerate);
        float window_width = ImGui::GetWindowWidth();;
        float text_width = ImGui::CalcTextSize(fps.c_str()).x;
        ImGui::SameLine(window_width - 10 - text_width);
        ImGui::Text("%s", fps.c_str());
        ImGui::EndMainMenuBar();

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (renderer) {
            processInputs();
            renderer->render();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // std::this_thread::sleep_until(nextFrame);
        // nextFrame += FRAMETIME;
    }

    // Clean up
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
