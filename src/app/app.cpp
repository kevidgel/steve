/**
 * @file app.cpp
 * @brief Main application for viewing renders
 */

#include "pathtracer/host/parser.hpp"
#include "pathtracer/host/render.hpp"
#include "app/display.hpp"

#include <spdlog/spdlog.h>

int main(int argc, char* argv[]) {
    try {
        spdlog::info("Parsing scene...");
        Result result = parseFile("/home/kevidgel/cmu/15468/final-project/models/test.json");

        spdlog::info("Setting up viewport...");
        Display display;

        spdlog::info("Setting up pathtracer...");
        auto pathtracer = std::make_shared<Render>(std::move(result.scene), std::move(result.camera), "");

        display.attachRenderer(pathtracer);
        display.start();
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
    }
}
