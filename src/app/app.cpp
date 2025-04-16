/**
 * @file app.cpp
 * @brief Main application for viewing renders
 */

#include "app/display.hpp"
#include "pathtracer/parser.hpp"
#include "pathtracer/render.hpp"

#include <spdlog/spdlog.h>

int main(int argc, char* argv[]) {
    Context ctx;
    ctx.width = 1920;
    ctx.height = 1080;

    try {
        spdlog::info("Parsing scene...");
        Result result = parseFile("/home/kevidgel/cmu/15468/final-project/models/test.json");

        spdlog::info("Setting up viewport...");
        Display display(ctx);

        spdlog::info("Setting up pathtracer...");
        auto pathtracer = std::make_shared<Render>(ctx, std::move(result.scene), std::move(result.camera), "");

        display.attachRenderer(pathtracer);
        display.start();
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
    }
}
