cmake_minimum_required(VERSION 3.25)
project(steve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Packages
find_package(fmt CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Fontconfig REQUIRED)
find_package(Freetype CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)

add_subdirectory(ext/owl EXCLUDE_FROM_ALL)
add_subdirectory(ext/rapidobj)

# Trace PTX compilation
enable_language(CUDA)
set(trace_srcs
        src/pathtracer/device/integrator.cu
)
set(trace_headers
        include/pathtracer/device/integrator.cuh
        include/pathtracer/device/geometry.cuh
        include/pathtracer/device/ray.cuh
        include/pathtracer/device/material.cuh
        include/pathtracer/device/sampling.cuh
        include/pathtracer/shared/geometry_defs.cuh
        include/pathtracer/shared/material_defs.cuh
        include/pathtracer/shared/integrator_defs.cuh
)
add_library(trace
        OBJECT
        ${trace_srcs}
        ${trace_headers}
)
target_link_libraries(trace
        owl::owl
)
target_include_directories(trace
        PRIVATE
        include/pathtracer/device
        include/pathtracer/shared
)
set_target_properties(trace
        PROPERTIES
        CUDA_PTX_COMPILATION ON
)
target_compile_options(trace
        PRIVATE
        $<$<CONFIG:DEBUG>:-G>
        $<$<CONFIG:RELEASE>:--use_fast_math>
)

# Create directory for generated ptx
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/generated/")

# NOTE: This is super hacky
set(PTX_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/trace.dir/src/pathtracer/device/integrator.ptx")
get_filename_component(PTX_NAME ${PTX_SOURCE} NAME_WE)
set(PTX_OUTPUT_DIR "${PROJECT_BINARY_DIR}/generated")
set(PTX_OUTPUT "${PTX_OUTPUT_DIR}/${PTX_NAME}.ptx.hpp")
add_custom_command(
        OUTPUT ${PTX_OUTPUT}
        COMMAND ${PROJECT_BINARY_DIR}/bin/embed ${PTX_SOURCE} ${PTX_OUTPUT}
        DEPENDS embed trace ${trace_srcs} ${trace_headers}
        COMMENT "Embedding PTX ${PTX_NAME}"
)
message(STATUS "PTX output: ${PTX_OUTPUT}")

# Add custom target
add_custom_target(embed_trace
        DEPENDS
        ${PTX_OUTPUT}
)

add_library(embed_shaders
        INTERFACE
        ${PTX_OUTPUT}
)

add_dependencies(embed_shaders embed_trace)
target_include_directories(embed_shaders
        INTERFACE
        ${CMAKE_BINARY_DIR}/generated
)

set(steve_headers
        include/app/display.hpp
        include/pathtracer/host/render.hpp
        include/pathtracer/host/scene.hpp
        include/pathtracer/host/texture.hpp
        include/pathtracer/host/camera.hpp
        include/pathtracer/host/parser.hpp
        include/pathtracer/host/material_buffer.hpp
        include/pathtracer/shared/integrator_defs.cuh
        include/pathtracer/shared/material_defs.cuh
        include/pathtracer/shared/geometry_defs.cuh
        include/utils/shader.hpp
)

set(steve_srcs
        src/app/app.cpp
        src/app/display.cpp
        src/pathtracer/host/render.cpp
        src/pathtracer/host/scene.cpp
        src/pathtracer/host/texture.cpp
        src/pathtracer/host/parser.cpp
        src/pathtracer/host/material_buffer.cpp
        src/utils/shader.cpp
)

# Compile library
add_library(steve_lib STATIC
        ${steve_headers}
        ${steve_srcs}
)

target_include_directories(steve_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/ext/owl/owl/include
        ${CMAKE_SOURCE_DIR}/ext/rapidobj/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${OptiX_ROOT_DIR}/include
        ${Stb_INCLUDE_DIR}
)

target_link_libraries(steve_lib PRIVATE
        embed_shaders
        fmt::fmt
        spdlog::spdlog
        glfw
        glad::glad
        imgui::imgui
        Freetype::Freetype
        Fontconfig::Fontconfig
        glm::glm
        owl::owl
        rapidobj::rapidobj
        nlohmann_json::nlohmann_json
        imguizmo::imguizmo
)

# Compile main app
add_executable(steve src/app/app.cpp)
target_link_libraries(steve steve_lib)

# Executable for embedding
add_executable(embed
        src/utils/embed.cpp
)
set_target_properties(embed
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
