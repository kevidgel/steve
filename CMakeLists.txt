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

set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
add_subdirectory(ext/owl EXCLUDE_FROM_ALL)
add_subdirectory(ext/rapidobj)
add_subdirectory(ext/tinygltf)

# Trace PTX compilation
enable_language(CUDA)
set(trace_srcs
        src/pathtracer/device/spatial.cu
        src/pathtracer/device/temporal.cu
        src/pathtracer/device/integrator.cu
        src/pathtracer/device/gbuffer.cu
)
set(trace_headers
        include/pathtracer/device/integrator.cuh
        include/pathtracer/device/geometry.cuh
        include/pathtracer/device/reservoir.cuh
        include/pathtracer/device/restir.cuh
        include/pathtracer/device/ray.cuh
        include/pathtracer/device/material.cuh
        include/pathtracer/device/sampling.cuh
        include/pathtracer/device/lights.cuh
        include/pathtracer/shared/geometry_defs.cuh
        include/pathtracer/shared/material_defs.cuh
        include/pathtracer/shared/integrator_defs.cuh
        include/pathtracer/shared/reservoir_defs.cuh
        include/pathtracer/shared/light_defs.cuh
        include/pathtracer/shared/ray_defs.cuh
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
set(PTX_HEADERS)
foreach(src IN LISTS trace_srcs)
    get_filename_component(name ${src} NAME_WE)
    string(REPLACE ".cu" ".ptx" ptx_rel ${src})
    set(ptx_source "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/trace.dir/${ptx_rel}")
    set(ptx_header "${PROJECT_BINARY_DIR}/generated/${name}.ptx.hpp")
    add_custom_command(
            OUTPUT   ${ptx_header}
            COMMAND  ${PROJECT_BINARY_DIR}/bin/embed ${ptx_source} ${ptx_header}
            DEPENDS embed trace ${trace_srcs} ${trace_headers}
            COMMENT  "Embedding PTX ${name}"
    )
    list(APPEND PTX_HEADERS ${ptx_header})
endforeach()

# Add custom target
add_custom_target(embed_trace
        DEPENDS
        ${PTX_HEADERS}
)

add_library(embed_shaders
        INTERFACE
        ${PTX_HEADERS}
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
        include/pathtracer/shared/reservoir_defs.cuh
        include/utils/shader.hpp
        ext/ImGuiFileDialog/ImGuiFileDialog.h
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
        ext/ImGuiFileDialog/ImGuiFileDialog.cpp
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
        ${CMAKE_SOURCE_DIR}/ext/tinygltf/
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${OptiX_ROOT_DIR}/include
        ${Stb_INCLUDE_DIR}
        ext/ImGuiFileDialog/
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
        tinygltf
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
