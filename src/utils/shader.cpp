#include "utils/shader.hpp"

#include <fstream>
#include <optional>
#include <spdlog/spdlog.h>
#include <vector>

std::optional<GLuint> load_spirv_shader(GLenum shader_type, const char* file_path, const char* entry_point = "main") {
    std::ifstream shader_file(file_path, std::ios::ate | std::ios::binary);
    if (!shader_file.is_open()) {
        return std::nullopt;
    }

    size_t file_size = (size_t) shader_file.tellg();
    std::vector<char> shader_file_contents(file_size);
    shader_file.seekg(0);
    shader_file.read(shader_file_contents.data(), file_size);
    shader_file.close();

    // Create shader
    GLuint shader = glCreateShader(shader_type);
    glShaderBinary(
        1,
        &shader,
        GL_SHADER_BINARY_FORMAT_SPIR_V,
        shader_file_contents.data(),
        file_size
        );
    glSpecializeShader(
        shader,
        entry_point,
        0,
        nullptr,
        nullptr
        );
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        GLchar info_log[512];
        glGetShaderInfoLog(shader, 512, NULL, info_log);
        spdlog::error("Failed to compile shader {}", file_path);
        glDeleteShader(shader);
        return std::nullopt;
    }

    return std::make_optional(shader);
}

std::optional<GLuint> load_spirv_shader_from_str(GLenum shader_type, const char payload[], const size_t payload_len, const char* entry_point = "main") {
    // Create shader
    GLuint shader = glCreateShader(shader_type);
    glShaderBinary(
        1,
        &shader,
        GL_SHADER_BINARY_FORMAT_SPIR_V,
        payload,
        payload_len
        );
    glSpecializeShader(
        shader,
        entry_point,
        0,
        nullptr,
        nullptr
        );
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        GLchar info_log[512];
        glGetShaderInfoLog(shader, 512, NULL, info_log);
        spdlog::error("Failed to compile shader from string");
        glDeleteShader(shader);
        return std::nullopt;
    }

    return std::make_optional(shader);
}

Shader::Shader(const char* vertex_shader_source, const char* fragment_shader_source) {
    GLuint vertex_shader, fragment_shader;
    int success;
    char info_log[512];

    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        glGetShaderInfoLog(vertex_shader, 512, NULL, info_log);
        spdlog::error("Failed to compile vertex shader: {}", info_log);
        glDeleteShader(vertex_shader);
        throw std::runtime_error("Failed to compile vertex shader");
    }

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        glGetShaderInfoLog(fragment_shader, 512, NULL, info_log);
        spdlog::error("Failed to compile fragment shader: {}", info_log);
        glDeleteShader(fragment_shader);
        throw std::runtime_error("Failed to compile fragment shader");
    }

    program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader);
    glAttachShader(program_id, fragment_shader);
    glLinkProgram(program_id);

    glGetProgramiv(program_id, GL_LINK_STATUS, &success);
    if (success == GL_FALSE) {
        glGetProgramInfoLog(program_id, 512, NULL, info_log);

        glDeleteProgram(program_id);
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        throw std::runtime_error("Failed to link shader program");
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
}

Shader::~Shader() {
    spdlog::info("Deleting shader program...");
    glDeleteProgram(program_id);
}

void Shader::use() {
    glUseProgram(program_id);
}

void Shader::set_bool(const std::string& name, bool value) const {
    glUniform1i(glGetUniformLocation(program_id, name.c_str()), (int)value);
}

void Shader::set_int(const std::string& name, int value) const {
    glUniform1i(glGetUniformLocation(program_id, name.c_str()), value);
}

void Shader::set_float(const std::string& name, float value) const {
    glUniform1f(glGetUniformLocation(program_id, name.c_str()), value);
}