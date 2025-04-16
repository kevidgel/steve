/**
 * @file Wrapper class for glsl shaders
 */

#pragma once

#include <glad/glad.h>
#include <string>

#include "owl/common/math/vec.h"

class Shader {
  public:
    GLuint program_id;

    Shader(const char *vertex_shader_source, const char *fragment_shader_source);
    ~Shader();

    void use();

    void set_bool(const std::string &name, bool value) const;
    void set_int(const std::string &name, int value) const;
    void set_float(const std::string &name, float value) const;
};