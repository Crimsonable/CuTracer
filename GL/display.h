#pragma once
#include "shader.h"

GLFWwindow *InitContext(unsigned int width, unsigned int height);

class Display {
public:
  GLFWwindow *window = nullptr;
  const float vertices[20] = {-1.0f, -1.0f, 0.0f,  0.0f, 0.0f, -1.0f, 1.0f,
                              0.0f,  0.0f,  1.0f,  1.0f, 1.0f, 0.0f,  1.0f,
                              1.0f,  1.0f,  -1.0f, 0.0f, 1.0f, 0.0f};
  const unsigned int indices[6] = {0, 3, 1, 1, 3, 2};
  unsigned int surfaceVAO, surfaceEBO, surfaceVBO;
  unsigned int height, width, channel;

  Shader shader;

public:
  Display(int width, int height, int channel);

  void setShaders(const char *vs, const char *fs);
};