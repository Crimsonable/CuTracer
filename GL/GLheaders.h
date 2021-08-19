#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static inline const char *glErrorToString(GLenum err) {
#define CASE_RETURN_MACRO(arg)                                                 \
  case arg:                                                                    \
    return #arg
  switch (err) {
    CASE_RETURN_MACRO(GL_NO_ERROR);
    CASE_RETURN_MACRO(GL_INVALID_ENUM);
    CASE_RETURN_MACRO(GL_INVALID_VALUE);
    CASE_RETURN_MACRO(GL_INVALID_OPERATION);
    CASE_RETURN_MACRO(GL_OUT_OF_MEMORY);
    CASE_RETURN_MACRO(GL_STACK_UNDERFLOW);
    CASE_RETURN_MACRO(GL_STACK_OVERFLOW);
#ifdef GL_INVALID_FRAMEBUFFER_OPERATION
    CASE_RETURN_MACRO(GL_INVALID_FRAMEBUFFER_OPERATION);
#endif
  default:
    break;
  }
#undef CASE_RETURN_MACRO
  return "*UNKNOWN*";
}

inline bool sdkCheckErrorGL(const char *file, const int line) {
  bool ret_val = true;

  // check for error
  GLenum gl_error = glGetError();

  if (gl_error != GL_NO_ERROR) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    char tmpStr[512];
    // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at
    // the right line when the user double clicks on the error line in the
    // Output pane. Like any compile error.
    sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line,
              glErrorToString(gl_error));
    fprintf(stderr, "%s", tmpStr);
#endif
    fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
    fprintf(stderr, "%s\n", glErrorToString(gl_error));
    ret_val = false;
  }

  return ret_val;
}

#define SDK_CHECK_ERROR_GL()                                                   \
  if (false == sdkCheckErrorGL(__FILE__, __LINE__)) {                          \
    exit(EXIT_FAILURE);                                                        \
  }