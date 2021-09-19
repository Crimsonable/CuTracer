#include "GL/display.h"
#include "core/canvas.h"
#include "core/gl_cuda_interop.h"
#include "core/model.h"
#include <chrono>

#define WIDTH 768
#define HEIGHT 512

using namespace Tracer;

int main() {
   Display dis(WIDTH, HEIGHT, 4);
   Tracer::Camera camera;
   Scene scene;
   Tracer::Canvas canvas(Expblas::Shape<2>({WIDTH, HEIGHT}), &camera, &scene);
   scene.AddModel("H:/toys/cuda_tracer/Tracer_cuda/model/bunny/reconstruction/"
                 "bun_zipper_res3.ply");
   canvas.init();
   dis.setShaders("H:/toys/cuda_tracer/Tracer_cuda/GL/shaders/dis.vs",
                 "H:/toys/cuda_tracer/Tracer_cuda/GL/shaders/dis.fs");

  // auto model =
  Tracer::Model("./model/bunny/reconstruction/bun_zipper.ply");
   int count = 100;
  // while (!glfwWindowShouldClose(dis.window)) {
   while (count--) {
    auto t1 = std::chrono::steady_clock::now();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    canvas.cudaDraw();
    canvas.submit();
    glBindTexture(GL_TEXTURE_2D, canvas.texture);
    SDK_CHECK_ERROR_GL();
    glBindVertexArray(dis.surfaceVAO);
    SDK_CHECK_ERROR_GL();
    dis.shader.use();
    SDK_CHECK_ERROR_GL();
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    SDK_CHECK_ERROR_GL();

    glfwSwapBuffers(dis.window);
    glfwPollEvents();
    auto t2 = std::chrono::steady_clock::now();
    /*std::cout << 1.0f /
                     std::chrono::duration_cast<std::chrono::duration<double>>(
                         t2 - t1)
                         .count()
              << "fps" << std::endl;*/
    // system("cls");
  }
   glDeleteVertexArrays(1, &dis.surfaceVAO);
   glDeleteBuffers(1, &dis.surfaceVBO);
   glDeleteBuffers(1, &dis.surfaceEBO);

   glfwTerminate();
  return 0;
}