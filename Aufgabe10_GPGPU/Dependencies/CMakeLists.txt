cmake_minimum_required(VERSION 2.6)
project(Dependencies)
set(CMAKE_MODULE_PATH "${GPGPU_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
add_subdirectory(freeglut)
add_subdirectory(glew)
add_subdirectory(AntTweakBar)
add_subdirectory(FreeImage)
add_subdirectory(GLTools)

