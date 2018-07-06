set(CMAKE_PREFIX_PATH ${Computergrafik_SOURCE_DIR}/Dependencies ${CMAKE_PREFIX_PATH})
set(CMAKE_FRAMEWORK_PATH ${Computergrafik_SOURCE_DIR}/Dependencies ${CMAKE_FRAMEWORK_PATH})
find_package(OpenGL)
if(UNIX)
find_package(X11 REQUIRED)
if(NOT X11_Xinput_FOUND)
message(FATAL_ERROR "Xinput Development Files not found. Install xorg-dev")
endif()
endif()

add_subdirectory(Dependencies)




# Add library and include paths from the dependencies
include_directories(
  ${FreeImage_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  ${freeglut_INCLUDE_DIR}
  ${GLTools_INCLUDE_DIR}
   ${AntTweakBar_INCLUDE_DIR}
   ${GLEW_INCLUDE_DIR}
   ${glm_INCLUDE_DIR}
  )

link_directories(
  ${OPENGL_LIBRARY_DIR}
  ${FreeImage_LIBRARY_DIR}
  ${freeglut_LIBRARY_DIR}
  ${GLEW_LIBRARY_DIR}
)

