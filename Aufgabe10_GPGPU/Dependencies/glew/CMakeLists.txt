project(glew)
include_directories(include)
set(SOURCES
	src/glew.c
)
add_definitions(-DGLEW_STATIC)
add_library(glew STATIC ${SOURCES})
SET_PROPERTY(TARGET glew         PROPERTY FOLDER "Libraries")

set(GLEW_INCLUDE_DIR "${glew_SOURCE_DIR}/include" CACHE PATH "" FORCE)
set(GLEW_LIBRARY_DBG glew CACHE STRING "" FORCE)
set(GLEW_LIBRARY_REL glew CACHE STRING "" FORCE)
mark_as_advanced(GLEW_INCLUDE_DIR GLEW_LIBRARY_DBG GLEW_LIBRARY_REL)
