project(AntTweakBar)


link_directories(
  ${OPENGL_LIBRARY_DIRS}
)
include_directories(include)

set(SOURCES src/TwPrecomp.cpp
			src/TwPrecomp.h
			src/TwMgr.cpp
			src/TwMgr.h
			src/TwFonts.cpp
			src/TwFonts.h
			src/TwColors.cpp
			src/TwColors.h
			src/TwBar.cpp
			src/TwBar.h
			src/TwGraph.h
			src/resource.h
	)


list(APPEND SOURCES src/TwOpenGL.h src/TwOpenGL.cpp src/LoadOGL.cpp src/TwOpenGLCore.h src/TwOpenGLCore.cpp src/LoadOGLCore.cpp)

list(APPEND SOURCES src/MiniGLUT.h src/TwEventGLUT.c) 


if(UNIX)
add_definitions(-D_UNIX)
endif()
add_definitions(/DTW_STATIC /DTW_NO_LIB_PRAGMA)
add_library(AntTweakBar STATIC ${SOURCES})
SET_PROPERTY(TARGET AntTweakBar PROPERTY FOLDER "Libraries")
target_link_libraries(AntTweakBar ${OPENGL_LIBRARIES} freeglut)

set(AntTweakBar_INCLUDE_DIR "${AntTweakBar_SOURCE_DIR}/include" CACHE PATH "" FORCE)
set(AntTweakBar_LIBRARY_DBG AntTweakBar CACHE STRING "" FORCE)
set(AntTweakBar_LIBRARY_REL AntTweakBar CACHE STRING "" FORCE)
mark_as_advanced(AntTweakBar_INCLUDE_DIR AntTweakBar_LIBRARY_DBG AntTweakBar_LIBRARY_REL)
