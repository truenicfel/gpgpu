project(freeglut)
find_package(OpenGL REQUIRED)
add_definitions(-DFREEGLUT_STATIC -D_CRT_SECURE_NO_WARNINGS -DFREEGLUT_LIB_PRAGMAS=0)
if(UNIX)
	add_definitions( -D__unix__ -DHAVE_FCNTL_H )
endif(UNIX)
include_directories(include)
set(SOURCES
	src/freeglut_callbacks.c
	src/freeglut_cursor.c
	src/freeglut_display.c
	src/freeglut_ext.c
	src/freeglut_font_data.c
	src/freeglut_font.c
	src/freeglut_gamemode.c
	src/freeglut_geometry.c
	src/freeglut_glutfont_definitions.c
	src/freeglut_init.c
	src/freeglut_input_devices.c
	src/freeglut_internal.h
	src/freeglut_joystick.c
	src/freeglut_main.c
	src/freeglut_menu.c
	src/freeglut_misc.c
	src/freeglut_overlay.c
	src/freeglut_spaceball.c
	src/freeglut_state.c
	src/freeglut_stroke_mono_roman.c
	src/freeglut_stroke_roman.c
	src/freeglut_structure.c
	src/freeglut_teapot_data.h
	src/freeglut_teapot.c
	src/freeglut_videoresize.c
	src/freeglut_window.c
	include/GL/freeglut.h
	include/GL/freeglut_ext.h
	include/GL/freeglut_std.h
	include/GL/glut.h
)

add_library(freeglut STATIC	${SOURCES} )
if(WIN32)
set(PLATFORM_LIBS winmm)
endif()
target_link_libraries(freeglut ${OPENGL_LIBRARIES} ${PLATFORM_LIBS})
SET_PROPERTY(TARGET freeglut         PROPERTY FOLDER "Libraries")

set(freeglut_INCLUDE_DIR "${freeglut_SOURCE_DIR}/include" CACHE PATH "" FORCE)
set(freeglut_LIBRARY_DBG freeglut CACHE STRING "" FORCE)
set(freeglut_LIBRARY_REL freeglut CACHE STRING "" FORCE)

mark_as_advanced(freeglut_INCLUDE_DIR freeglut_LIBRARY_DBG freeglut_LIBRARY_REL)
