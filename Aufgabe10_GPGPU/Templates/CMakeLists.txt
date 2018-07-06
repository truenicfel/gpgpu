### CMakeLists.txt Template für neue Projekte

#TODO Projektnamen ändern
SET(PROJECT_NAME MeinNeuesProjekt)
#TODO Quelldateien hinzufügen, auch glsl Shader. 
#mit Leerzeichen getrennt
set(SOURCES Test.cpp )
#TODO Bibliotheken anpassen
set(LIBRARIES GLTools ObjLoader AntTweakBar freeglut)

project(${PROJECT_NAME} )

add_executable(${PROJECT_NAME} ${SOURCES})
create_target_launcher(${PROJECT_NAME}  WORKING_DIRECTORY ${${PROJECT_NAME}_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
SET_PROPERTY(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Aufgaben")
install_sample(${PROJECT_NAME} ${SOURCES})