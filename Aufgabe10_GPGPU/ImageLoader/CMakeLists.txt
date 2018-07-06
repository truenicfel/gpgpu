FILE(GLOB Sources "*.cpp")
FILE(GLOB Headers "*.h")

add_library(ImageLoader ${Sources} ${Headers} )
target_link_libraries(ImageLoader FreeImage)
SET_PROPERTY(TARGET ImageLoader PROPERTY FOLDER "Libraries")