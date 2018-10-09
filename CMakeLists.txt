cmake_minimum_required (VERSION 3.9.2)
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(PROG_NAME FRACTAL_FLAMES)
project(${PROG_NAME})


if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(INCLUDE_DIRECTORY include)
set(SOURCE_DIRECTORY src)

find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories(${INCLUDE_DIRECTORY}  ${OPENCL_INCLUDE_DIR}  ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11" )
if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

file(GLOB CPP_SOURCE src/*.cpp)
add_executable (FractalFlame ${CPP_SOURCE})
target_link_libraries(FractalFlame ${OpenCL_LIBRARY}  ${OPENGL_LIBRARIES}  ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})

