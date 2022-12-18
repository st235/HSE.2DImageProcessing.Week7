cmake_minimum_required(VERSION 3.0.0)
project(FaceDetector VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB CODE_FILES "./src/*.cpp")

add_executable(FaceDetector main.cpp ${CODE_FILES})
target_link_libraries(FaceDetector ${OpenCV_LIBS})
