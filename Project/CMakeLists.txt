cmake_minimum_required(VERSION 3.0.0)
project(FaceDetector VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/misc/haarcascade_frontalface_alt.xml DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/misc/haarcascade_frontalface_alt2.xml DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/misc/haarcascade_frontalface_default.xml DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/misc/haarcascade_lefteye_2splits.xml DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/misc/haarcascade_righteye_2splits.xml DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/misc/shape_predictor_68_face_landmarks.dat DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/misc/dlib_face_recognition_resnet_model_v1.dat DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

include_directories(include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB CODE_FILES "./src/*.cpp")

include(FetchContent)
FetchContent_Declare(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.24
        )
FetchContent_MakeAvailable(dlib)

add_executable(FaceDetector main.cpp ${CODE_FILES})
target_link_libraries(FaceDetector ${OpenCV_LIBS} dlib::dlib)
