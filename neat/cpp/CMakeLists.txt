cmake_minimum_required(VERSION 3.16)
project(yolov8x_p2_neat_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(yolov8x_p2_neat_cpp main.cpp)
target_link_libraries(yolov8x_p2_neat_cpp PRIVATE SimaNeat::sima_neat ${OpenCV_LIBS})
