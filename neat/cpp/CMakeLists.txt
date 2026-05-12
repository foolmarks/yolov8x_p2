cmake_minimum_required(VERSION 3.16)
project(run_neat_sync LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(run_neat_sync main_sync.cpp)
target_link_libraries(run_neat_sync PRIVATE SimaNeat::sima_neat ${OpenCV_LIBS})

add_executable(run_neat_async main_async.cpp)
target_link_libraries(run_neat_async PRIVATE SimaNeat::sima_neat ${OpenCV_LIBS})

if(EXISTS "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu")
  set_target_properties(run_neat_sync PROPERTIES
    BUILD_RPATH "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu"
    INSTALL_RPATH "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu"
  )
  set_target_properties(run_neat_async PROPERTIES
    BUILD_RPATH "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu"
    INSTALL_RPATH "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu"
  )
endif()
