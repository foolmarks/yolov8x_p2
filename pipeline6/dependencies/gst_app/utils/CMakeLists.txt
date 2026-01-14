cmake_minimum_required(VERSION 3.16)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED glib-2.0)
pkg_check_modules(LTTNG REQUIRED lttng-ctl)

add_library(utils
    src/string_utils.cpp
    src/cmdline_utils.cpp
    src/lttng_session.cpp
)

target_include_directories(utils
    PUBLIC
    ${GLIB2_INCLUDE_DIRS}
    ${LTTNG_INCLUDE_DIRS}
    include
)

target_link_libraries(utils
    PUBLIC
    ${GLIB2_LIBRARY}
    ${LTTNG_LIBRARIES}
)