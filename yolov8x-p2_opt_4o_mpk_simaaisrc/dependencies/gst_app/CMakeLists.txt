cmake_minimum_required(VERSION 3.16)

# Set the project name
set(PROJECT_NAME "test-app")

project("${PROJECT_NAME}"
  VERSION 0.1
  DESCRIPTION "Test gst app"
  LANGUAGES C CXX)

# Execute process to get the git branch and commit hash
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Link directories
link_directories(${CMAKE_INSTALL_DIR}/core
  ${CMAKE_INSTALL_DIR}/gst
)

# Include GNUInstallDirs module
include(GNUInstallDirs)

# Find libraries
find_library(GLIB2_LIBRARY glib-2.0 PATHS ${GLIB2_LIBRARY_DIRS})
find_library(GOBJECT2_LIBRARY gobject-2.0 PATHS ${GLIB2_LIBRARY_DIRS})
find_library(GSTBASE_LIBRARY gstbase-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS})
find_library(GST_LIBRARY gstreamer-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS})
find_library(GST_APP gstapp-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS})
find_library(GST_VIDEO gstvideo-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS})
find_library(GST_RTP gstrtp-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS})
find_library(MOSQUITTO_LIBRARY mosquitto PATHS ${MOSQUITTO_LIBRARY_DIRS})
find_library(JSON_GLIB_LIBRARY json-glib-1.0 PATHS ${JSON_GLIB_LIBRARY_DIRS})

# Verify if the library is found
if(NOT MOSQUITTO_LIBRARY)
    message(FATAL_ERROR "Mosquitto library not found.")
endif()

# Find packages using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(MOSQUITTO REQUIRED libmosquitto)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)

# Common include directories
set(COMMON_INCLUDE_DIRS
  PRIVATE
  .
  ${CMAKE_SOURCE_DIR}
)

include_directories (
        PRIVATE
        .)

# Add subdirectories
add_subdirectory(utils)
add_subdirectory(live_trace_reader)

set(COMMON_PUBLIC_INCLUDE_DIRS
  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
  ${GLIB2_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${MOSQUITTO_INCLUDE_DIRS}
  ${JSON_GLIB_INCLUDE_DIRS}
)

# Add an executable target for the main application
set(GST_APP "gst_app")
add_executable(${GST_APP} 
  gst_app.cpp
  mqtt_client.cpp
  pipeline.cpp
  manifest_parser.cpp
)

# Include directories
target_include_directories(${GST_APP}
  PRIVATE ${COMMON_INCLUDE_DIRS}
  PUBLIC ${COMMON_PUBLIC_INCLUDE_DIRS}
  PRIVATE utils/include
)

# Link libraries to the executable
target_link_libraries(${GST_APP}
  PUBLIC
  ${GLIB2_LIBRARY}
  ${GOBJECT2_LIBRARY}
  ${GSTBASE_LIBRARY}
  ${GST_LIBRARY}
  ${MOSQUITTO_LIBRARY}
  ${JSON_GLIB_LIBRARY}
  simaailog
  mosquitto
  simaaiparser
  utils
  live_trace_reader
)

# Install the target
install(TARGETS ${GST_APP}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Add an executable target for the tests
set(TEST_APP_GST "test_app_gst")
add_executable(${TEST_APP_GST}
  tests/validate_gst_string.cpp
)

# Include directories for the test executable
target_include_directories(${TEST_APP_GST}
  PRIVATE ${COMMON_INCLUDE_DIRS}
  PUBLIC ${COMMON_PUBLIC_INCLUDE_DIRS}
  PRIVATE utils/include
)

# Link libraries to the test executable
target_link_libraries(${TEST_APP_GST}
  PUBLIC
  ${GLIB2_LIBRARY}
  ${GOBJECT2_LIBRARY}
  ${GSTBASE_LIBRARY}
  ${GST_LIBRARY}
  ${MOSQUITTO_LIBRARY}
  simaailog
  mosquitto
  simaaiparser
  utils
)

# Install the test executable
install(TARGETS ${TEST_APP_GST}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Add an executable target for the tests
set(TEST_APP_MANIFEST "test_app_manifest")
add_executable(${TEST_APP_MANIFEST}
  tests/validate_manifest_json.cpp
  manifest_parser.cpp
)

# Include directories for the test executable
target_include_directories(${TEST_APP_MANIFEST}
  PRIVATE ${COMMON_INCLUDE_DIRS}
  PUBLIC ${COMMON_PUBLIC_INCLUDE_DIRS}
  PRIVATE utils/include
)

# Link libraries to the test executable
target_link_libraries(${TEST_APP_MANIFEST}
  PUBLIC
  ${GLIB2_LIBRARY}
  ${GOBJECT2_LIBRARY}
  ${GSTBASE_LIBRARY}
  ${GST_LIBRARY}
  ${MOSQUITTO_LIBRARY}
  simaailog
  mosquitto
  simaaiparser
  utils
)

# Install the test executable
install(TARGETS ${TEST_APP_MANIFEST}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

set(TEST_PARAM_PARSER "test_param_parser")
add_executable(${TEST_PARAM_PARSER}
  tests/test_param_parser.cpp
)

# Include directories for the test executable
target_include_directories(${TEST_PARAM_PARSER}
  PRIVATE ${COMMON_INCLUDE_DIRS}
  PUBLIC ${COMMON_PUBLIC_INCLUDE_DIRS}
  PRIVATE utils/include
)

# Link libraries to the test executable
target_link_libraries(${TEST_PARAM_PARSER}
  PUBLIC
  ${GLIB2_LIBRARY}
  ${GOBJECT2_LIBRARY}
  ${GSTBASE_LIBRARY}
  ${GST_LIBRARY}
  ${MOSQUITTO_LIBRARY}
  simaailog
  mosquitto
  simaaiparser
  utils
)

# Install the test executable
install(TARGETS ${TEST_PARAM_PARSER}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
