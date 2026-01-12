cmake_minimum_required(VERSION 3.16)

# set the project name
set(PROJECT_NAME "gstsimacore-lite")

project("${PROJECT_NAME}"
  VERSION 0.1
  DESCRIPTION "Sima Core libs"
  LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set (CORE_LIBRARY_SOURCES
  dispatcher.c
  ev_dispatcher.c
  mla_dispatcher.c
  mla_client.c
  host_dispatcher.c
  utils.c
  simamm.c)

find_package(PkgConfig)
pkg_check_modules(GLIB2 glib-2.0)
pkg_check_modules(GSTREAMER gstreamer-1.0)

if(NOT GLIB2_FOUND OR NOT GSTREAMER_FOUND )
    message(WARNING "GstRemoteOffload project is not configured due to absence of gstreamer component(s)" )
    message(WARNING "Please install GStreamer 1.14.0+ before running the sample." )
    return()
endif()

set(GLIBS ${GLIBS} glib-2.0 gobject-2.0 gio-2.0 gstcontroller-1.0 gstbase-1.0 gstreamer-1.0 gstvideo-1.0)

add_library(${PROJECT_NAME}
  SHARED
  ${CORE_LIBRARY_SOURCES})

include(GNUInstallDirs)

target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
  ${GLIB2_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS})

set (DISPATCHER_LITE_PUBLIC_HEADERS
  "dispatcher_common.h"
  "dispatcher.h"
  "ev_dispatcher.h"
  "mla_dispatcher.h"
  "mla_client.h"
  "host_dispatcher.h"
  "utils.h"
  "handlers.h")

set_target_properties(${PROJECT_NAME} PROPERTIES
  PUBLIC_HEADER
  "${DISPATCHER_LITE_PUBLIC_HEADERS}")

find_library(GLIB2_LIBRARY glib-2.0 PATHS ${GLIB2_LIBRARY_DIRS} )
find_library(GOBJECT2_LIBRARY gobject-2.0 PATHS ${GLIB2_LIBRARY_DIRS} )
find_library(GSTBASE_LIBRARY gstbase-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS} )
find_library(GST_LIBRARY gstreamer-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS} )

target_include_directories( ${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
  ${GLIB2_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
  PUBLIC  ${GLIB2_LIBRARY} ${GOBJECT2_LIBRARY} ${GSTBASE_LIBRARY} ${GST_LIBRARY}
  sgptransport
  sgpdispatch
  evdispatch
  simaaimem
  simaaiparser
  gstsimaallocator
  MLArt)

target_include_directories ("${PROJECT_NAME}"
  PRIVATE
  include)

INSTALL(TARGETS "${PROJECT_NAME}")
INSTALL(TARGETS "${PROJECT_NAME}" PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simaai/dispatcher-lite)
