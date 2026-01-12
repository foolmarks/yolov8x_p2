#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2025 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************

cmake_minimum_required(VERSION 3.16)

project("gstsimaaicaps"
    VERSION 0.1
    DESCRIPTION "SiMa.ai GstCaps library"
    HOMEPAGE_URL "https://bitbucket.org/sima-ai/gst-simaai-plugins-base/"
    LANGUAGES C
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_library(${PROJECT_NAME}
    SHARED
    "gstsimaaicaps.c"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER
    "gstsimaaicaps.h"
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED IMPORTED_TARGET gstreamer-base-1.0)
pkg_check_modules(GLIB REQUIRED IMPORTED_TARGET glib-2.0)
pkg_check_modules(JSON_GLIB REQUIRED IMPORTED_TARGET json-glib-1.0)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC
    ${JSON_GLIB_INCLUDE_DIRS}
)

target_compile_options(${PROJECT_NAME}
    PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GLIB_CFLAGS_OTHER}
    ${JSON_GLIB_CFLAGS_OTHER}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    PkgConfig::GSTREAMER
    PkgConfig::GSTREAMER_BASE
    PkgConfig::GLIB
    PkgConfig::JSON_GLIB
)

include(GNUInstallDirs)

install(TARGETS "${PROJECT_NAME}"
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simaai
)
