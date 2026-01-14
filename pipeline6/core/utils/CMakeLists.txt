# **************************************************************************
# ||                        SiMa.ai CONFIDENTIAL                          ||
# ||   Unpublished Copyright (c) 2022-2024 SiMa.ai, All Rights Reserved.  ||
# **************************************************************************
#  NOTICE:  All information contained herein is, and remains the property of
#  SiMa.ai. The intellectual and technical concepts contained herein are
#  proprietary to SiMa and may be covered by U.S. and Foreign Patents,
#  patents in process, and are protected by trade secret or copyright law.
#
#  Dissemination of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained from
#  SiMa.ai.  Access to the source code contained herein is hereby forbidden
#  to anyone except current SiMa.ai employees, managers or contractors who
#  have executed Confidentiality and Non-disclosure agreements explicitly
#  covering such access.
#
#  The copyright notice above does not evidence any actual or intended
#  publication or disclosure  of  this source code, which includes information
#  that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
#  ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
#  DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
#  CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
#  LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
#  CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
#  REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
#  SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# **************************************************************************

cmake_minimum_required(VERSION 3.16)

project("commonutils"
  VERSION 0.1
  DESCRIPTION "SiMa.AI Utils library Library"
  HOMEPAGE_URL "https://bitbucket.org/sima-ai/gst-simaai-plugins-base"
  LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_library(${PROJECT_NAME}
  SHARED
  "utils_string.c")

set_target_properties(${PROJECT_NAME} PROPERTIES
  PUBLIC_HEADER
  "utils_string.h")

include(GNUInstallDirs)

# Install library and headers
INSTALL(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/utils)
