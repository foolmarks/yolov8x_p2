# **************************************************************************
# ||                        SiMa.ai CONFIDENTIAL                          ||
# ||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
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

# set the project name
set(PROJECT_NAME "test_allocator")

project("${PROJECT_NAME}"
  VERSION 0.1
  DESCRIPTION "SiMa.AI GstMemory Allocator test"
  LANGUAGES C CXX)

set (TEST_ALLOCATOR_LIBRARY_SOURCES
  "test_allocator.cc")

add_executable(${PROJECT_NAME}
  ${TEST_ALLOCATOR_LIBRARY_SOURCES})

include(GNUInstallDirs)

target_include_directories( ${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
  ${GLIB2_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS})

find_library(GLIB2_LIBRARY glib-2.0 PATHS ${GLIB2_LIBRARY_DIRS} )
find_library(GOBJECT2_LIBRARY gobject-2.0 PATHS ${GLIB2_LIBRARY_DIRS} )
find_library(GSTBASE_LIBRARY gstbase-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS} )
find_library(GST_LIBRARY gstreamer-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS} )

target_link_libraries(${PROJECT_NAME}
  PUBLIC  ${GLIB2_LIBRARY} ${GOBJECT2_LIBRARY} ${GSTBASE_LIBRARY} ${GST_LIBRARY}
  simaaimem
  gstsimaallocator)

link_directories(
  ${CMAKE_SOURCE_DIR}/../build/core/allocator/)

target_include_directories ("${PROJECT_NAME}"
  PRIVATE
  .
  ${CMAKE_CURRENT_SOURCE_DIR}/../../metadata
  )

INSTALL(TARGETS "${PROJECT_NAME}")
