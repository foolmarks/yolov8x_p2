cmake_minimum_required(VERSION 3.10)
project(kpi_sender C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fPIC")

find_library(MOSQUITTO_LIBRARY mosquitto PATHS ${MOSQUITTO_LIBRARY_DIRS})
find_library(SIMAAILOG_LIB simaailog)

# Verify if the library is found
if(NOT MOSQUITTO_LIBRARY)
    message(FATAL_ERROR "Mosquitto library not found.")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(BABELTRACE2 REQUIRED IMPORTED_TARGET babeltrace2)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(JSON_GLIB REQUIRED IMPORTED_TARGET json-glib-1.0)
pkg_check_modules(MOSQUITTO REQUIRED libmosquitto)

add_library(kpi_sender
	SHARED
	kpi_sender.c
	kpi.c
	trace.c
	json_glib_helper.c
	mqtt.c
	utils.c
)
target_include_directories(kpi_sender
	PRIVATE
	${BABELTRACE2_INCLUDE_DIRS}
	${GLIB_INCLUDE_DIRS}
	PUBLIC
    ${JSON_GLIB_INCLUDE_DIRS}
	${MOSQUITTO_INCLUDE_DIRS}
)

target_compile_options(kpi_sender
	PRIVATE
    ${JSON_GLIB_CFLAGS_OTHER}
)

target_link_libraries(kpi_sender
	PRIVATE
	PkgConfig::BABELTRACE2
    PkgConfig::JSON_GLIB
	${GLIB_LDFLAGS}
	${MOSQUITTO_LIBRARY}
	${SIMAAILOG_LIB}
)

install(TARGETS kpi_sender LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/babeltrace2/plugins)
