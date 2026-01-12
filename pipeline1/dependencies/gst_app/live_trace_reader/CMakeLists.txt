cmake_minimum_required(VERSION 3.10)
project(live_trace_reader C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g")

set(API_DIR "api")

find_library(SIMAAILOG_LIB simaailog)

find_package(PkgConfig REQUIRED)
pkg_check_modules(BABELTRACE2 REQUIRED IMPORTED_TARGET babeltrace2)
pkg_check_modules(LTTNG REQUIRED lttng-ctl)
pkg_check_modules(GLIB REQUIRED glib-2.0)

add_library(live_trace_reader STATIC
    ${API_DIR}/live_trace_reader_api.c
)

target_include_directories(live_trace_reader
	PUBLIC
	${API_DIR}
	PRIVATE
	${BABELTRACE2_INCLUDE_DIRS}
	${GLIB_INCLUDE_DIRS}
	${LTTNG_INCLUDE_DIRS}
)
target_link_libraries(live_trace_reader PRIVATE
	PkgConfig::BABELTRACE2
	${SIMAAILOG_LIB}
	${GLIB_LDFLAGS}
	${LTTNG_LIBRARIES}
)

add_subdirectory(plugins)
