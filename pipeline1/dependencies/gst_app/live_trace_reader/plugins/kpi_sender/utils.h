#ifndef _UTILS_H
#define _UTILS_H

#include <glib.h>
#include <babeltrace2/babeltrace.h>

#include "kpi.h"

#define TIMESPEC_TO_MS(timespec_struct) ((timespec_struct.tv_sec * 1000) + (timespec_struct.tv_nsec / 1000000))

int get_payload_field_by_name(const bt_field *payload, const char *name, const bt_field **result);
int parse_uint64_field(const bt_field *field, uint64_t *result);
int parse_field(const bt_field *field, GString *result);

uint32_t hash_string_to_uint32(const char *str);
bt_bool is_remote_core_kpi(const plugin_kpi_t * const kpi);

#endif // _UTILS_H