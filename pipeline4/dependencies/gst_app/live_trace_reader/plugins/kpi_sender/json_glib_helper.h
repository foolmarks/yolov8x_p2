#ifndef _JSON_GLIB_HELPER_H
#define _JSON_GLIB_HELPER_H

#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include <json-glib-1.0/json-glib/json-glib.h>

void json_add_string(JsonBuilder *builder, const char *name, const char *value);
void json_add_number(JsonBuilder *builder, const char *name, int64_t value);
void json_add_double(JsonBuilder *builder, const char *name, gdouble value);

void json_add_nested_kpi_number(JsonBuilder *builder, const char *name, int64_t value);
void json_add_nested_kpi_double(JsonBuilder *builder, const char *name, gdouble value);

#endif // _JSON_GLIB_HELPER_H