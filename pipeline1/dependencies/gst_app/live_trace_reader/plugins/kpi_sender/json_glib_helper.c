#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include <json-glib-1.0/json-glib/json-glib.h>

#include "json_glib_helper.h"
#include "kpi.h"

void json_add_string(JsonBuilder *builder, const char *name, const char *value)
{
	json_builder_set_member_name(builder, name);
	json_builder_add_string_value(builder, value);
}

void json_add_number(JsonBuilder *builder, const char *name, int64_t value)
{
	json_builder_set_member_name(builder, name);
	json_builder_add_int_value(builder, value);
}

void json_add_double(JsonBuilder *builder, const char *name, gdouble value)
{
	json_builder_set_member_name(builder, name);
	json_builder_add_double_value(builder, value);
}

void json_add_nested_kpi_number(JsonBuilder *builder, const char *name, int64_t value)
{
	JsonBuilder *sub_builder = json_builder_new();
	json_builder_begin_object(sub_builder);
	json_add_number(sub_builder, "value", value);
	json_builder_end_object(sub_builder);

	JsonNode *sub_node = json_builder_get_root(sub_builder);
	g_object_unref(sub_builder);

	json_builder_set_member_name(builder, name);
	json_builder_add_value(builder, sub_node);
}

void json_add_nested_kpi_double(JsonBuilder *builder, const char *name, gdouble value)
{
	JsonBuilder *sub_builder = json_builder_new();
	json_builder_begin_object(sub_builder);
	json_add_double(sub_builder, "value", value);
	json_builder_end_object(sub_builder);

	JsonNode *sub_node = json_builder_get_root(sub_builder);
	g_object_unref(sub_builder);

	json_builder_set_member_name(builder, name);
	json_builder_add_value(builder, sub_node);
}
