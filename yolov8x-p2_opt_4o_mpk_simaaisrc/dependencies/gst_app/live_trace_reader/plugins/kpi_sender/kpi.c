#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include <inttypes.h>
#include <stdio.h>
#include <simaai/simaailog.h>

#include "trace.h"
#include "kpi.h"
#include "json_glib_helper.h"
#include "kpi_sender.h"
#include <inttypes.h>
#include "utils.h"

///////////////////////////////////////////////////////////////////////////////
// plugin_kpi_t
///////////////////////////////////////////////////////////////////////////////

void plugin_kpi_print(const plugin_kpi_t *plugin_kpi)
{
	if (!plugin_kpi) {
        simaailog(SIMAAILOG_ERR, "Cannot print a plugin_kpi: plugin_kpi is NULL");
		return;
	}

	printf(
		"{ frame_id=%lu, plugin_id=%s, element_id_hash=%" PRIu64 ", plugin_type=%s, stream_id=%s, "
		"kernel_start=%lu, kernel_end=%lu,"
		"plugin_start=%lu, plugin_end=%lu, qid=%u  }\n",
		plugin_kpi->frame_id,
		plugin_kpi->plugin_id ? plugin_kpi->plugin_id->str : "", plugin_kpi->element_id_hash,
		plugin_kpi->plugin_type, plugin_kpi->stream_id->str,
		plugin_kpi->kernel_start, plugin_kpi->kernel_end,
		plugin_kpi->plugin_start, plugin_kpi->plugin_end,
		plugin_kpi->qid
	);
}

int plugin_kpi_is_kernel_timestamp_set(const plugin_kpi_t *plugin_kpi)
{
	if (!plugin_kpi) {
        simaailog(SIMAAILOG_ERR, "Cannot check kernel timestamps: plugin_kpi is NULL");
		return -1;
	}

	if (plugin_kpi->kernel_start != 0 && plugin_kpi->kernel_end != 0) {
		return 1;
	}

	return 0;
}

int plugin_kpi_is_plugin_timestamp_set(const plugin_kpi_t *plugin_kpi)
{
	if (!plugin_kpi) {
        simaailog(SIMAAILOG_ERR, "Cannot check plugin timestamps: plugin_kpi is NULL");
		return -1;
	}

	if (plugin_kpi->plugin_start != 0 && plugin_kpi->plugin_end != 0) {
		return 1;
	}

	return 0;
}

int plugin_kpi_is_all_timestamp_set(const plugin_kpi_t *plugin_kpi)
{
	int plugin_timestamps = plugin_kpi_is_plugin_timestamp_set(plugin_kpi);
	int kernel_timestamps = plugin_kpi_is_kernel_timestamp_set(plugin_kpi);

	if (plugin_timestamps < 0 || kernel_timestamps < 0) {
        simaailog(SIMAAILOG_ERR, "Cannot check all timestamps: plugin_kpi is NULL");
		return -1;
	}

	return plugin_timestamps && kernel_timestamps;
}

JsonNode *plugin_kpi_convert_to_json(const plugin_kpi_t *plugin_kpi)
{
	JsonBuilder *kpi_builder = json_builder_new();
	json_builder_begin_object(kpi_builder);
	json_add_nested_kpi_number(kpi_builder, "kernelStartTime", plugin_kpi->kernel_start);
	json_add_nested_kpi_number(kpi_builder, "kernelEndTime", plugin_kpi->kernel_end);
	json_add_nested_kpi_number(kpi_builder, "pluginStartTime", plugin_kpi->plugin_start);
	json_add_nested_kpi_number(kpi_builder, "pluginEndTime", plugin_kpi->plugin_end);

	gdouble execution_time;
	if (plugin_kpi->kernel_start && plugin_kpi->kernel_end) {
		execution_time = plugin_kpi->kernel_end - plugin_kpi->kernel_start;
	} else {
		execution_time = plugin_kpi->plugin_end - plugin_kpi->plugin_start;
	}
	execution_time /= 1000.0; // convert from us to ms
	json_add_nested_kpi_double(kpi_builder, "executionTime", execution_time);

	json_add_nested_kpi_number(kpi_builder, "powerConsumed", 0);
	json_add_nested_kpi_number(kpi_builder, "dma_BW", 0);
	json_add_nested_kpi_number(kpi_builder, "dram_SIZE", 0);

	json_builder_end_object(kpi_builder);
	JsonNode *kpi_node = json_builder_get_root(kpi_builder);

	JsonBuilder *root_builder = json_builder_new();
	json_builder_begin_object(root_builder);
	json_add_number(root_builder, "frame_id", plugin_kpi->frame_id);
	json_add_string(root_builder, "plugin_id", plugin_kpi->plugin_id ? plugin_kpi->plugin_id->str : "");
	json_add_string(root_builder, "plugin_type", plugin_kpi->plugin_type);
	json_add_string(root_builder, "stream_id", plugin_kpi->stream_id->str);
	if (plugin_kpi->qid != QUERY_ID_INVALID) {
		json_add_number(root_builder, "qid", plugin_kpi->qid);
	}

	json_builder_set_member_name(root_builder, "kpis");
	json_builder_add_value(root_builder, kpi_node);

	json_builder_end_object(root_builder);

	JsonNode *root = json_builder_get_root(root_builder);

	g_object_unref(kpi_builder);
	g_object_unref(root_builder);

	return root;
}

static uint64_t plugin_kpi_generate_request_id(const plugin_kpi_t *plugin_kpi)
{
	if (plugin_kpi == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot generate request_id: plugin_kpi is NULL");
		return 0;
	}

	return ((uint64_t)plugin_kpi->element_id_hash << 32) | plugin_kpi->frame_id;
}

static int plugin_kpi_insert_timestamp(plugin_kpi_t *plugin_kpi, const trace_t *trace)
{
	if (plugin_kpi == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot insert trace fields into plugin_kpi: plugin_kpi is NULL");
		return 0;
	}

	if (trace == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot insert trace fields into plugin_kpi: trace is NULL");
		return 0;
	}

	// Check request_id
	uint64_t trace_req = trace_generate_request_id_from_trace(trace);
	uint64_t plugin_kpi_req = plugin_kpi_generate_request_id(plugin_kpi);

	if (trace_req != plugin_kpi_req) {
        simaailog(SIMAAILOG_ERR, "Cannot insert trace fields into plugin_kpi: request_id for trace and plugin kpi is different");
		return -1;
	}

	int is_remote_core = (intptr_t)strstr(trace->event_name->str, TRACE_EVENT_CLASS_NAME_REMOTE_CORE);
	if (is_remote_core) {
		if (trace->event_type == EVENT_TYPE_START) {
			plugin_kpi->kernel_start = trace->timestamp;
			return 0;
		} else if (trace->event_type == EVENT_TYPE_END) {
			plugin_kpi->kernel_end = trace->timestamp;
			return 0;
		}
	} else {
		if (trace->event_type == EVENT_TYPE_START) {
			plugin_kpi->plugin_start = trace->timestamp;
			return 0;
		} else if (trace->event_type == EVENT_TYPE_END) {
			plugin_kpi->plugin_end = trace->timestamp;
			return 0;
		}
	}

	// We reach this line, that means that this is trace with custom event, this is an error
	simaailog(SIMAAILOG_ERR, "Cannot insert trace fields into plugin_kpi: something went wrong");
	return -1;
}

plugin_kpi_t *plugin_kpi_create_from_trace(const trace_t *trace)
{
	if (trace == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot create a plugin_kpi from trace: trace is NULL");
		return NULL;
	}

	plugin_kpi_t *result = calloc(1, sizeof(plugin_kpi_t));

	result->frame_id = trace->frame_id;
	result->element_id_hash = trace->element_id_hash;

	if (trace->plugin_id) {
		result->plugin_id = g_string_new(trace->plugin_id->str);
	} else {
		result->plugin_id = NULL;
	}

	result->plugin_type = trace->plugin_type;
	result->stream_id = g_string_new(trace->stream_id->str);
	result->qid = trace->qid;

	plugin_kpi_insert_timestamp(result, trace);

	return result;
}

int plugin_kpi_merge_trace(plugin_kpi_t *plugin_kpi, const trace_t *trace)
{
	if (plugin_kpi == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot merge a trace into a plugin_kpi: plugin_kpi is NULL");
		return -1;
	}

	if (trace == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot merge a trace into a plugin_kpi: trace is NULL");
		return -1;
	}

	// Check request_id
	uint64_t trace_req = trace_generate_request_id_from_trace(trace);
	uint64_t plugin_kpi_req = plugin_kpi_generate_request_id(plugin_kpi);

	if (trace_req != plugin_kpi_req) {
        simaailog(SIMAAILOG_ERR, "Cannot merge a trace into a plugin_kpi: request_id for trace and plugin kpi is different");
		return -1;
	}

	if (plugin_kpi->stream_id->len == 0 && trace->stream_id->len != 0) {
		// FIXME: replace with g_string_append()
		plugin_kpi->stream_id = g_string_new(trace->stream_id->str);
	}

	if (plugin_kpi->qid == QUERY_ID_INVALID) {
		plugin_kpi->qid = trace->qid;
	}

	return plugin_kpi_insert_timestamp(plugin_kpi, trace);
}

void plugin_kpi_free(plugin_kpi_t *plugin_kpi)
{
	if (plugin_kpi == NULL) {
		return;
	}

	if (plugin_kpi->plugin_id) {
		g_string_free(plugin_kpi->plugin_id, TRUE);
	}

	if (plugin_kpi->stream_id) {
		g_string_free(plugin_kpi->stream_id, TRUE);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// plugin_kpi_list
////////////////////////////////////////////////////////////////////////////////////////////////////////

static uint64_t _plugin_kpi_list_generate_key(uint32_t element_id_hash, uint32_t frame_id)
{
    return ((uint64_t)element_id_hash << 32) | frame_id;
}

void plugin_kpi_list_store(GHashTable* map, plugin_kpi_t* plugin_kpi)
{
	if (map == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot store plugin_kpi into a hash table: map(hash_table) is NULL");
		return;
	}

	if (plugin_kpi == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot store plugin_kpi into a hash table: plugin_kpi is NULL");
		return;
	}

    uint64_t key = _plugin_kpi_list_generate_key(plugin_kpi->element_id_hash, plugin_kpi->frame_id);
    g_hash_table_insert(map, g_memdup2(&key, sizeof(uint64_t)), plugin_kpi);
}

plugin_kpi_t *plugin_kpi_list_get_plugin_kpi_by_key(GHashTable *map, uint64_t key)
{
	if (map == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot get plugin_kpi from a hash table: map(hash_table) is NULL");
		return NULL;
	}

    return (plugin_kpi_t*)g_hash_table_lookup(map, &key);
}

static plugin_kpi_t* plugin_kpi_list_get_plugin_kpi(GHashTable* map, uint32_t element_id_hash, uint32_t frame_id)
{
	if (map == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot get plugin_kpi from a hash table: map(hash_table) is NULL");
		return NULL;
	}

	uint64_t key = _plugin_kpi_list_generate_key(element_id_hash, frame_id);
	return plugin_kpi_list_get_plugin_kpi_by_key(map, key);
}

void plugin_kpi_list_destroy_map(GHashTable* map)
{
	if (map == NULL) {
		return;
	}

    g_hash_table_destroy(map);
}

void plugin_kpi_list_print_map(GHashTable* map)
{
	if (map == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot print plugin_kpi_list: map(hash_table) is NULL");
		return;
	}

    GHashTableIter iter;
    gpointer key, value;
    g_hash_table_iter_init(&iter, map);
    while (g_hash_table_iter_next(&iter, &key, &value)) {
        plugin_kpi_t* kpi = (plugin_kpi_t*)value;
		plugin_kpi_print(kpi);
    }
}

void plugin_kpi_list_remove_plugin_kpi_by_key(GHashTable* map, uint32_t element_id_hash, uint32_t frame_id)
{
	if (map == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot remove plugin_kpi from the hash_table: map(hash_table) is NULL");
		return;
	}

	uint64_t key = _plugin_kpi_list_generate_key(element_id_hash, frame_id);

	plugin_kpi_t *kpi = plugin_kpi_list_get_plugin_kpi(map, element_id_hash, frame_id);
	plugin_kpi_free(kpi);

    g_hash_table_remove(map, &key);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// json_kpi_list_t
////////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t json_kpi_list_generate_id(const uint32_t stream_id, const uint32_t frame_id)
{
    return ((uint64_t)stream_id << 32) | frame_id;
}

// id -- is combination stream_id + frame_id
void json_kpi_list_add_json_node(GHashTable *table, uint64_t id, JsonNode *node)
{
	if (table == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot add json_node into the hash_table: hash_table is NULL");
		return;
	}

	if (node == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot add json_node into the hash_table: json_node is NULL");
		return;
	}

    json_kpi_list_t *list = g_hash_table_lookup(table, &id);
    if (!list) {
        list = g_new(json_kpi_list_t, 1);
		list->nodes = g_ptr_array_new();

        g_hash_table_insert(table, g_memdup2(&id, sizeof(uint64_t)), list);
    }
    g_ptr_array_add(list->nodes, json_node_ref(node));

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    list->last_updated = TIMESPEC_TO_MS(now);
}

gboolean json_kpi_list_is_full(GHashTable *table, uint64_t id, guint limit)
{
	if (table == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot check if json_kpi_list is full: hash_table is NULL");
		return FALSE;
	}

    json_kpi_list_t *list = g_hash_table_lookup(table, &id);
    return list && list->nodes->len >= limit;
}

/// @brief get first JsonNode in the list. NOTE: free the data and pointer afrer usage
/// @param table 
/// @param id 
/// @return 
static JsonNode *json_kpi_list_pop_json_node(GHashTable *table, uint64_t id)
{
	if (table == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot pop json_node from the json_kpi_list: hash_table is NULL");
		return NULL;
	}

    json_kpi_list_t *list = g_hash_table_lookup(table, &id);
    if (list && list->nodes->len > 0) {
        JsonNode *node = g_ptr_array_index(list->nodes, 0);
		if (node) {
			json_node_unref(node);
		}

        g_ptr_array_remove_index(list->nodes, 0);

        if (list->nodes->len == 0) {
            g_hash_table_remove(table, &id);
        }

        return node;
    }
    return NULL;
}

void json_kpi_list_free(json_kpi_list_t *list)
{
	if (list == NULL) {
		return;
	}

	if (list->nodes) {
		for (guint i = 0; i < list->nodes->len; i++) {
			json_node_unref(g_ptr_array_index(list->nodes, i));
		}
		g_ptr_array_free(list->nodes, TRUE);
		list->nodes = NULL;
	}
}

void json_kpi_list_free_map(GHashTable *table)
{
	if (table == NULL) {
		return;
	}

    g_hash_table_destroy(table);
}

JsonNode *json_kpi_list_make_pipeline_kpi(GHashTable *map, uint64_t id, const char *pipeline_id, pid_t pid, const char* stream_id)
{
	if (map == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot make pipeline_kpi: map(hash_table) is NULL");
		return NULL;
	}

	uint32_t frame_id = id & UINT32_MAX;

	JsonBuilder *main_builder = json_builder_new();
	json_builder_begin_object(main_builder);

	json_add_number(main_builder, "frame_id", frame_id);
	json_add_string(main_builder, "stream_id", stream_id ? stream_id : "");
	json_add_string(main_builder, "pipeline_id", pipeline_id ? pipeline_id : "");
	json_add_number(main_builder, "pid", pid);

	json_builder_set_member_name(main_builder, "plugins");
	json_builder_begin_array(main_builder);

	JsonNode *plugin_kpi = json_kpi_list_pop_json_node(map, id);
	while (plugin_kpi) {
		json_builder_add_value(main_builder, plugin_kpi);

		plugin_kpi = json_kpi_list_pop_json_node(map, id);
	}

	json_builder_end_array(main_builder);
	json_builder_end_object(main_builder);

	JsonNode *root = json_builder_get_root(main_builder);
	g_object_unref(main_builder);

	return root;
}

int json_kpi_list_get_plugins_count(json_kpi_list_t *list)
{
	return list ? list->nodes->len : -1;
}
