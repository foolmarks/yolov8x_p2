#ifndef _KPI_H
#define _KPI_H

#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include <inttypes.h>
#include <json-glib-1.0/json-glib/json-glib.h>
#include <time.h>

#include "trace.h"

typedef struct {
	uint64_t frame_id;
	GString *plugin_id;
	uint32_t element_id_hash;
	const char *plugin_type;
	GString *stream_id;
	uint32_t qid;

	uint64_t kernel_start;
	uint64_t kernel_end;
	uint64_t plugin_start;
	uint64_t plugin_end;
} plugin_kpi_t;

typedef struct {
	GPtrArray *nodes;
	uint64_t last_updated;
} json_kpi_list_t;

////////////////////////////////////////////////////////////////////////////////////////////////////////
// plugin_kpi_t
////////////////////////////////////////////////////////////////////////////////////////////////////////

JsonNode *plugin_kpi_convert_to_json(const plugin_kpi_t *plugin_kpi);

int plugin_kpi_is_all_timestamp_set(const plugin_kpi_t *plugin_kpi);
int plugin_kpi_is_plugin_timestamp_set(const plugin_kpi_t *plugin_kpi);
int plugin_kpi_is_kernel_timestamp_set(const plugin_kpi_t *plugin_kpi);

plugin_kpi_t *plugin_kpi_create_from_trace(const trace_t *trace);
int  plugin_kpi_merge_trace(plugin_kpi_t *plugin_kpi, const trace_t *trace);
void plugin_kpi_free(plugin_kpi_t *plugin_kpi);
void plugin_kpi_print(const plugin_kpi_t *plugin_kpi);

////////////////////////////////////////////////////////////////////////////////////////////////////////
// plugin_kpi_list
////////////////////////////////////////////////////////////////////////////////////////////////////////

void plugin_kpi_list_store(GHashTable *map, plugin_kpi_t *plugin_kpi);
void plugin_kpi_list_destroy_map(GHashTable* map);
void plugin_kpi_list_print_map(GHashTable *map);
void plugin_kpi_list_remove_plugin_kpi_by_key(GHashTable *map, uint32_t element_id_hash, uint32_t frame_id);

plugin_kpi_t *plugin_kpi_list_get_plugin_kpi_by_key(GHashTable *map, uint64_t key);

////////////////////////////////////////////////////////////////////////////////////////////////////////
// json_kpi_list_t
////////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t json_kpi_list_generate_id(const uint32_t stream_id, const uint32_t frame_id);
void json_kpi_list_add_json_node(GHashTable *map, uint64_t id, JsonNode *node);
gboolean json_kpi_list_is_full(GHashTable *map, uint64_t id, guint limit);
void json_kpi_list_free(json_kpi_list_t *list);
void json_kpi_list_free_map(GHashTable *map);

JsonNode *json_kpi_list_make_pipeline_kpi(GHashTable *map, uint64_t id, const char *pipeline_id, pid_t pid, const char* stream_id);

#endif // _KPI_H
