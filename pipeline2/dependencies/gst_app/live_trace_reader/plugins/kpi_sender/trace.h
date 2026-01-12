#ifndef _TRACE_H
#define _TRACE_H

#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include <inttypes.h>

#define TRACE_EVENT_CLASS_NAME_REMOTE_CORE "remote_core"

#define PLUGIN_TYPE_EV74            "EV74"
#define PLUGIN_TYPE_A65             "A65"
#define PLUGIN_TYPE_CVU             "CVU"
#define PLUGIN_TYPE_MLA             "MLA"
#define PLUGIN_TYPE_M4              "M4"
#define PLUGIN_TYPE_ALLEGRO_DECODER "AllegroDecoder"
#define PLUGIN_TYPE_ALLEGRO_ENCODER "AllegroEncoder"
#define PLUGIN_TYPE_PCIE            "PCIe"
#define PLUGIN_TYPE_PCIE_SRC        "PCIeSrc"
#define PLUGIN_TYPE_PCIE_SINK       "PCIeSink"
#define PLUGIN_TYPE_UNKNOWN         "UNKNOWN"

#define EVENT_TYPE_START  (0)
#define EVENT_TYPE_END    (1)

#define QUERY_ID_INVALID UINT32_MAX
typedef struct {
	uint64_t timestamp;
	uint64_t frame_id;
	GString *plugin_id;
	uint32_t element_id_hash; // plugin_id + stream_id
	const char *plugin_type;
	uint32_t event_type;
	GString *event_name;
	GString *stream_id;
	uint32_t qid; // PCIe only
} trace_t;

void trace_print(const trace_t *trace);
int  trace_parse_from_message(const bt_message *message, trace_t *trace);
GString *trace_make_element_id(const char *plugin_id, const char *stream_id);
uint64_t trace_generate_request_id_from_trace(const trace_t *trace);
uint64_t trace_generate_request_id(uint32_t element_id_hash, uint32_t frame_id);

void trace_free(trace_t *trace);

#endif // _TRACE_H
