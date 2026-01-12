#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include <stdio.h>
#include <simaai/simaailog.h>

#include "trace.h"
#include "utils.h"
#include "kpi_sender.h"

#define NANOSEC_PER_MICROSEC (1000)

#define PAYLOAD_FIELD_CUSTOM_TIMESTAMP "custom_timestamp"
#define PAYLOAD_FIELD_REQUEST_ID       "request_id"
#define PAYLOAD_FIELD_FRAME_ID         "frame_id"
#define PAYLOAD_FIELD_PLUGIN_ID        "plugin_id"
#define PAYLOAD_FIELD_EVENT_TYPE       "event_type"
#define PAYLOAD_FIELD_STREAM_ID        "stream_id"
#define PAYLOAD_FIELD_QUERY_ID         "query_id"

static int get_timestamp_ms(const bt_message *message, uint64_t *result)
{
	if (!message) {
        simaailog(SIMAAILOG_ERR, "Cannot get timestamp: message is NULL");
		return -1;
	}

	if (!result) {
        simaailog(SIMAAILOG_ERR, "Cannot get timestamp: result pointer is NULL");
		return -1;
	}

	int64_t ts_nsec = 0;

	// Common linux timestamp
	if (!bt_message_event_borrow_stream_class_default_clock_class_const(message)) {
		/* No default clock class: skip the timestamp without an error */
        simaailog(SIMAAILOG_ERR, "Cannot get timestamp: No default clock class");
		return -1;
	} else {
		const bt_clock_snapshot *clock_snapshot = bt_message_event_borrow_default_clock_snapshot_const(message);
		bt_clock_snapshot_get_ns_from_origin(clock_snapshot, &ts_nsec);
	}

	*result = ts_nsec / NANOSEC_PER_MICROSEC;

	return 0;
}

void trace_print(const trace_t *trace)
{
	if (!trace) {
        simaailog(SIMAAILOG_ERR, "Cannot print a trace: trace is NULL");
		return;
	}

	printf(
		"{event_name=%s, timestamp=%lu, frame_id=%lu, plugin_id=%s, "
		"plugin_type=%s, event_type=%u, stream_id=%s, qid=%u}\n",
		trace->event_name && trace->event_name->str ? trace->event_name->str : "null",
		trace->timestamp,
		trace->frame_id,
		trace->plugin_id && trace->plugin_id->str ? trace->plugin_id->str : "null",
		trace->plugin_type ? trace->plugin_type : "null",
		trace->event_type,
		trace->stream_id && trace->stream_id->str ? trace->stream_id->str : "null",
		trace->qid
	);
}

int trace_parse_from_message(const bt_message *message, trace_t *trace)
{
	if (message == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot parse message: message is NULL");
		return -1;
	}
	if (trace == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot parse message: trace pointer is NULL");
		return -1;
	}

	if (bt_message_get_type(message) != BT_MESSAGE_TYPE_EVENT) {
        simaailog(SIMAAILOG_ERR, "Cannot parse message: messages type is not BT_MESSAGE_TYPE_EVENT");
		return -1;
	}

	const bt_event *event = bt_message_event_borrow_event_const(message);
	if (!event) {
        simaailog(SIMAAILOG_ERR, "Cannot parse message: event is NULL");
		return -1;
	}

	const bt_event_class *event_class = bt_event_borrow_class_const(event);
	if (!event_class) {
        simaailog(SIMAAILOG_ERR, "Cannot parse message: event_class is NULL");
		return -1;
	}

	const bt_field *payload_field = bt_event_borrow_payload_field_const(event);
	if (!payload_field) {
        simaailog(SIMAAILOG_ERR, "Cannot parse message: payload_field is NULL");
		return -1;
	}

	// Event name
	const char *event_name = bt_event_class_get_name(event_class);
	if (event_name == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot parse message: event_name is NULL");
		return -1;
	}
	trace->event_name = g_string_new(event_name);

	// Timestamp
	int is_remote_core = (intptr_t)strstr(event_name, TRACE_EVENT_CLASS_NAME_REMOTE_CORE);
	if (is_remote_core) {
		// Need to use a custom_timestamp
		const bt_field *custom_timestamp_field = NULL;
		get_payload_field_by_name(payload_field, PAYLOAD_FIELD_CUSTOM_TIMESTAMP, &custom_timestamp_field);
		if (custom_timestamp_field) {
			parse_uint64_field(custom_timestamp_field, &(trace->timestamp));
		} else {
			simaailog(SIMAAILOG_WARNING, "Timestamp set to 0: custom_timestamp_field does not exist for remote_core event. Timestamp will be set using the default Linux clock.");
			trace->timestamp = 0;
		}
	}
	if (trace->timestamp == 0) {
		// linux timestamp
		if (get_timestamp_ms(message, &(trace->timestamp))) {
			simaailog(SIMAAILOG_ERR, "Cannot parse message: cannot get a timestamp for the event");
			return -1;
		}
	}
	if (is_remote_core) {
		// request_id
		const bt_field *request_id_field = NULL;
		get_payload_field_by_name(payload_field, PAYLOAD_FIELD_REQUEST_ID, &request_id_field);

		uint64_t request_id;
		parse_uint64_field(request_id_field, &request_id);

		trace->frame_id = request_id & UINT32_MAX;
		trace->element_id_hash = (request_id >> 32) & UINT32_MAX;
		trace->plugin_id = NULL;
	} else {
		// frame_id
		const bt_field *frame_id_field = NULL;
		get_payload_field_by_name(payload_field, PAYLOAD_FIELD_FRAME_ID, &frame_id_field);
		uint64_t frame_id_u64;
		parse_uint64_field(frame_id_field, &frame_id_u64);
		trace->frame_id = frame_id_u64 & UINT32_MAX;

		// plugin_id
		const bt_field *plugin_id_field = NULL;
		get_payload_field_by_name(payload_field, PAYLOAD_FIELD_PLUGIN_ID, &plugin_id_field);
		trace->plugin_id = g_string_new("");
		parse_field(plugin_id_field, trace->plugin_id);
	}

	// plugin_type
	if (strstr(event_name, PLUGIN_TYPE_A65)) {
		trace->plugin_type = PLUGIN_TYPE_A65;
	} else if (strstr(event_name, PLUGIN_TYPE_CVU) || strstr(event_name, PLUGIN_TYPE_EV74)) {
		trace->plugin_type = PLUGIN_TYPE_CVU;
	} else if (strstr(event_name, PLUGIN_TYPE_MLA) || strstr(event_name, PLUGIN_TYPE_M4)) {
		trace->plugin_type = PLUGIN_TYPE_MLA;
	} else if (strstr(event_name, PLUGIN_TYPE_ALLEGRO_DECODER)) {
		trace->plugin_type = PLUGIN_TYPE_ALLEGRO_DECODER;
	} else if (strstr(event_name, PLUGIN_TYPE_ALLEGRO_ENCODER)) {
		trace->plugin_type = PLUGIN_TYPE_ALLEGRO_ENCODER;
	} else if (strstr(event_name, PLUGIN_TYPE_PCIE)) {
		trace->plugin_type = PLUGIN_TYPE_PCIE;
	} else {
		simaailog(SIMAAILOG_WARNING, "Cannot determine a plugin type: '%s'", event_name);
		trace->plugin_type = PLUGIN_TYPE_UNKNOWN;
	}

	// event_type
	const bt_field *event_type_field = NULL;
	get_payload_field_by_name(payload_field, PAYLOAD_FIELD_EVENT_TYPE, &event_type_field);
	uint64_t event_type;
	parse_uint64_field(event_type_field, &event_type);
	trace->event_type = event_type;

	// stream_id
	const bt_field *stream_id_field = NULL;
	get_payload_field_by_name(payload_field, PAYLOAD_FIELD_STREAM_ID, &stream_id_field);
	trace->stream_id = g_string_new("");
	if (stream_id_field) {
		parse_field(stream_id_field, trace->stream_id);

		GString *element_id = trace_make_element_id(trace->plugin_id->str, trace->stream_id->str);
		trace->element_id_hash = hash_string_to_uint32(element_id->str);
		g_string_free(element_id, TRUE);
	}

	// Query ID (PCIe only)
	const bt_field *qid_field = NULL;
	get_payload_field_by_name(payload_field, PAYLOAD_FIELD_QUERY_ID, &qid_field);
	uint64_t qid = QUERY_ID_INVALID;
	if (qid_field) {
		parse_uint64_field(qid_field, &qid);
	}
	trace->qid = qid;

	return 0;
}

void trace_free(trace_t *trace)
{
	if (trace->plugin_id) {
		g_string_free(trace->plugin_id, TRUE);
	}

	if (trace->event_name) {
		g_string_free(trace->event_name, TRUE);
	}

	if (trace->stream_id) {
		g_string_free(trace->stream_id, TRUE);
	}
}

// Note: use g_string_free() for returned value after usage
GString *trace_make_element_id(const char *plugin_id, const char *stream_id)
{
	GString *element_id = g_string_new("");
	g_string_printf(element_id, "%s%s", plugin_id, stream_id);

	return element_id;
}

uint64_t trace_generate_request_id(uint32_t element_id_hash, uint32_t frame_id)
{
    return ((uint64_t)element_id_hash << 32) | frame_id;
}

uint64_t trace_generate_request_id_from_trace(const trace_t *trace)
{
	if (trace == NULL) {
		return 0;
	}

    return trace_generate_request_id(trace->element_id_hash, trace->frame_id);
}
