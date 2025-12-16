#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include <stdio.h>
#include <malloc.h>
#include <limits.h>
#include <simaai/simaailog.h>

#include "kpi_sender.h"
#include "trace.h"
#include "kpi.h"
#include "mqtt.h"
#include "json_glib_helper.h"
#include "utils.h"
#include <math.h>

#include <time.h>
#include <unistd.h>

#define SIMAAI_MQTT_HOST                    "127.0.0.1"
#define SIMAAI_MQTT_PORT                    1883
#define SIMAAI_MQTT_KEEPALIVE               60

#define START_KPIS_STR "start-kpis"
#define STOP_KPIS_STR "stop-kpis"

#define PARAM_STR_PIPELINE_ID "pipeline_id"
#define PARAM_STR_PID "pid"
#define PARAM_STR_PLUGINS_COUNT "plugins_count"

#define RCTD_INTERVAL_MS   (1000)
#define INSURANCE_DELAY_MS (250)
#define THRESHOLD_TIME_MS  (RCTD_INTERVAL_MS + INSURANCE_DELAY_MS)
#define OBSERVER_THREAD_PERIOD_CHECK_MS (100)

#define THRESHOLD_PUBLISHED_MSGS (5)

static void *observer_thread_func(void *arg);

bt_component_class_initialize_method_status sink_init(
    bt_self_component_sink *self_comp_sink,
    bt_self_component_sink_configuration *config,
    const bt_value *params,
    void *init_data)
{
    struct kpi_sender_sink *sink_data = (struct kpi_sender_sink *)malloc(sizeof(*sink_data));
    if (!sink_data) {
        return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_MEMORY_ERROR;
	}

	sink_data->plugin_kpi_map = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
	sink_data->json_kpi_map = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, (GDestroyNotify)json_kpi_list_free);

	pthread_mutex_init(&sink_data->mutex_json_kpi, NULL);

	// Parse parameters
	// Parse pipeline_id parameter
	bt_value *pipeline_id_value;
    if (bt_value_map_has_entry(params, PARAM_STR_PIPELINE_ID)) {
        pipeline_id_value = bt_value_map_borrow_entry_value((bt_value *)params, PARAM_STR_PIPELINE_ID);
        sink_data->pipeline_id = bt_value_string_get(pipeline_id_value);
    } else {
        simaailog(SIMAAILOG_ERR, "kpi_sender initialization: 'pipeline_id' field did not set.");
		return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_ERROR;
	}

	// Parse gst_app pid parameter
	bt_value *pid_value;
    if (bt_value_map_has_entry(params, PARAM_STR_PID)) {
        pid_value = bt_value_map_borrow_entry_value((bt_value *)params, PARAM_STR_PID);
        sink_data->pipeline_pid = bt_value_integer_signed_get(pid_value);
    } else {
        simaailog(SIMAAILOG_ERR, "kpi_sender initialization: 'pid' field did not set");
		return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_ERROR;
	}

	// Parse plugins_count parameter
	bt_value *plugins_count_value;
    if (bt_value_map_has_entry(params, PARAM_STR_PLUGINS_COUNT)) {
        plugins_count_value = bt_value_map_borrow_entry_value((bt_value *)params, PARAM_STR_PLUGINS_COUNT);
        sink_data->plugins_count = bt_value_integer_unsigned_get(plugins_count_value);
        simaailog(SIMAAILOG_INFO, "kpi_sender initialization: 'plugins_count' set as %u", sink_data->plugins_count);

    } else {
        simaailog(SIMAAILOG_ERR, "kpi_sender initialization: 'plugins_count' field did not set");
		return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_ERROR;
	}

	sink_data->mosquitto_instance = mqtt_init(sink_data->pipeline_pid, sink_data);
	if (sink_data->mosquitto_instance == NULL) {
        simaailog(SIMAAILOG_ERR, "kpi_sender initialization: cannot create a mosquitto_instance");
		return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_ERROR;
	}

	bool ret = mqtt_connect(sink_data->mosquitto_instance, SIMAAI_MQTT_HOST, SIMAAI_MQTT_PORT, SIMAAI_MQTT_KEEPALIVE);
	if (!ret) {
        simaailog(SIMAAILOG_ERR, "kpi_sender initialization: cannot connect to a MQTT '%s:%d'", SIMAAI_MQTT_HOST, SIMAAI_MQTT_PORT);
		return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_ERROR;
	}

#ifdef LTR_DEBUG_STATISTICS
	sink_data->sent_kpis_count = 0;
#endif // LTR_DEBUG_STATISTICS

	sink_data->sent_plugins_count = 0;
	sink_data->is_running = 1;
	pthread_create(&sink_data->observer_thread, NULL, observer_thread_func, sink_data);

	// Store private structure
	bt_self_component_set_data(bt_self_component_sink_as_self_component(self_comp_sink), sink_data);

	// Add input port
	bt_self_component_sink_add_input_port(self_comp_sink, "in", NULL, NULL);

	simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: kpi_sender successfully initialized");
	return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_OK;
}

void sink_finalize(bt_self_component_sink *self_comp_sink)
{
	struct kpi_sender_sink *sink_data = (struct kpi_sender_sink *)bt_self_component_get_data(bt_self_component_sink_as_self_component(self_comp_sink));

	sink_data->is_running = 0;
	pthread_join(sink_data->observer_thread, NULL);

	mqtt_disconnect(sink_data->mosquitto_instance);
	mqtt_deinit(sink_data->mosquitto_instance);

	plugin_kpi_list_destroy_map(sink_data->plugin_kpi_map);
	json_kpi_list_free_map(sink_data->json_kpi_map);

	pthread_mutex_destroy(&sink_data->mutex_json_kpi);

#ifdef LTR_DEBUG_STATISTICS
	simaailog(SIMAAILOG_INFO, "[kpi_sender]: kpi_sender sent %d KPI frames", sink_data->sent_kpis_count);
#endif // LTR_DEBUG_STATISTICS

    free(sink_data);

    simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: kpi_sender successfully deinitialized");
}

/*
 * Called when the trace processing graph containing the sink component
 * is configured.
 *
 * This is where we can create our upstream message iterator.
 */
static bt_component_class_sink_graph_is_configured_method_status graph_is_configured(bt_self_component_sink *self_component_sink)
{
	struct kpi_sender_sink *sink_data = (struct kpi_sender_sink *)bt_self_component_get_data(bt_self_component_sink_as_self_component(self_component_sink));

    // Borrow our unique port
    bt_self_component_port_input *in_port = bt_self_component_sink_borrow_input_port_by_index(self_component_sink, 0);

    // Create the upstream message iterator
    bt_message_iterator_create_from_sink_component(self_component_sink, in_port, &sink_data->msg_iter);

    return BT_COMPONENT_CLASS_SINK_GRAPH_IS_CONFIGURED_METHOD_STATUS_OK;
}

static void publish_pipeline_kpi(struct kpi_sender_sink *sink, gpointer key)
{
	json_kpi_list_t *list = (json_kpi_list_t *)g_hash_table_lookup(sink->json_kpi_map, key);

	// combine all plugins kpi into single kpi message
	JsonNode *json_kpi = g_ptr_array_index(list->nodes, 0);
	JsonObject *obj = json_node_get_object(json_kpi);
	char stream_id[64];
	strcpy(stream_id, json_object_get_string_member(obj, "stream_id"));

#ifdef LTR_DEBUG_MQTT
	gint64 frame_id = json_object_get_int_member(obj, "frame_id");
	int plugins_count = json_kpi_list_get_plugins_count(list);
#endif // LTR_DEBUG_MQTT

	JsonNode *pipeline_kpi = json_kpi_list_make_pipeline_kpi(sink->json_kpi_map, *((uint64_t*)key), sink->pipeline_id, sink->pipeline_pid, (const char*)stream_id);
	char *kpi_str = json_to_string(pipeline_kpi, FALSE);
	json_node_unref(pipeline_kpi);

#ifdef LTR_DEBUG_MQTT
	simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Publishing KPI for '%s#%ld' with plugins count: %d", stream_id, frame_id, plugins_count);
#endif // LTR_DEBUG_MQTT

	if (!mqtt_publish(sink->mosquitto_instance, SIMMAI_MQTT_KPI_PUB_TOPIC, kpi_str)) {
		printf("MQTT publishing error: something went wrong during publishing kpi\n");
		simaailog(SIMAAILOG_ERR, "MQTT publishing error: something went wrong during publishing kpi");
	} else {
#ifdef LTR_DEBUG_STATISTICS
		sink->sent_kpis_count++;
#endif // LTR_DEBUG_STATISTICS

#ifdef LTR_DEBUG_MQTT
		simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: KPI published for '%s#%ld'", stream_id, frame_id);
#endif // LTR_DEBUG_MQTT
	}

	g_free(kpi_str);
}


bt_component_class_sink_consume_method_status consume(bt_self_component_sink *self_comp_sink)
{
    struct kpi_sender_sink *sink_data = (struct kpi_sender_sink *)bt_self_component_get_data(bt_self_component_sink_as_self_component(self_comp_sink));
	bt_component_class_sink_consume_method_status status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_OK;

    /* Consume a batch of messages from the upstream message iterator */
    bt_message_array_const messages = NULL;
    uint64_t message_count = 0;
    bt_message_iterator_next_status next_status = bt_message_iterator_next(sink_data->msg_iter, &messages, &message_count);
    uint32_t actual_plugins_count = sink_data->sent_plugins_count;
    if (actual_plugins_count == 0)
    {
        actual_plugins_count = sink_data->plugins_count;
    }

    switch (next_status) {
    case BT_MESSAGE_ITERATOR_NEXT_STATUS_END:
        /* End of iteration: put the message iterator's reference */
        bt_message_iterator_put_ref(sink_data->msg_iter);
        status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_END;
        goto end;
    case BT_MESSAGE_ITERATOR_NEXT_STATUS_AGAIN:
        status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_AGAIN;
        goto end;
    case BT_MESSAGE_ITERATOR_NEXT_STATUS_MEMORY_ERROR:
        status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_MEMORY_ERROR;
        goto end;
    case BT_MESSAGE_ITERATOR_NEXT_STATUS_ERROR:
        status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_ERROR;
        goto end;
    default:
        break;
    }

    /* For each consumed message */
    for (uint64_t i = 0; i < message_count; i++) {
        const bt_message *message = messages[i];

		bt_message_type msg_type = bt_message_get_type(message);
		if (msg_type == BT_MESSAGE_TYPE_EVENT) {
			// 1. LTTNG -> C struct
			// Parse trace from message
			trace_t trace = {0};
			if (trace_parse_from_message(message, &trace)) {
				continue;
			}
			// trace_print(&trace);

			// 2. C struct -> Plugin_kpi_t
			// Find existing plugin_kpi and merge trace to plugin kpi
			// or
			// create new plugin_kpi based on the trace
			uint64_t request_id = trace_generate_request_id_from_trace(&trace);
			plugin_kpi_t *plugin_kpi = plugin_kpi_list_get_plugin_kpi_by_key(sink_data->plugin_kpi_map, request_id);
			if (plugin_kpi) {
				int ret = plugin_kpi_merge_trace(plugin_kpi, &trace);
				if (ret) {
					simaailog(SIMAAILOG_ERR, "Cannot add trace data into plugin_kpi");
					continue;
				}
#ifdef LTR_DEBUG_PLUGIN_KPI
				const char *plugin_id_str = (plugin_kpi->plugin_id && plugin_kpi->plugin_id->str) ? plugin_kpi->plugin_id->str : "";
				const char *stream_id_str = (plugin_kpi->stream_id && plugin_kpi->stream_id->str) ? plugin_kpi->stream_id->str : "";

				simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Merged plugins KPI for '%s' plugin, '%s#%lu' stream",
					plugin_id_str, stream_id_str, plugin_kpi->frame_id);
#endif // LTR_DEBUG_PLUGIN_KPI
			} else {
				plugin_kpi = plugin_kpi_create_from_trace(&trace);
				plugin_kpi_list_store(sink_data->plugin_kpi_map, plugin_kpi);
#ifdef LTR_DEBUG_PLUGIN_KPI
				const char *plugin_id_str = (plugin_kpi->plugin_id && plugin_kpi->plugin_id->str) ? plugin_kpi->plugin_id->str : "";
				const char *stream_id_str = (plugin_kpi->stream_id && plugin_kpi->stream_id->str) ? plugin_kpi->stream_id->str : "";

				simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Created plugins KPI for '%s' plugin, '%s#%lu' stream",
					plugin_id_str, stream_id_str, plugin_kpi->frame_id);
#endif // LTR_DEBUG_PLUGIN_KPI
			}

			// PCIe KPI creation (specific plugin KPI merging way)
			// At the start event we do not know anything about PCIe data/buffer,
			// we provide on the plugin side only plugin_id and timestamp.
			// Also we set frame_id as -1 that means 4294967295U in the uin32_t.
			// When we receive EVENT_TYPE_END we can get all info, but we have add a start timestamp for this KPI
			if (strstr(trace.event_name->str, "PCIe") && trace.event_type == EVENT_TYPE_END) {
				// Since the pciesrc plugin does not know any metadata before processing the buffer,
				// we have to use NULL instead of stream_id.
				GString *pcie_start_element_id = trace_make_element_id(trace.plugin_id->str, NULL);
				uint32_t pcie_start_element_id_hash = hash_string_to_uint32(pcie_start_element_id->str);
				g_string_free(pcie_start_element_id, TRUE);

				// Find KPI with start timestamp
				uint64_t request_id_start_event = trace_generate_request_id(pcie_start_element_id_hash, -1);
				plugin_kpi_t *plugin_kpi_start_event = plugin_kpi_list_get_plugin_kpi_by_key(sink_data->plugin_kpi_map, request_id_start_event);
				if (plugin_kpi_start_event == NULL) {
					fprintf(stderr, "LTR: Did not find a plugin_kpi_t with plugin_id: %s\n", trace.plugin_id->str);

					trace_free(&trace);
					continue;
				}

				// Add start timestamp to the PCIe KPI
				plugin_kpi->plugin_start = plugin_kpi_start_event->plugin_start;

#ifdef LTR_DEBUG_PLUGIN_KPI
				simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Updated PCIe data for plugins KPI for '%s' plugin, '%s#%lu' stream",
					plugin_kpi->plugin_id->str, plugin_kpi->stream_id->str, plugin_kpi->frame_id);
#endif // LTR_DEBUG_PLUGIN_KPI
			}
			trace_free(&trace);

			// 3. Plugin_kpi_t -> Json_kpi_t
			// plugin_kpi_print(plugin_kpi);
			bt_bool is_need_to_convert_to_json = BT_FALSE;
			if (is_remote_core_kpi(plugin_kpi)) {
				is_need_to_convert_to_json = plugin_kpi_is_all_timestamp_set(plugin_kpi);
			} else {
				is_need_to_convert_to_json = plugin_kpi_is_plugin_timestamp_set(plugin_kpi);
			}

			if (is_need_to_convert_to_json) {
				// printf("KPI is ready to convert to JSON: ['%s', '%s', %lu] \n", plugin_kpi->plugin_id->str, plugin_kpi->stream_id->str, plugin_kpi->frame_id);
#ifdef LTR_DEBUG_STREAM_KPI
				simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Add KPI for '%s' plugin, '%s#%lu' stream",
					plugin_kpi->plugin_id->str, plugin_kpi->stream_id->str, plugin_kpi->frame_id);
#endif // LTR_DEBUG_STREAM_KPI
				JsonNode *json = plugin_kpi_convert_to_json(plugin_kpi);

				uint64_t key = json_kpi_list_generate_id(hash_string_to_uint32(plugin_kpi->stream_id->str), plugin_kpi->frame_id);

				pthread_mutex_lock(&sink_data->mutex_json_kpi);
				json_kpi_list_add_json_node(sink_data->json_kpi_map, key, json);

				if (json_kpi_list_is_full(sink_data->json_kpi_map, key, actual_plugins_count)) {
#ifdef LTR_DEBUG_MQTT
					simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Publishing KPI due to the KPI is full");
#endif // LTR_DEBUG_MQTT
					if (sink_data->sent_plugins_count == 0)
					{
						sink_data->sent_plugins_count = actual_plugins_count;
					}
					publish_pipeline_kpi(sink_data, (gpointer)&key);
				}
				pthread_mutex_unlock(&sink_data->mutex_json_kpi);

				// plugin_kpi_list_print_map(sink_data->plugin_kpi_map);

				plugin_kpi_list_remove_plugin_kpi_by_key(sink_data->plugin_kpi_map, plugin_kpi->element_id_hash, plugin_kpi->frame_id);
			}
		}

        bt_message_put_ref(message);
    }

end:
    return status;
}

static int get_maximum_plugins_count(struct kpi_sender_sink *sink)
{
	GHashTableIter iter;
	gpointer key, value;
	int max_count = 0;

	g_hash_table_iter_init(&iter, sink->json_kpi_map);
	while (g_hash_table_iter_next(&iter, &key, &value))
	{
		json_kpi_list_t *list = (json_kpi_list_t *)value;
		int plugins_count_in_msg = json_kpi_list_get_plugins_count(list);

		if (max_count < plugins_count_in_msg)
		{
			max_count = plugins_count_in_msg;
		}
	}

	return max_count;
}

static void *observer_thread_func(void *arg)
{
	simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Observer thread has started");

	struct kpi_sender_sink *sink = (struct kpi_sender_sink *)arg;

	while (sink->is_running)
	{
		GHashTableIter iter;
		gpointer key, value;
		pthread_mutex_lock(&sink->mutex_json_kpi);
		g_hash_table_iter_init(&iter, sink->json_kpi_map);

		struct timespec now;
		clock_gettime(CLOCK_MONOTONIC, &now);
		uint64_t now_ms = TIMESPEC_TO_MS(now);

#ifdef LTR_DEBUG_STATISTICS
		simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Observer thread checks the list");

		int checked_count = 0;

		guint list_size_before = g_hash_table_size(sink->json_kpi_map);
#endif // LTR_DEBUG_STATISTICS

		int published_msgs = 0;
		int maximum_plugins_count;
		if (sink->sent_plugins_count == 0)
		{
			maximum_plugins_count = get_maximum_plugins_count(sink);
		} else {
			maximum_plugins_count = sink->sent_plugins_count;
		}

		while (g_hash_table_iter_next(&iter, &key, &value))
		{
			json_kpi_list_t *list = (json_kpi_list_t *)value;

			uint64_t diff_ms = now_ms - list->last_updated;
			if(diff_ms > THRESHOLD_TIME_MS)
			{
#ifdef LTR_DEBUG_MQTT
				simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Publishing KPI due to the timeout");
#endif // LTR_DEBUG_MQTT

				json_kpi_list_t *list = (json_kpi_list_t *)g_hash_table_lookup(sink->json_kpi_map, key);
				int plugins_count_in_msg = json_kpi_list_get_plugins_count(list);

				if (plugins_count_in_msg >= maximum_plugins_count)
				{
					publish_pipeline_kpi(sink, key);
					published_msgs++;

					// Invalid iterator, refresh this
					g_hash_table_iter_init(&iter, sink->json_kpi_map);
				}

#ifdef LTR_DEBUG_STATISTICS
				published_msgs++;
#endif // LTR_DEBUG_STATISTICS

			}

#ifdef LTR_DEBUG_STATISTICS
			checked_count++;
#endif // LTR_DEBUG_STATISTICS
		}
		pthread_mutex_unlock(&sink->mutex_json_kpi);

		// Set actual plugins count, only if we have sent a lot of msgs.
		// This helps to avoid fake maximum plugins count setting
		if (sink->sent_plugins_count == 0 && published_msgs > THRESHOLD_PUBLISHED_MSGS)
		{
			sink->sent_plugins_count = maximum_plugins_count;
			simaailog(SIMAAILOG_INFO, "[kpi_sender]: Set actual plugins count as '%d'", sink->sent_plugins_count);
		}


#ifdef LTR_DEBUG_STATISTICS
		guint list_size_after = g_hash_table_size(sink->json_kpi_map);
		simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Observer thread check result: published: %d, checked: %d, size_before: %d, size_after: %d",
			published_msgs, checked_count, list_size_before, list_size_after);
#endif // LTR_DEBUG_STATISTICS

		usleep(OBSERVER_THREAD_PERIOD_CHECK_MS * 1000);
	}

	simaailog(SIMAAILOG_DEBUG, "[kpi_sender]: Thread finished");
	return NULL;
}


BT_PLUGIN_MODULE();
BT_PLUGIN(simaai_kpi);
BT_PLUGIN_DESCRIPTION("This plugin recieves the all LTTNG-traces and sends it to the MQTT topic");
BT_PLUGIN_AUTHOR("Mykola Voitekh");
BT_PLUGIN_LICENSE("MIT");

BT_PLUGIN_SINK_COMPONENT_CLASS(kpi_sink, consume);
BT_PLUGIN_SINK_COMPONENT_CLASS_INITIALIZE_METHOD(kpi_sink, sink_init);
BT_PLUGIN_SINK_COMPONENT_CLASS_FINALIZE_METHOD(kpi_sink, sink_finalize);
BT_PLUGIN_SINK_COMPONENT_CLASS_GRAPH_IS_CONFIGURED_METHOD(kpi_sink, graph_is_configured);
