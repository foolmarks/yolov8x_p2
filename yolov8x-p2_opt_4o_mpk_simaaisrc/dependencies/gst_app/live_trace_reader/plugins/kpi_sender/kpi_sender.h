#ifndef _KPI_SENDER_H
#define _KPI_SENDER_H

#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include "mqtt.h"
#include <pthread.h>

// #define LTR_DEBUG_MQTT
// #define LTR_DEBUG_STREAM_KPI
// #define LTR_DEBUG_PLUGIN_KPI
// #define LTR_DEBUG_STATISTICS

// Private structure
struct kpi_sender_sink {
    bt_message_iterator *msg_iter;

	GHashTable *plugin_kpi_map;
	GHashTable *json_kpi_map;

	struct mosquitto *mosquitto_instance;

	pid_t pipeline_pid;
	const char *pipeline_id;

	uint32_t plugins_count;
	uint32_t sent_plugins_count;

	pthread_t observer_thread;
	pthread_mutex_t mutex_json_kpi;
	int is_running;

#ifdef LTR_DEBUG_STATISTICS
	volatile int sent_kpis_count;
#endif // LTR_DEBUG_STATISTICS
};

#endif // _KPI_SENDER_H
