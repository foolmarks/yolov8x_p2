#ifndef _KPI_SENDER_H
#define _KPI_SENDER_H

#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include "mqtt.h"
#include <pthread.h>

// Private structure
struct kpi_sender_sink {
    bt_message_iterator *msg_iter;

	GHashTable *plugin_kpi_map;
	GHashTable *json_kpi_map;

	struct mosquitto *mosquitto_instance;

	pid_t pipeline_pid;
	const char *pipeline_id;
	bool push_to_mqtt;

	uint32_t plugins_count;

	pthread_t observer_thread;
	pthread_mutex_t mutex_json_kpi;
	int is_running;
};

#endif // _KPI_SENDER_H
