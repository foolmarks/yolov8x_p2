#include "mqtt.h"
#include <inttypes.h>
#include <mosquitto.h>
#include <stdio.h>
#include <unistd.h>
#include <simaai/simaailog.h>

#include "kpi_sender.h"

static void on_connect(struct mosquitto* mosq_instance, void* obj, int rc)
{
    if (rc == 0) {
        simaailog(SIMAAILOG_INFO, "Connected to the broker!");
    } else {
        simaailog(SIMAAILOG_ERR, "Failed to connect, return code: %d", rc);
        return;
    }
}

static void on_log(struct mosquitto* mosq_instance, void* obj, int level, const char* str)
{
    if (level & MOSQ_LOG_ERR) {
        simaailog(SIMAAILOG_ERR, str);
    } else if (level & MOSQ_LOG_WARNING) {
        simaailog(SIMAAILOG_WARNING, str);
    } else if (level & MOSQ_LOG_INFO) {
        simaailog(SIMAAILOG_INFO, str);
    } else if (level & MOSQ_LOG_DEBUG) {
        simaailog(SIMAAILOG_DEBUG, str);
    } else if (level & MOSQ_LOG_NOTICE) {
        simaailog(SIMAAILOG_NOTICE, str);
    }
}

struct mosquitto * mqtt_init(pid_t pid, void *cb_user_data)
{
	GString *client_id = g_string_new("kpi_client_");
	g_string_append_printf(client_id, "%u", pid);

    mosquitto_lib_init();
    struct mosquitto *mosq_instance = mosquitto_new(client_id->len ? client_id->str : NULL, true, cb_user_data);
    if (!mosq_instance) {
        simaailog(SIMAAILOG_ERR, "Cannot create a new mosquitto client instance");
		g_string_free(client_id, TRUE);
		return NULL;
    }

    mosquitto_connect_callback_set(mosq_instance, on_connect);
    mosquitto_log_callback_set(mosq_instance, on_log);

	g_string_free(client_id, TRUE);
	return mosq_instance;
}

void mqtt_deinit(struct mosquitto* mosq_instance)
{
    simaailog(SIMAAILOG_INFO, "MQTT destructor called");
    mosquitto_destroy(mosq_instance);
    mosquitto_lib_cleanup();
}

bool mqtt_connect(struct mosquitto *mosq_instance, const char *host, int port, int keepalive)
{
    if (mosquitto_connect(mosq_instance, host, port, keepalive) != MOSQ_ERR_SUCCESS) {
        simaailog(SIMAAILOG_ERR, "Unable to connect to MQTT broker");
        return false;
    }
    mosquitto_loop_start(mosq_instance);

    return true;
}

void mqtt_disconnect(struct mosquitto *mosq_instance)
{
    mosquitto_disconnect(mosq_instance);
    mosquitto_loop_stop(mosq_instance, true);
}

bool mqtt_publish(struct mosquitto *mosq_instance, const char *topic, const char *message)
{
	size_t message_size = strlen(message);
    int ret = mosquitto_publish(mosq_instance, NULL, topic, message_size, message, 0, false);
    if (ret != MOSQ_ERR_SUCCESS) {
        simaailog(SIMAAILOG_ERR, "Failed to publish message. Error code: %d", ret);
        return false;
    }

    return true;
}
