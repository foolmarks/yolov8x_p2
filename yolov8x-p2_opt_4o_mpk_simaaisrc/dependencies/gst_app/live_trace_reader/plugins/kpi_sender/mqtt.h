#ifndef _MQTT_H
#define _MQTT_H

#include <glib.h>
#include <json-glib-1.0/json-glib/json-glib.h>
#include <stdbool.h>
#include <mosquitto.h>

#define SIMMAI_MQTT_KPI_PUB_TOPIC           "simaai/gst/kpis"
#define SIMAAI_MQTT_KPI_REQ_TOPIC           "simaai/gst/req"
#define SIMAAI_MQTT_KPI_RES_TOPIC           "simaai/gst/res"

typedef void (*mqtt_message_callback)(struct mosquitto *, void *, const struct mosquitto_message *);

struct mosquitto * mqtt_init(pid_t pid, mqtt_message_callback on_message, void *cb_user_data);
bool mqtt_connect(struct mosquitto *mosq_instance, const char *host, int port, int keepalive);

bool mqtt_publish(struct mosquitto *mosq_instance, const char *topic, const char *message);

void mqtt_disconnect(struct mosquitto *mosq_instance);
void mqtt_deinit(struct mosquitto* mosq_instance);

#endif // _MQTT_H
