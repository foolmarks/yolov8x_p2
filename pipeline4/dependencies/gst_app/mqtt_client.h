#ifndef MQTT_CLIENT_H
#define MQTT_CLIENT_H
#define SIMAAI_MQTT_HOST                    "127.0.0.1"
#define SIMAAI_MQTT_PORT                    1883
#define SIMAAI_MQTT_KEEPALIVE               60
#define SIMMAI_MQTT_KPI_PUB_TOPIC           "simaai/gst/kpis"
#define SIMAAI_MQTT_KPI_REQ_TOPIC           "simaai/gst/req"
#define SIMAAI_MQTT_KPI_RES_TOPIC           "simaai/gst/res"

#include <mosquitto.h>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

class MQTTClient {
public:
    bool running = true;
    MQTTClient(const std::string& clientId = "", void* context=nullptr);
    ~MQTTClient();

    bool connect(const std::string& host, int port, int keepalive);
    void disconnect();
    bool publish(const std::string& topic, const json& message);
    bool subscribe(const std::string& topic);
    void set_message_callback(void (*callback)(struct mosquitto *, void *, const struct mosquitto_message *)); // allow the callback to be defined within the template, so that its easier to process the payload. 
    void stop_listener();

private:
    static void on_connect(struct mosquitto* mosq, void* obj, int rc);
    static void on_log(struct mosquitto* mosq, void* obj, int level, const char* str);

    struct mosquitto* mosq_;
};

#endif // MQTT_CLIENT_H
