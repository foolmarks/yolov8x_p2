#include "mqtt_client.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <mosquitto.h>
#include <simaai/simaailog.h>
#include <sys/types.h>
#include <unistd.h>
#include <sstream>

MQTTClient::MQTTClient(const std::string& clientId, void* context) {
    mosquitto_lib_init();
    mosq_ = mosquitto_new(clientId.empty() ? nullptr : clientId.c_str(), true, context);
    if (!mosq_) {
        simaailog(SIMAAILOG_ERR, "Failed to create Mosquitto instance.");
        throw std::runtime_error("Failed to create mosquitto instance");
    }
    mosquitto_connect_callback_set(mosq_, on_connect);
    mosquitto_log_callback_set(mosq_, on_log);
}

MQTTClient::~MQTTClient() {
    simaailog(SIMAAILOG_INFO, "MQTT destructor called");
    std::cout << "MQTT Destructor called." <<std::endl;
    mosquitto_destroy(mosq_);
    mosquitto_lib_cleanup();
}

bool MQTTClient::connect(const std::string& host, int port, int keepalive) {
    if (mosquitto_connect(mosq_, host.c_str(), port, keepalive) != MOSQ_ERR_SUCCESS) {
        simaailog(SIMAAILOG_ERR, "Failed to connect to the broker.");
        std::cerr << "Unable to connect to MQTT broker" << std::endl;
        return false;
    }
    mosquitto_loop_start(mosq_);
    return true;
}

void MQTTClient::disconnect() {
    mosquitto_disconnect(mosq_);
    mosquitto_loop_stop(mosq_, true);
}

// Unusable
bool MQTTClient::publish(const std::string& topic, const json& message) {
    std::string payload = message.dump();
    int ret = mosquitto_publish(mosq_, nullptr, topic.c_str(), payload.size(), payload.c_str(), 0, false);
    if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to publish message. Error code: " << ret << std::endl;
        return false;
    }
    return true;
}

void MQTTClient::on_connect(struct mosquitto* mosq, void* obj, int rc) {
    if (rc == 0) {
        std::cout << "Connected to the broker!" << std::endl;
    } else {
        std::cerr << "Failed to connect, return code: " << rc << std::endl;
    }

    //subcribe to the incoming topic
    //Append current pid to the subcribed topic
    std::stringstream ssSubTopic;
    pid_t currentPid = getpid();
    ssSubTopic << SIMAAI_MQTT_KPI_REQ_TOPIC << "/" << currentPid;
    std::string sub_topic = ssSubTopic.str();

    simaailog(SIMAAILOG_INFO, "MQTTClient subscribing to Topic: %s", sub_topic.c_str());
    std::cout << "MQTTClient subscribing to Topic: " << sub_topic << std::endl;

    int mid;
    int qos = 0;
    int ret = mosquitto_subscribe(mosq, &mid, sub_topic.c_str(), qos);
    if (ret != MOSQ_ERR_SUCCESS) {
        simaailog(SIMAAILOG_ERR, "Failed to subscribe to %s",sub_topic.c_str() );
        std::cerr << "Error: Unable to subscribe to topic. Error: " << mosquitto_strerror(ret) << std::endl;
        std::runtime_error("Unable to subscribe");
    } else {
        simaailog(SIMAAILOG_INFO, "Subscribed to topic: %s", sub_topic.c_str());
        std::cout << "Subscribed to topic: " << sub_topic.c_str() << std::endl;
    }
}

void MQTTClient::set_message_callback(void (*callback)(struct mosquitto *, void *, const struct mosquitto_message *)){
    mosquitto_message_callback_set(mosq_, callback);
}


void MQTTClient::on_log(struct mosquitto* mosq, void* obj, int level, const char* str) {
    simaailog(SIMAAILOG_DEBUG, str);
}

void MQTTClient::stop_listener() {
    running = false;
}
