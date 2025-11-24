#ifndef PIPELINE_H
#define PIPELINE_H
#include <iostream>
#include <mqtt_client.h>
#include <nlohmann/json.hpp>
#include <string>
#include <gst/gst.h>
#include <filesystem>
#include <manifest_parser.h>

#include <lttng_session.h>
#include <future>

using json = nlohmann::json;

/// @brief pipeline class
class Pipeline {
    public: 
        Pipeline(const std::string& manifest_json_path, 
                const std::string& gst_string,
                const std::vector<std::string>& rtsp_urls_vec,
                const std::vector<std::string>& host_ips_vec,
                const std::vector<std::string>& host_ports_vec,
                json &gst_replacement_json,
                bool enable_lttng_param);
        ~Pipeline();
        /// @brief This function will orchestrate the loginc of building and running the pipeline
        void pipeline_driver();
        void terminate_pipeline();
    private:
        enum
        {
            SIGNAL_PIPELINE_STOP,
            LAST_SIGNAL
        };

        // data members
        GstElement *pipeline=nullptr;
        GstBus *bus=nullptr;
        GstMessage *msg = nullptr;
        gboolean terminate;
        GError *error = nullptr;
        gchar *debug_info = nullptr;
        std::string manifest_json_path, gst_string, pipeline_name;
        std::vector<std::string> rtsp_urls, host_ips, host_ports;
        json gst_replacement_json;
        MQTTClient *client;
        utils::LttngSession *lttng_session = nullptr;
        std::future<int> ltr;

        /// @brief Vector Map that stores KPI messages from plugins until all received. 
        std::map<std::string, std::vector<json>> frame_map;
        /// @brief Map to store 1-1 pluginId map from what is in the application 
        ///        json and what is assigned by gstreamer. 
        std::map<std::string, std::string> plugin_map; 
        ManifestParser parser;
        /// @brief number of plugins that transmit kpi information.
        int num_transmit_plugins;
        guint signals[LAST_SIGNAL] = { 0 };
        pid_t gstAppPid;
        int transmit_plugin_count = 0;
        bool enable_lttng;

        // private member functions
        void start_pipeline();
        /// @brief Performs all replacements in gst string
        void process_gst_string();
        /// @brief Replaces all `tags` to `replacements` described in `gst_replacement_json`
        void replace_json_tags();
        /// @brief Check instances of `rtsp_src`, `host_ips`, `host_ports`, 
        ///        replace as many as provided in cmdline
        void replace_vector_tags();
        /// @brief Check if all plugins described in JSON have name property and
        ///        sets config property if it is not provided in gst-string 
        /// @return true on success, false if gst-string is not valid
        void replace_configs();
        gboolean buildPipeline();
        void parse_pipeline(); // create a map from json pluginId to gstId
        void initBus();
        bool initMQTTClient();
        static void mqtt_callback(struct mosquitto *mosq, void * obj,  const struct mosquitto_message *message);
        void handle_callback( const struct mosquitto_message *message);
        //TODO: This should be generalized, take 2 params: property, and the value to be set.
        void set_property(const char *name, gboolean value);
        void set_transmit_property(gboolean transmit_value);
        void init_signals();

};

#endif // PIPELINE_H