#include <iostream>
#include <mqtt_client.h>
#include <nlohmann/json.hpp>
#include <string>
#include <gst/gst.h>
#include <pipeline.h>
#include <manifest_parser.h>
#include <thread>
#include <simaai/simaailog.h>
#include <set>
#include <regex>
#include <string_utils.h>
#include <sys/types.h>
#include <unistd.h>
#include <sstream>
#include <glib.h>
#include <json-glib/json-glib.h>
#include <filesystem>

#include <live_trace_reader_api.h>

Pipeline::Pipeline(const std::string& manifest_json_path,
                const std::string& gst_string,
                const std::vector<std::string>& rtsp_urls_vec,
                const std::vector<std::string>& host_ips_vec,
                const std::vector<std::string>& host_ports_vec,
                json &gst_replacement_json,
                bool enable_lttng_param,
                bool swappable) {
    gst_init(nullptr, nullptr);
    this->manifest_json_path = manifest_json_path;
    this->gst_string = utils::StringUtils::remove_single_quotes(gst_string);

    //check if file exists
    if (!std::filesystem::exists(this->manifest_json_path)) {
        std::cerr << "Json path is invalid" << std::endl;
        simaailog(SIMAAILOG_ERR,"Input Json Path is Invalid");
        exit(-1);
    }

    //init json parser
    parser =  ManifestParser(this->manifest_json_path);
    pipeline_name = parser.get_pipeline_name();
    gstAppPid = getpid();

    std::stringstream ss;
    ss << "Starting GstApp for PipelineId: " << pipeline_name << ", Pid: " << gstAppPid;
    std::cout << ss.str() << std::endl;
    simaailog(SIMAAILOG_INFO, ss.str().c_str());

    //init simaailogger
    simaailog_init(pipeline_name.c_str());

    //init members
    this->rtsp_urls = rtsp_urls_vec;
    this->host_ips = host_ips_vec;
    this->host_ports = host_ports_vec;
    this->gst_replacement_json = gst_replacement_json;
    this->enable_lttng = enable_lttng_param;
    this->swappable = swappable;

    this->client = nullptr;
    this->lttng_session = nullptr;

    this->session_url = utils::LttngSession::create_session_url(this->pipeline_name.c_str());

    //process gst_string. This will modify the gst_string class member
    process_gst_string();
}

Pipeline::~Pipeline() {
    if (client) {
        delete client;
    }

    if (lttng_session) {
        delete lttng_session;
    }

    if (bus) {
        gst_object_unref(bus);
    }

    if (pipeline) {
        std::cout << "Destructor called." << std::endl;
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }

    if (this->swappable) {
        g_free((gchar *) this->gst_string_swappable);
        g_free((gchar *) this->artifacts_path);
        g_free((gchar *) this->render_rule);
    }
}

bool Pipeline::initMQTTClient() {
    std::stringstream clientId;
    clientId << "gst_app_client" << "_" << gstAppPid;
    std::cout << "Initializing MQTT Client with ClientId: " << clientId.str() << std::endl;

    client = new MQTTClient(clientId.str(), this);
    if (!client->connect(SIMAAI_MQTT_HOST, SIMAAI_MQTT_PORT, SIMAAI_MQTT_KEEPALIVE)) {
        std::cerr << "MQTT Client couldn't connect" << std::endl;
        simaailog(SIMAAILOG_ERR, "PipelineId: [%s] MQTT client init failed", pipeline_name.c_str());
        return false;
    }

    return true;
}

gboolean Pipeline::buildPipeline() {
    pipeline =
        gst_parse_launch(this->swappable ? this->gst_string_swappable : gst_string.c_str(), &error);
    if (!pipeline) {
        std::cerr << "Error: " << error->message << std::endl;
        simaailog(SIMAAILOG_ERR, "[%s] Error building pipeline: %s", pipeline_name.c_str(), error->message);
        g_clear_error(&error);
        return FALSE;
    }
    simaailog(SIMAAILOG_INFO, "PipelineId: [%s] Pipeline created successfully", pipeline_name.c_str());
    std::cout << "Pipeline created successfully." << std::endl;

    if (this->swappable) {
        this->swap_num = 0;

        if (!swap_model()) {
            std::cerr << "Failed to perform substitution." << std::endl;
            simaailog(SIMAAILOG_ERR, "[%s] Failed to perform substitution.", pipeline_name.c_str());
            return FALSE;
        }
    }

    return TRUE;
}

void Pipeline::initBus() {
    bus = gst_element_get_bus(pipeline);
    simaailog(SIMAAILOG_INFO, "GStreamer Bus Initialized.");
}

void Pipeline::init_signals() {
    signals[SIGNAL_PIPELINE_STOP] = g_signal_newv (
        "pipeline-stop",                                        // signal name
        G_TYPE_FROM_CLASS (GST_PIPELINE_GET_CLASS(pipeline)),   // class owner of the signal
        (GSignalFlags)(G_SIGNAL_RUN_FIRST | G_SIGNAL_ACTION),   // signal flags
        NULL,                                                   // class closure
        NULL,                                                   // accumulator
        NULL,                                                   // accu_data
        g_cclosure_marshal_VOID__VOID,                          // marshal function
        G_TYPE_NONE,                                            // return type
        0,                                                      // count of parameters
        NULL                                                    // list of parameters
    );
}

//TODO: Look into how this can be better implemented. Pass a callback for each item?
void Pipeline::parse_pipeline() {
    GstIterator *it = gst_bin_iterate_elements(GST_BIN(pipeline));
    gboolean done = FALSE;
    int transmit_count = 0;
    int overlay_count = 0;
    int source_count = 0;

    std::set<std::string> source_plugins = {"GstFileSrc", "GstSimaaiSrc", 
                                            "GstRTSPSrc", "GstUDPSrc"};

    while (!done) {
        GValue item = G_VALUE_INIT;
        switch (gst_iterator_next(it, &item)) {
            case GST_ITERATOR_OK: {
                GstElement *element = GST_ELEMENT(g_value_get_object(&item));
                const gchar *element_name = gst_element_get_name(element);

                std::cout << "Element Name: " << element_name << std::endl;
                gchar* config_path = NULL;
                std::string config_name;

                // Check if the "config" property exists
                GParamSpec *config_pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), "config");
                if (config_pspec && G_IS_PARAM_SPEC_STRING(config_pspec)) {
                    g_object_get(element, "config", &config_path, NULL);

                    if (config_path) {
                        g_print("Element: %s, config: %s\n", element_name, config_path);
                        config_name = parser.parse_json_name(config_path);

                        if(parser.config_plugin_map.find(config_name) != parser.config_plugin_map.end()) {
                            plugin_map[element_name] = parser.config_plugin_map[config_name];
                        }

                        g_free(config_path);
                    }
                }

                // Check for "transmit" property
                GParamSpec *transmit_pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), "transmit");
                if (transmit_pspec && G_IS_PARAM_SPEC_BOOLEAN(transmit_pspec)) {
                    transmit_count++;
                }

                // Check if base class is GstSimaaiOverlay2
                std::string class_name = G_OBJECT_TYPE_NAME(element);
                if (class_name == "GstSimaaiOverlay2") {
                    overlay_count++;
                    plugin_map[element_name] = element_name;
                }

                // Check for source plugins by element name
                if (source_plugins.find(class_name) != source_plugins.end()) {
                    source_count++;
                }

                g_value_reset(&item);
                break;
            }
            case GST_ITERATOR_RESYNC:
                gst_iterator_resync(it);
                break;
            case GST_ITERATOR_ERROR:
            case GST_ITERATOR_DONE:
                done = TRUE;
                break;
        }
        g_value_unset(&item);
    }

    int total_plugins = overlay_count ? (transmit_count - overlay_count) + 1 : transmit_count;
    // Print the counts
    std::cout << "Number of plugins with 'transmit' property: " << transmit_count << std::endl;
    std::cout << "Number of plugins with base class 'GstSimaaiOverlay2': " << overlay_count << std::endl;
    std::cout << "Number of source plugins: " << source_count << std::endl;
    std::cout << "Number of Total plugins: " << total_plugins << std::endl;

    transmit_plugin_count = transmit_count;
    num_transmit_plugins = total_plugins;
    std::cout << "Total plugins per frame: " << num_transmit_plugins << std::endl;

    //TODO: Clean later
    std::cout << "FINAL MAPPING #############" << std::endl;
    for(auto &elem  : plugin_map) {
        std::cout << elem.first << " -> " << elem.second << std::endl;
    }

    gst_iterator_free(it);
}

void Pipeline::start_KPI_collecting() {
    if (this->enable_lttng == false) {
        return;
    }

    stop_KPI_collecting();

    pid_t pipeline_pid = getpid();
    this->lttng_session = new utils::LttngSession(this->pipeline_name.c_str(), pipeline_pid);
    if (this->lttng_session == nullptr) {
        std::cerr << "Error creating the LTTNG session, exiting..." << std::endl;
        simaailog(SIMAAILOG_ERR,"PipelineID: [%s] Error creating the LTTNG session, exiting...", pipeline_name.c_str());
        return;
    }

    live_trace_reader_init_data_t ltr_args;
    ltr_args.pid = pipeline_pid;
    ltr_args.pipeline_id = this->pipeline_name.c_str();
    ltr_args.session_url = session_url.c_str();
    ltr_args.plugins_count = this->transmit_plugin_count;

    this->ltr = std::async(std::launch::async, live_trace_reader_run, ltr_args);
}

void Pipeline::stop_KPI_collecting() {
    if (this->enable_lttng == false) {
        return;
    }

    live_trace_reader_set_running_status(LIVE_TRACE_READER_RUNNING_STATUS_STOP);
    if (ltr.valid()) {
        ltr.wait();
        std::cout << "LTR finished with code " << ltr.get() << std::endl;
    }

    if (this->lttng_session) {
        delete this->lttng_session;
        this->lttng_session = nullptr;
    }
}

void Pipeline::pipeline_driver(){
    //build pipeline from gst_string
    if(!buildPipeline()){
        std::cerr << "Error building pipeline, exiting..." << std::endl;
        simaailog(SIMAAILOG_ERR,"PipelineID: [%s] Error building pipeline", pipeline_name.c_str());
        return;
    }

    parse_pipeline();

    if (this->enable_lttng || this->swappable) {
        //Connect to the Mqtt broker
        if(!initMQTTClient()) {
            std::cerr << "Error connecting to the MQTT client, exiting..." << std::endl;
            simaailog(SIMAAILOG_ERR,"PipelineID: [%s] Error connecting to MQTT broker, exiting.", pipeline_name.c_str());
            return;
        }

        //register calback function
        client->set_message_callback(mqtt_callback);
    }

    init_signals();

    //start playing the pipeline
    start_pipeline();

    //initialize GstBus
    initBus();

    //loop to drive the code
    while(!terminate){
        msg = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_APPLICATION));
        if (msg==NULL) continue;
        switch(GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR:
                gst_message_parse_error(msg, &error, &debug_info);
                std::cerr << "Error received from element " << GST_OBJECT_NAME(msg->src) << ": " << error->message << std::endl;
                std::cerr << "Debugging information: " << (debug_info ? debug_info : "none") << std::endl;
                simaailog(SIMAAILOG_ERR, "PipelineId: [%s] Error received from element: %s, debug info: %s", pipeline_name.c_str(), GST_OBJECT_NAME(msg->src), (debug_info ? debug_info : "none"));
                g_clear_error(&error);
                g_free(debug_info);
                terminate = TRUE;
                stop_KPI_collecting();
                break;

            case GST_MESSAGE_EOS: {
                simaailog(SIMAAILOG_INFO, "PipelineId: [%s] End of Stream reached.", pipeline_name.c_str());
                terminate = TRUE;
                stop_KPI_collecting();
                break;
            }

            case GST_MESSAGE_APPLICATION: {
                // FIXME: Currently ignoring this
                break;
            }

            default:
                simaailog(SIMAAILOG_INFO, "PipelineId: [%s] Unexpeted message received.", pipeline_name.c_str());
                std::cerr << "Unexpected message received." << std::endl;
                break;
        }
        gst_message_unref(msg);
    }
    g_signal_emit (pipeline, signals[SIGNAL_PIPELINE_STOP], 0);

    return;
}

void Pipeline::terminate_pipeline(){
    std::cout << "Terminating pipeline ..." << std::endl;
    terminate = TRUE;
    stop_KPI_collecting();
}

void Pipeline::handle_callback( const struct mosquitto_message *message) {
    std::string topic(message->topic);
    json jsonMessage;
    json response;
    std::string payload(static_cast <char *> (message->payload), message->payloadlen);
    const gchar *status = "success";

    try{
        jsonMessage = json::parse(payload);
    }
    catch (...) {
        std::cerr << "Error Parsing incoming payload: " << payload << std::endl;
        simaailog(SIMAAILOG_ERR,"PipelineID: [%s] Error parsing incoming MQTT message", pipeline_name.c_str());
        return;
    }

    // TODO: implement a message handler class. See if pipeline handler can write json interfaces in a way that different messages can be segragated easily.
    std::cout << "Received " << jsonMessage.dump(4) << "from [ " << topic << " ]" << std::endl;
    std::cout << "pipeline id: " << pipeline_name << std::endl;
    std::cout << "GstApp ProcessId: " << gstAppPid << std::endl;
    if(jsonMessage.find("pid") == jsonMessage.end())
    {
        std::cerr << "'pid' parameter not found in received json message: " << jsonMessage.dump() << std::endl;
        simaailog(SIMAAILOG_ERR,"'pid' parameter not found in received json message. Returning...", jsonMessage.dump().c_str());
        return;
    }

    std::string jsonMsgPipelineId = jsonMessage.at("pipeline_id").get<std::string>();
    pid_t jsonMsgPid = jsonMessage.at("pid").get<pid_t>();
    std::string jsonMsgCmd = jsonMessage.at("command").get<std::string>();
    std::cout << "JSON Message PipelineId: " << jsonMsgPipelineId << std::endl;
    std::cout << "JSON Message ProcessId: " << jsonMsgPid << std::endl;
    std::cout << "JSON Message Command: " << jsonMsgCmd << std::endl;

    if(jsonMessage["pipeline_id"] == pipeline_name && jsonMsgPid == gstAppPid) {
        gboolean property_value = jsonMessage["command"] == "start-kpis";

        if (jsonMessage["command"] == "start-kpis") {
            start_KPI_collecting();
            property_value = TRUE;
        } else if (jsonMessage["command"] == "stop-kpis") {
            stop_KPI_collecting();
            property_value = FALSE;
        } else if (jsonMessage["command"] == "swap-model") {
            this->artifacts_path =
                strdup(jsonMessage.at("artifacts_path").get<std::string>().c_str());
            this->render_rule = strdup(jsonMessage.at("render_rule").get<std::string>().c_str());
            this->swap_num++;

            if (!swap_model()) {
                std::cerr << "Failed to swap models." << std::endl;
                simaailog(SIMAAILOG_ERR,"PipelineID: [%s] Failed to swap models",
                    pipeline_name.c_str());
                status = "fail";
            }
        } else {
            // TODO: ERROR
            simaailog(SIMAAILOG_ERR,"Received unknown command: %s", jsonMessage["command"]);

        }

        if (jsonMessage["command"] != "swap-model") {
            set_transmit_property(property_value);
            std::cout << "Set transmit parameter value to '" << property_value << "'" << std::endl;
        }

        // Make response
        response["command"] = jsonMessage["command"];
        response["pipeline_id"] = pipeline_name;
        response["pid"] = gstAppPid;
        response["status"] = status;

        if (!client->publish(SIMAAI_MQTT_KPI_RES_TOPIC, response)) {
            std::cerr << "Something went wrong while pushing the message." << std::endl;
            simaailog(SIMAAILOG_ERR,"PipelineID: [%s] Error sending response to pipeline handler", pipeline_name.c_str());
        }
    } else {
        simaailog(SIMAAILOG_ERR,"Mismatch in received Json Message request PipelineId: %s or Pid: %d with GstApp's PipelineId: %s or Pid: %d", jsonMsgPipelineId.c_str(), jsonMsgPid, pipeline_name.c_str(), gstAppPid);
        simaailog(SIMAAILOG_ERR,"Not processing KPI request command: %s", jsonMsgCmd.c_str());
    }
}

void Pipeline::mqtt_callback(struct mosquitto *mosq, void * obj,  const struct mosquitto_message *message) {
    Pipeline *self = static_cast<Pipeline *>(obj);
    if (self){
        self->handle_callback(message);
    }
}

void Pipeline::set_property(const char *name, gboolean value) {
    if (!GST_IS_PIPELINE(pipeline)) {
        std::cerr << "The provided GstElement is not a pipeline." << std::endl;
        return;
    }

    GstIterator *iter = gst_bin_iterate_elements(GST_BIN(pipeline));
    GValue item = G_VALUE_INIT;
    gboolean done = FALSE;
    GError *err = NULL;

    while (!done) {
        switch (gst_iterator_next(iter, &item)) {
            case GST_ITERATOR_OK: {
                GstElement *element = GST_ELEMENT(g_value_get_object(&item));
                GParamSpec *property = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name);

                if (property && G_IS_PARAM_SPEC_BOOLEAN(property)) {
                    g_object_set(element, name, value, NULL);
                    std::cout << "Set '" << name << "' property to " << (value ? "TRUE" : "FALSE") << " for element: " << GST_ELEMENT_NAME(element) << std::endl;
                }

                g_value_reset(&item);
                break;
            }
            case GST_ITERATOR_RESYNC:
                gst_iterator_resync(iter);
                break;
            case GST_ITERATOR_ERROR:
                if (err) {
                    std::cerr << "Iterator error: " << error->message << std::endl;
                    g_clear_error(&err);
                }
                done = TRUE;
                break;
            case GST_ITERATOR_DONE:
                done = TRUE;
                break;
            default:
                done = TRUE;
                break;
        }
    }

    g_value_unset(&item);
    gst_iterator_free(iter);
}

void Pipeline::set_transmit_property(gboolean transmit_value) {
    set_property("transmit", transmit_value);
}

void Pipeline::replace_json_tags() {
    if (gst_replacement_json.size()) {
        // json strcture: 
        // {
        //     "TAG_1" : "replacement str 1",
        //     "TAG_2" : "reaplcement str 2",
        //     ...
        // }
        simaailog(SIMAAILOG_INFO, "Using the replacement json to repalce tags in gst-string");
        for (auto it = gst_replacement_json.begin(); it != gst_replacement_json.end(); it++){
            std::string tag_to_replace = it.key();
            std::string replacement = it.value();
            simaailog(SIMAAILOG_INFO, "Replacing tag: %s", tag_to_replace.c_str());
            utils::StringUtils::string_replace(gst_string, tag_to_replace, replacement);
        }
        simaailog(SIMAAILOG_INFO, "GST String After replacement: %s", gst_string.c_str());
    }
    return;
}

void Pipeline::replace_vector_tags() {
    std::string rtsp_re = "<(RTSP_SRC|RTSP_SOURCE|RTSP_URL)[^>]*>";
    std::string ip_re = "<(HOST_IP|SINK_IP)[^>]*>";
    std::string port_re = "<(HOST_PORT|SINK_PORT)[^>]*>";

    if (rtsp_urls.size()) utils::StringUtils::regex_replace_all_instances(rtsp_re, rtsp_urls, gst_string);
    if (host_ips.size()) utils::StringUtils::regex_replace_all_instances(ip_re, host_ips, gst_string);
    if (host_ports.size()) utils::StringUtils::regex_replace_all_instances(port_re, host_ports, gst_string);
    return;
}

void Pipeline::replace_configs() {
    std::string config_root_path = this->parser.get_config_instalation_prefix();

    // divide gst string to separate plugins strings
    std::vector<std::string> gst_plugins = 
                                    utils::StringUtils::split(gst_string, '!');

    // search for plugins described in manifest.json
    auto config = this->parser.get_plugins_info();
    std::cout << "Number of configs in manifest:" << config.size() << std::endl;
    for (auto & plugin_info : config) {
        std::cout << "========================================"
                  << "========================================" << std::endl; 
        std::cout << "Check plugin " << plugin_info.name << std::endl;
        if (plugin_info.config_name == "not found") {
            std::cout << "No config for this plugin" << std::endl;
            continue;
        }
        for (auto & plugin_str : gst_plugins) {
            // search for name property
            // regex to find name property with and without quotation marks.
            std::string name_property = 
                    utils::StringUtils::find_string_by_regex( plugin_str, 
                                      "\\sname *= *\"?" + plugin_info.name + "\"?");

            // search for config property
            std::string config_property = 
                    utils::StringUtils::find_string_by_regex( plugin_str, 
                                                              "\\sconfig *= *");

            // if name property not found go to next line
            if (name_property.empty())
                continue;
            
            // check if there is no config property
            if (config_property.empty()) 
            {
                std::cout << "Found plugin. Adding config:" << std::endl;
                std::cout << "Old gst plugin line: " << plugin_str << std::endl;
                std::size_t position = plugin_str.find(name_property);
                if (position != std::string::npos) 
                {
                    // compose config path
                    std::filesystem::path config_file_path (config_root_path);
                    config_file_path /= plugin_info.config_name;
                    // compose porperty
                    config_property = " config=";
                    config_property += config_file_path.string();
                    // insert property to plugin (right after name property)
                    position += name_property.length();
                    plugin_str.insert(position, config_property);
                    std::cout << "New gst plugin line: " << plugin_str << std::endl;
                }
            }
            else {
                std::cout << "Plugin already have config property. " 
                          << "Ignoring path composed from manifest." << std::endl;
            }
            break;
        }
    }

    // compose updated gst string
    std::string new_gst_string = "";
    for (int i = 0; i < gst_plugins.size(); i++) {
        new_gst_string += gst_plugins[i];
        // do not add ! after last plugin
        if (i != gst_plugins.size() - 1)
            new_gst_string += " ! ";
    }
    this->gst_string = new_gst_string;
}

void Pipeline::get_artifacts_path()
{
    this->artifacts_path = g_path_get_dirname(this->manifest_json_path.c_str());
}

void Pipeline::get_render_rule()
{
    const gchar *render_rule_pos = strchr(strstr(this->gst_string.c_str(), "render-info"), ',') + 1;

    this->render_rule =
        g_strndup(render_rule_pos, (strstr(render_rule_pos, "::") - render_rule_pos));
}

static gboolean correct_boxdecode_config(const gchar *boxdecode_config, const gchar *boxdecode_name)
{
    JsonParser *parser = json_parser_new();
    gboolean retval = FALSE;
    JsonNode *root = NULL;
    JsonObject *root_obj = NULL;
    JsonGenerator *generator = NULL;

    if (!json_parser_load_from_file(parser, boxdecode_config, NULL)) {
        g_printerr("\nError loading a JSON stream.\n");
        goto correct_boxdecode_exit1;
    }

    root = json_parser_get_root(parser);
    if (!JSON_NODE_HOLDS_OBJECT(root)) {
        g_printerr("\nError retrieving a top level node.\n");
        goto correct_boxdecode_exit1;
    }
    root_obj = json_node_get_object(root);

    json_object_set_string_member(root_obj, "node_name", boxdecode_name);

    generator = json_generator_new();

    json_generator_set_root(generator, root);

    if (!json_generator_to_file(generator, boxdecode_config, NULL)) {
        g_printerr("\nFailed to write changes to the MLA config.\n");
        goto correct_boxdecode_exit2;
    }

    retval = TRUE;

correct_boxdecode_exit2:
    g_object_unref(generator);
correct_boxdecode_exit1:
    g_object_unref(parser);
    return retval;
}

static gboolean correct_mla_config(const gchar *process_mla_config, const gchar *process_mla_model)
{
    JsonParser *parser = json_parser_new();
    gboolean retval = FALSE;
    JsonNode *root = NULL;
    JsonObject *root_obj = NULL;
    JsonObject *simaai_params = NULL;
    JsonGenerator *generator = NULL;

    if (!json_parser_load_from_file(parser, process_mla_config, NULL)) {
        g_printerr("\nError loading a JSON stream.\n");
        goto correct_mla_exit1;
    }

    root = json_parser_get_root(parser);
    if (!JSON_NODE_HOLDS_OBJECT(root)) {
        g_printerr("\nError retrieving a top level node.\n");
        goto correct_mla_exit1;
    }
    root_obj = json_node_get_object(root);

    simaai_params = json_object_get_object_member(root_obj, "simaai__params");
    if (!simaai_params) {
        g_printerr("\nFailed to retrieve \"simaai__params\" from the MLA config.\n");
        goto correct_mla_exit1;
    }

    json_object_set_string_member(simaai_params, "model_path", process_mla_model);

    generator = json_generator_new();

    json_generator_set_root(generator, root);

    if (!json_generator_to_file(generator, process_mla_config, NULL)) {
        g_printerr("\nFailed to write changes to the MLA config.\n");
        goto correct_mla_exit2;
    }

    retval = TRUE;

correct_mla_exit2:
    g_object_unref(generator);
correct_mla_exit1:
    g_object_unref(parser);
    return retval;
}

gboolean Pipeline::swap_model()
{
    namespace fs = std::filesystem;
    
    GstPad *sinkpad = NULL, *srcpad = NULL;
    guint32 swap_num_local = this->swap_num + 1;
    gboolean retval = FALSE, playing = FALSE;
    GstElement *preprocess = NULL, *process_mla = NULL, *boxdecode = NULL, *overlay = NULL;
    GstPad *model_out_sel_pad = NULL, *model_in_sel_pad = NULL, *source_out_sel_pad = NULL;

    gchar *preprocess_name = g_strdup_printf("simaai_preprocess_%u", swap_num_local);
    gchar *process_mla_name = g_strdup_printf("simaai_process_mla_%u", swap_num_local);
    gchar *boxdecode_name = g_strdup_printf("simaai_boxdecode_%u", swap_num_local);
    gchar *overlay_name = g_strdup_printf("overlay_%u", swap_num_local);

    gchar *preprocess_config = g_strdup_printf("%s/etc/gen_preproc.json", this->artifacts_path);
    gchar *process_mla_config = g_strdup_printf("%s/etc/mla.json", this->artifacts_path);
    gchar *process_mla_model = g_strdup_printf("%s/share/processmla/model.elf",
        this->artifacts_path);
    gchar *boxdecode_config = g_strdup_printf("%s/etc/boxdecode.json", this->artifacts_path);
    gchar *overlay_render_info = g_strdup_printf("input::decoder,%s::%s", this->render_rule,
        boxdecode_name);
    gchar *overlay_labels = g_strdup_printf("%s/share/overlay/labels.txt", std::filesystem::path(this->manifest_json_path).parent_path().string().c_str());

    std::cout << "Labels path: " << overlay_labels << std::endl;

    simaailog(SIMAAILOG_ERR,"Labels path is: %s", std::filesystem::path(this->manifest_json_path).parent_path().string().c_str());

    gchar *src_pad_name = g_strdup_printf("src_%u", swap_num_local);
    gchar *sink_pad_name = g_strdup_printf("sink_%u", swap_num_local);

    GstElement *model_out_sel = gst_bin_get_by_name(GST_BIN(this->pipeline), "model_out_sel");
    GstElement *model_in_sel = gst_bin_get_by_name(GST_BIN(this->pipeline), "model_in_sel");
    GstElement *source_out_sel = gst_bin_get_by_name(GST_BIN(this->pipeline), "source_out_sel");
    if (!model_out_sel || !model_in_sel || !source_out_sel) {
        g_printerr("\nFailed to get one or more required elements from the pipeline.\n");
        goto swap_model_exit1;
    }

    preprocess = gst_element_factory_make("simaaiprocesscvu", preprocess_name);
    process_mla = gst_element_factory_make("simaaiprocessmla", process_mla_name);
    boxdecode = gst_element_factory_make("simaaiboxdecode", boxdecode_name);
    overlay = gst_element_factory_make("simaai-overlay2", overlay_name);
    if (!preprocess || !process_mla || !boxdecode || !overlay) {
        g_printerr("\nFailed to create one or more new GstElement.\n");
        goto swap_model_exit2;
    }

    g_object_set(preprocess, "config", preprocess_config, "name", preprocess_name, NULL);
    g_object_set(process_mla, "config", process_mla_config, "name", process_mla_name, NULL);
    g_object_set(boxdecode, "config", boxdecode_config, "name", boxdecode_name, NULL);
    g_object_set(overlay, "render-info", overlay_render_info, "labels-file", overlay_labels, "name",
        overlay_name, NULL);

    if (!correct_mla_config(process_mla_config, process_mla_model)) {
        g_printerr("\nFailed to modify the MLA config.\n");
        goto swap_model_exit2;
    }

    if (!correct_boxdecode_config(boxdecode_config, boxdecode_name)) {
        g_printerr("\nFailed to modify the BoxDecode config.\n");
        goto swap_model_exit2;
    }

    gst_bin_add_many(GST_BIN(this->pipeline), preprocess, process_mla, boxdecode, overlay, NULL);

    if (!gst_element_link_many(preprocess, process_mla, boxdecode, NULL)) {
        g_printerr("\nFailed to link the new model processing chain.\n");
        goto swap_model_exit2;
    }

    srcpad = gst_element_get_static_pad(boxdecode, "src");
    sinkpad = gst_element_request_pad_simple(overlay, "sink_application_data");
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("\nFailed to link BoxDecode to Overlay.\n");
        goto swap_model_exit3;
    }
    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
    srcpad = NULL;
    sinkpad = NULL;

    model_out_sel_pad = gst_element_request_pad_simple(model_out_sel, src_pad_name);
    sinkpad = gst_element_request_pad_simple(preprocess, "sink_0");
    if (gst_pad_link(model_out_sel_pad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("\nFailed to link the model output selector to PreProcess.\n");
        goto swap_model_exit4;
    }
    gst_object_unref(sinkpad);
    sinkpad = NULL;

    srcpad = gst_element_get_static_pad(overlay, "src");
    model_in_sel_pad = gst_element_request_pad_simple(model_in_sel, sink_pad_name);
    if (gst_pad_link(srcpad, model_in_sel_pad) != GST_PAD_LINK_OK) {
        g_printerr("\nFailed to link Overlay to the model input selector.\n");
        goto swap_model_exit5;
    }
    gst_object_unref(srcpad);
    srcpad = NULL;

    source_out_sel_pad = gst_element_request_pad_simple(source_out_sel, src_pad_name);
    sinkpad = gst_element_request_pad_simple(overlay, "sink_in_img_src"); 
    if (gst_pad_link(source_out_sel_pad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("\nFailed to link the source output selector to Overlay.\n");
        goto swap_model_exit6;
    }
    gst_object_unref(sinkpad);
    sinkpad = NULL;

    gst_element_set_state(overlay, GST_STATE_PLAYING);
    gst_element_set_state(boxdecode, GST_STATE_PLAYING);
    gst_element_set_state(process_mla, GST_STATE_PLAYING);
    gst_element_set_state(preprocess, GST_STATE_PLAYING);

    g_object_set(model_in_sel, "active-pad", model_in_sel_pad, NULL);
    g_object_set(model_out_sel, "active-pad", model_out_sel_pad, NULL);
    g_object_set(source_out_sel, "active-pad", source_out_sel_pad, NULL);

    playing = retval = TRUE;

swap_model_exit6:
    if (source_out_sel_pad)
        gst_object_unref(source_out_sel_pad);

swap_model_exit5:
    if (model_in_sel_pad)
        gst_object_unref(model_in_sel_pad);

swap_model_exit4:
    if (model_out_sel_pad)
        gst_object_unref(model_out_sel_pad);

swap_model_exit3:
    if (sinkpad)
        gst_object_unref(sinkpad);
    if (srcpad)
        gst_object_unref(srcpad);

swap_model_exit2:
    if (!playing) {
        if (overlay)
            gst_object_unref(overlay);
        if (boxdecode)
            gst_object_unref(boxdecode);
        if (process_mla)
            gst_object_unref(process_mla);
        if (preprocess)
            gst_object_unref(preprocess);
    }

swap_model_exit1:
    if (source_out_sel)
        gst_object_unref(source_out_sel);
    if (model_in_sel)
        gst_object_unref(model_in_sel);
    if (model_out_sel)
        gst_object_unref(model_out_sel);

    g_free(sink_pad_name);
    g_free(src_pad_name);
    g_free(overlay_labels);
    g_free(overlay_render_info);
    g_free(boxdecode_config);
    g_free(process_mla_model);
    g_free(process_mla_config);
    g_free(preprocess_config);
    g_free(overlay_name);
    g_free(boxdecode_name);
    g_free(process_mla_name);
    g_free(preprocess_name);

    return retval;
}

void Pipeline::process_gst_string_swappable()
{
    GString *gst_string_mod = g_string_new(this->gst_string.c_str());

    gsize len = strchr(strstr(gst_string_mod->str, "simaaidecoder"), '!') - gst_string_mod->str;
    gchar *sink_str = g_strndup(gst_string_mod->str, len);
    gchar *src_str = g_strdup(strstr(gst_string_mod->str, "simaaiencoder"));

    g_string_free(gst_string_mod, TRUE);

    const gchar *model_selector_str = "! tee name=source ! queue2 ! output-selector "
        "name=model_out_sel input-selector name=model_in_sel ! ";
    const gchar *source_selector_str = " model_out_sel.src_0 ! identity ! model_in_sel.sink_0 "
        "source. ! queue2 ! output-selector name=source_out_sel source_out_sel.src_0 ! identity";

    this->gst_string_swappable =
        g_strconcat(sink_str, model_selector_str, src_str, source_selector_str, NULL);

    std::cout << "\n\n GST string (swappable): \n" << this->gst_string_swappable << std::endl;
}

void Pipeline::process_gst_string() {
    if (this->swappable) {
        get_artifacts_path();
        get_render_rule();
        process_gst_string_swappable();
    }

    replace_json_tags();
    replace_vector_tags();
    replace_configs();
    std::cout << "\n\n Finall GST string: \n" << this->gst_string << std::endl;
}

void Pipeline::start_pipeline(){
    //TODO: Move this to a function, check status after playing and communate back to the PH
    gst_element_set_state(pipeline, GST_STATE_PLAYING); 
    terminate = FALSE;
}
