#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <gst/gst.h>
#include <pipeline.h>
#include <manifest_parser.h>
#include <thread>
#include <simaai/simaailog.h>
#include <set>



GstElement* buildPipeline(std::string gst_string) {
    GstElement *pipeline=nullptr;
    GError *error = nullptr;
    pipeline = gst_parse_launch(gst_string.c_str(), &error);
    if(error) {
        std::cerr << "Error: " << error->message << std::endl;
        g_clear_error(&error);
        return pipeline;
    }
    std::cout << "Pipeline created successfully." << std::endl;
    return pipeline;
}

void parse_pipeline(GstElement* pipeline){
    GstIterator *it = gst_bin_iterate_elements(GST_BIN(pipeline));
    gboolean done = FALSE;
    int transmit_count = 0;
    int overlay_count = 0;
    int source_count = 0;

    std::set<std::string> source_plugins = {"GstFileSrc", "GstSimaaiSrc", "GstRTSPSrc", "GstUDPSrc"};

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
                        //config_name = parser.parse_json_name(config_path);

                        // if(parser.config_plugin_map.find(config_name) != parser.config_plugin_map.end()) {
                        //     plugin_map[element_name] = parser.config_plugin_map[config_name];
                        // }

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
                std:: cout << "Element: " << element_name << " classname: " << class_name << std::endl;
                if (class_name == "GstSimaaiOverlay2") {
                    overlay_count++;
                    //plugin_map[element_name] = element_name;
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
    int total_plugins = (transmit_count - overlay_count) + 1;
    // Print the counts
    std::cout << "Number of plugins with 'transmit' property: " << transmit_count << std::endl;
    std::cout << "Number of plugins with base class 'GstSimaaiOverlay2': " << overlay_count << std::endl;
    std::cout << "Number of source plugins: " << source_count << std::endl;
    std::cout << "Number of Total plugins: " << total_plugins << std::endl;

}

std::string remove_single_quotes(std::string input){
    std::string result;
    for (char ch : input) {
        if (ch != '\'') {
            result += ch;
        }
    }
    return result;
}

int main(int argc, char *argv[]){
    std::string gst_string;
    gst_init(nullptr, nullptr);
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " --gst-string=<gst string> " << std::endl;
        return -1;
    }

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--gst-string=", 13) == 0) {
            gst_string = argv[i] + 13;
        } 
        else if (std::strcmp(argv[i], "--gst-string") == 0 && i + 1 < argc) {
            gst_string = argv[++i];
        }
        else {
            std::cout << "Args have not been passed properly, please check usage." << std::endl;
        }
    }

    std::cout << "gst-string: " << gst_string << std::endl;
    gst_string = remove_single_quotes(gst_string);
    std::cout << "\n\n\n\n After removing double quotes: " << gst_string;
    parse_pipeline(buildPipeline(gst_string));
}