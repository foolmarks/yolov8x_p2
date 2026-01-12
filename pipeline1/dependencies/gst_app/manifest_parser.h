#ifndef MANIFEST_PARSER_H
#define MANIFEST_PARSER_H
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <set>
using json = nlohmann::json;

struct PluginInfo {
    std::string name;
    std::string gid;
    std::string config_name;
};

// TODO: This class needs to be structured better. Implement it as a generic manifest json parser.
class ManifestParser {
    public:
        //data members, these are exposed to the gst-app
        std::map<std::string, std::string> config_plugin_map;

        ManifestParser();
        ManifestParser(const std::string& json_path);
        ~ManifestParser();
        std::string get_pipeline_name();
        /// @brief returns path where configurations are stored on a board
        std::string get_config_instalation_prefix();
        std::vector<json> get_plugin_list();
        std::vector<PluginInfo> get_plugins_info();
        std::string parse_json_name(std::string path);
    private:
        std::string json_path;
        json manifest_json;
        std::vector<json> plugins_vec;
        std::vector<PluginInfo> plugins_info;

        /// @brief List of SiMa plugins that we are not capturing KPI info for
        std::set<std::string> blacklist{"simaaisrc", "simaaiencoder",
                                        "simaaidecoder", "simaaifilter"};


        //private member functions
        void read_json();
        void parse_plugins();

        bool starts_with(std::string main_string, std::string query);
        bool in_blacklist(std::string plugin_gid);
        bool is_key_in_json(const std::string& key, const nlohmann::json& json);
        bool is_non_empty_config(const nlohmann::json& plugin_json);
};

#endif // MANIFEST_PARSER_H