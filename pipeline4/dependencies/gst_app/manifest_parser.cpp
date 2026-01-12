#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <manifest_parser.h>
#include <string_utils.h>

ManifestParser::ManifestParser(const std::string& json_path) {
    this->json_path = json_path;

    //read json
    read_json();

    //parse plugins
    parse_plugins();
}

ManifestParser::ManifestParser(){}
ManifestParser::~ManifestParser(){}

void ManifestParser::read_json() {
    std::ifstream ifs(json_path);
    manifest_json = json::parse(ifs);
}

std::string ManifestParser::get_pipeline_name() {
    return manifest_json["applications"][0]["pipelines"][0]["name"];
}

std::string ManifestParser::parse_json_name(std::string path){
    std::size_t lastSlash = path.find_last_of("/\\");
    std::string filename = path.substr(lastSlash + 1);
    return filename;
}

bool ManifestParser::in_blacklist(std::string plugin_gid){
    if ((blacklist.find(plugin_gid) != blacklist.end())){
        return true;
    }

    return false;
}

bool ManifestParser::is_key_in_json(const std::string& key, const nlohmann::json& json) {
    return json.find(key) != json.end();
}

std::vector<PluginInfo> ManifestParser::get_plugins_info() {
    return plugins_info;
}

std::string ManifestParser::get_config_instalation_prefix() {
    return manifest_json["applications"][0]["configuration"]
                        ["installationPrefixes"]["configurations"];
}

bool ManifestParser::is_non_empty_config(const nlohmann::json& plugin_json) {
    if (plugin_json.is_array() &&
        !plugin_json.empty() &&
        plugin_json[0].is_string()) {
            return !plugin_json[0].get<std::string>().empty();
    }
    return false;
}


void ManifestParser::parse_plugins() {
    /*
    1. Parse out the plugins list
    2. For each plugin, check if "name" parameter starts with simaai - use regex
    */

    //1. Parse out plugins list
    plugins_vec = manifest_json["applications"][0]["pipelines"][0]["plugins"].get<std::vector<json>>();
    PluginInfo tmp_plugin_info;

    //2. Step into each plugin and create pluginId -> config mapping
    for (auto & plugin_json: plugins_vec) {

        if (is_key_in_json("name" , plugin_json)){
            tmp_plugin_info.name = plugin_json["name"].get<std::string>();
        } else {
            tmp_plugin_info.name = "not found";
        }

        if (is_key_in_json("pluginGid", plugin_json)){
            tmp_plugin_info.gid = plugin_json["pluginGid"].get<std::string>();
        } else {
            tmp_plugin_info.gid = "not found";
        }

        //only include in the map if plugin_name starts with simaai and not in the blacklist
        if(utils::StringUtils::starts_with(tmp_plugin_info.name, "simaai") && !in_blacklist(tmp_plugin_info.gid)){
            // check if plugin config exists.
            // for SDK pipelines: resources -> configs -> 0
            // for edgematic pipelines: configParams -> advanced -> config -> value

            if(is_key_in_json("resources", plugin_json) && is_key_in_json("configs", plugin_json["resources"]) && is_non_empty_config(plugin_json["resources"]["configs"])){
                // sdk json
                tmp_plugin_info.config_name = parse_json_name(plugin_json["resources"]["configs"][0].get<std::string>());
            } else if (is_key_in_json("configParams", plugin_json) && is_key_in_json("advanced", plugin_json["configParams"]) && is_key_in_json("config", plugin_json["configParams"]["advanced"])) {
                // edgematic json
                tmp_plugin_info.config_name  = plugin_json["configParams"]["advanced"]["config"]["value"];
            } else {
                tmp_plugin_info.config_name = "not found";
                continue;
            }

            config_plugin_map[tmp_plugin_info.config_name] = tmp_plugin_info.name;

            plugins_info.push_back(tmp_plugin_info);
        }
    }

    //TODO: Clean later
    for(auto &elem  : config_plugin_map) {
        std::cout << elem.first << " -> " << elem.second << std::endl;
    }
}
