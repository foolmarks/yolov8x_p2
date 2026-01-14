#ifndef EVXX_NLOHMANN_PARSER
#define EVXX_NLOHMANN_PARSER

#include <iostream>
#include <fstream>
#include <filesystem>
#include <type_traits>
#include <simaai/nlohmann/json.hpp>
#include <simaai/ev_cfg_helper.h>

#define INT32_PARAM 0
#define FLOAT_PARAM 1

bool parse_json_from_file(GstSimaaiProcesscvu * plugin,
                          const std::string config_file_path, 
                          nlohmann::json &json) 
{
  try {
    
    //parse file to nlohmann::json objects
    std::ifstream input_file(config_file_path.c_str());
    if (!input_file) 
      throw std::runtime_error( std::string("Error opening file ") +
                                config_file_path );
    std::ostringstream string_stream;
    string_stream << input_file.rdbuf();

    json = nlohmann::json::parse(string_stream.str());

    return true;

  } catch (std::exception & ex) {
    GST_ERROR_OBJECT(plugin, "Unable to parse config file: %s", ex.what());
    return false;
  }
}

#endif //EVXX_NLOHMANN_PARSER