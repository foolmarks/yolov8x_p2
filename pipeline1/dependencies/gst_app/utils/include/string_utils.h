#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string>
#include <vector>
#include <regex>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace utils {
namespace StringUtils {
    
    std::string remove_single_quotes(const std::string input);
    
    std::vector<std::string> split(const std::string &s, char delimiter);
    
    void strip(std::string& str);
    
    /// @brief checks if `main_string` starts with `querry` 
    bool starts_with(const std::string main_string, const std::string query);

    /// @brief searches for substring in `input` that matches regex `regex_str`
    /// @return on success string. On fail - empty string
    std::string find_string_by_regex(std::string input, std::string regex_str);
    
    void regex_replace_all_instances(std::string re_str, 
                                     std::vector<std::string>& replacements, 
                                    std::string& input_string);
    
    /// @brief converts parsed to string JSON file `json_str` to `nlohmann::json` 
    json string_to_json(std::string json_str);
    
    /// @brief Search for a regular expression within a string for multiple
    ///   times, and replace the matched parts through filling a format string.
    /// @param input_string string to be modified
    /// @param tofind regex used for search
    /// @param replace format string
    void string_replace(std::string &input_string, 
                        std::string tofind, 
                        std::string replace);
} // namespace StringUtils
} // namespace utils

#endif // STRING_UTILS_H