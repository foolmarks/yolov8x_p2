#include <string_utils.h>
#include <string> 
#include <vector>
#include <sstream>
#include <regex>
#include <iostream>
#include <nlohmann/json.hpp>

namespace utils {
namespace StringUtils {

    std::string remove_single_quotes(const std::string input)
    {
        std::string result;
        for (char ch : input) {
            if (ch != '\'') {
                result += ch;
            }
        }
        return result;
    }

    std::vector<std::string> split(const std::string &s, char delimiter)
    {
        std::vector<std::string> res;
        std::string token;
        std::istringstream tokenStream(s);
        while(std::getline(tokenStream, token, delimiter)){
            strip(token);
            if (!token.empty()) res.push_back(token);
        }
        return res;
    }

    void strip(std::string& str)
    {
        if (str.length() == 0) {
            return;
        }

        auto start_it = str.begin();
        auto end_it = str.rbegin();
        while (std::isspace(*start_it)) {
            ++start_it;
            if (start_it == str.end()) break;
        }
        while (std::isspace(*end_it)) {
            ++end_it;
            if (end_it == str.rend()) break;
        }
        int start_pos = start_it - str.begin();
        int end_pos = end_it.base() - str.begin();
        str = start_pos <= end_pos ? std::string(start_it, end_it.base()) : "";
    }

    bool starts_with(const std::string main_string, const std::string query)
    {
        if(main_string.rfind(query, 0) == 0){
        return true;
        }
        return false;
    }

    std::string find_string_by_regex(std::string input, std::string regex_str)
    {
        std::regex reg(regex_str); 
        auto it = std::sregex_iterator( input.begin(), input.end(), reg);
        return it == std::sregex_iterator() ? std::string() : it->str();
    }

    void regex_replace_all_instances(std::string re_str,
                                     std::vector<std::string> &replacements,
                                     std::string &input_string)
    {

        std::regex re(re_str);
        std::vector<std::string> matches;
        auto words_begin = std::sregex_iterator(input_string.begin(), input_string.end(), re);
        auto words_end = std::sregex_iterator();
        
        //find all regex matches, record exact matches
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            matches.push_back(match.str());
        }

        std::cout << "Found Matches: " << std::endl;
        for (auto &x : matches) {
            std::cout << "Found: " << x << std::endl;
        }
        
        // number of matches should == number of replacements
        if(!(matches.size() == replacements.size())) { 
            std::cerr << "Number of cmd line params is not matching the tags" 
                      << "in gst-string. ";
            exit(1);
        }

        //replace each entry
        //replace each matched instance with replacement
        for(int i = 0; i < matches.size(); i++) {
            std::regex placeholder(matches[i]);
            input_string = std::regex_replace(input_string, placeholder, replacements[i]);
        }

        std::cout << "GST STRING AFTER REPLACING TAGS: " << std::endl;
        std::cout << input_string << std::endl;
    }

    void string_replace(std::string &input_string, 
                        std::string tofind, std::string replace)
    {
        std::regex replacing_regex(tofind);
        input_string = std::regex_replace(input_string, replacing_regex, replace);
    }

    json string_to_json(std::string json_str)
    {
        try{
            json jsonObj = json::parse(json_str);
            return jsonObj;
        }
        catch (json::parse_error& ex) {
            std::cerr << "Error Parsing jsonstring at byte: " << ex.byte 
                      << std::endl;
            exit(1);
        }
        catch(...){
            std::cerr << "Something went wrong, unknown error. Exiting."; 
        }
        
    }
} // namespace StringUtils
} // namespace utils