#ifndef CMDLINE_UTILS_H
#define CMDLINE_UTILS_H
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace utils {

namespace CmdLineUtils {

    void print_usage();

    void parse_cmdline_args(int argc, char *argv[],
                            std::string &manifest_json_path,
                            std::string &gst_string,
                            std::vector<std::string> &rtsp_urls,
                            std::vector<std::string> &host_ips,
                            std::vector<std::string> &host_ports,
                            json &gst_replacement_json,
                            bool &enable_lttng);

    bool validate_required_parameters(const std::string &gst_string,
                                      const std::string &manifest_json_path);

    void print_vector(const std::vector<std::string> vec);

    bool check_required_params( const std::string &gst_string,
                                const std::string &manifest_json_path);

    void print_parsed_values( const std::string &gst_string,
                              const std::string &manifest_json_path,
                              const std::vector<std::string> &rtsp_urls,
                              const std::vector<std::string> &host_ips,
                              const std::vector<std::string> &host_ports,
                              json &gst_replacement_json,
                              bool enable_lttng);

} // namespace CmdLineUtils
} // namespace utils

#endif // CMDLINE_UTILS_H