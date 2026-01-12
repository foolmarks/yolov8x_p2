#ifndef GST_APP_UTILS_H
#define GST_APP_UTILS_H

#include <vector>

void print_usage();

std::vector<std::string> split(const std::string &s, char delimiter);

void parse_cmdline_args(int argc, char *argv[],
                                  std::string &manifest_json_path,
                                  std::string &gst_string,
                                  std::vector<std::string> &rtsp_urls,
                                  std::vector<std::string> &host_ips,
                                  std::vector<std::string> &host_ports);
bool validate_required_parameters(const std::string &gst_string,
                                  const std::string &manifest_json_path);

void print_vector(const std::vector<std::string> &vec);

bool check_required_params(const std::string &gst_string,
                         const std::string &manifest_json_path);

void print_parsed_values(const std::string &gst_string,
                         const std::string &manifest_json_path,
                         const std::vector<std::string> &rtsp_urls,
                         const std::vector<std::string> &host_ips,
                         const std::vector<std::string> &host_ports);

void strip(std::string& str);

#endif // GST_APP_UTILS_H