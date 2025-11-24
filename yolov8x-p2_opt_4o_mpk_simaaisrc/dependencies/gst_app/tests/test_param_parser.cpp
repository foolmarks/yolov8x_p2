#include <iostream>
#include <pipeline.h>
#include <csignal>
#include <getopt.h>
#include <sstream>
#include <cmdline_utils.h>

int main(int argc, char *argv[]){

    std::string gst_string, manifest_json_path;
    std::vector<std::string> rtsp_urls, host_ips, host_ports;
    json gst_replacement_json;
    bool enable_lttng;

    utils::CmdLineUtils::parse_cmdline_args(argc, argv, manifest_json_path, gst_string, rtsp_urls, host_ips, host_ports, gst_replacement_json, enable_lttng);

    if(! utils::CmdLineUtils::check_required_params(manifest_json_path, gst_string)){
        return 1;
    }

    utils::CmdLineUtils::print_parsed_values(gst_string, manifest_json_path, rtsp_urls, host_ips, host_ports, gst_replacement_json, enable_lttng);

    exit(0);

}