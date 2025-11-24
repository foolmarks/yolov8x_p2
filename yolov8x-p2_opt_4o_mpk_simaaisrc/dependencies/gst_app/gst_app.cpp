#include <iostream>
#include <csignal>
#include <sstream>
#include <getopt.h>
#include <pipeline.h>
#include <cmdline_utils.h>

Pipeline* pipeline_obj_ptr = nullptr;

void signalHandler(int signum) {
    std::cout << "Interrupt signal code received: " << signum <<std::endl;
    pipeline_obj_ptr->terminate_pipeline();
}

int main(int argc, char *argv[]){

    // register signal handler for SIGTERM and SIGINT
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGUSR1, signalHandler);
    signal(SIGUSR2, signalHandler);

    std::string gst_string, manifest_json_path;
    std::vector<std::string> rtsp_urls, host_ips, host_ports;
    json gst_replacement_json;
    bool enable_lttng = true;

    utils::CmdLineUtils::parse_cmdline_args(argc, argv, manifest_json_path, gst_string, rtsp_urls, host_ips, host_ports, gst_replacement_json, enable_lttng);

    if(!utils::CmdLineUtils::check_required_params(manifest_json_path, gst_string)){
        return 1;
    }

    utils::CmdLineUtils::print_parsed_values(gst_string, manifest_json_path, rtsp_urls, host_ips, host_ports, gst_replacement_json, enable_lttng);

    Pipeline pipeline_obj = Pipeline(manifest_json_path, gst_string, rtsp_urls, host_ips, host_ports, gst_replacement_json, enable_lttng);
    pipeline_obj_ptr = &pipeline_obj;

    pipeline_obj.pipeline_driver();
}