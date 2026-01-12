#include <iostream>
#include <getopt.h>
#include <sstream>
#include <vector>

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

std::vector<std::string> split(const std::string &s, char delimiter){
    std::vector<std::string> res;
    std::string token;
    std::istringstream tokenStream(s);
    while(std::getline(tokenStream, token, delimiter)){
        strip(token);
        if (!token.empty()) res.push_back(token);
    }
    return res;
}

void print_usage() {
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "  --manifest-json <path>  Path to the manifest JSON file (required)\n"
              << "  --gst-string <string>   GStreamer pipeline string (required)\n"
              << "  --rtsp-url <urls>       \"url1 url2 url3\" Space-separated list of RTSP URLs (optional)\n"
              << "  --host-ip <ips>         \"ip1 ip2 ip3\" Space-separated list of host IP addresses (optional)\n"
              << "  --host-port <ports>     \"port1 port2 port3\" Space-separated list of host port numbers (optional)\n"
              << std::endl;
}

void print_vector(std::vector<std::string> vec){
    for(auto x: vec){
        std::cout << x << std::endl;
    }
    std::cout << "\n\n";
}

void parse_cmdline_args(
    int argc, char *argv[],
    std::string &manifest_json_path,
    std::string &gst_string,
    std::vector<std::string> &rtsp_urls,
    std::vector<std::string> &host_ips,
    std::vector<std::string> &host_ports) 
{

    struct option cmdline_options[] = {
        {"manifest-json", required_argument, 0, 'm'},
        {"gst-string", required_argument, 0, 'g'},
        {"rtsp-url", required_argument, 0, 'r'},
        {"host-ip", required_argument, 0, 'i'},
        {"host-port", required_argument, 0, 'p'},
        {0,0,0,0}
    };

    int opt;
    int option_index = 0;

    while((opt = getopt_long(argc, argv, "m:g:r:i:p:", cmdline_options, &option_index)) != -1) {
        switch(opt){
            case 'm':
                manifest_json_path = optarg;
                break;
            case 'g':
                gst_string = optarg;
                break;
            case 'r':
                rtsp_urls = split(optarg, ' ');
                break;
            case 'i':  
                host_ips = split(optarg, ' ');
                break;
            case 'p':
                host_ports = split(optarg, ' ');
                break;
            default: 
                print_usage();
                exit(1);
            
        }
    }
}

bool check_required_params(const std::string &manifest_json_path, const std::string &gst_string){

    if((gst_string.empty() || manifest_json_path.empty())){
        std::cerr << "Error: Misssing params\n";
        print_usage();
        return false;
    }
    return true;

}

void print_parsed_values(const std::string &gst_string,
                         const std::string &manifest_json_path,
                         const std::vector<std::string> &rtsp_urls,
                         const std::vector<std::string> &host_ips,
                         const std::vector<std::string> &host_ports) {

    std::cout << "gst-string: " << gst_string << std::endl;
    std::cout << "manifest-json: " << manifest_json_path << std::endl;

    if (!rtsp_urls.empty()) {
        std::cout << "Rtsp URLs: \n";
        print_vector(rtsp_urls);
    }

    if (!host_ips.empty()) {
        std::cout << "Host IPs: \n";
        print_vector(host_ips);
    }

    if (!host_ports.empty()) {
        std::cout << "Host Ports: \n";
        print_vector(host_ports);
    }
}