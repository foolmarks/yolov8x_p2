#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <gst/gst.h>
#include <pipeline.h>
#include <manifest_parser.h>
#include <thread>
#include <simaai/simaailog.h>

int main(int argc, char *argv[]) {
    std::string manifest_json_path;
    gst_init(nullptr, nullptr);
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " --manifest-json=<manifest json> " << std::endl;
        return -1;
    }

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--manifest-json=", 11) == 0) {
            manifest_json_path = argv[i] + 16;
        } 
        else if (std::strcmp(argv[i], "--manifest-json") == 0 && i + 1 < argc) {
            manifest_json_path = argv[++i];
        }
        else {
            std::cout << "Args have not been passed properly, please check usage." << std::endl;
        }
    }

    std::cout << "Manifest Json Path: " << manifest_json_path << std::endl;

    ManifestParser parser =  ManifestParser(manifest_json_path);
    std::string pipeline_name = parser.get_pipeline_name();

    std::cout<< "Pipeline Id: " << pipeline_name << std::endl;


}