#include <file_utils.h> 
#include <filesystem>
#include <string> 
#include <vector>
#include <sstream>
namespace utils {
    
    bool FileUtils::file_exists(const std::string& path){
        return std::filesystem::exists(path);
    }
}