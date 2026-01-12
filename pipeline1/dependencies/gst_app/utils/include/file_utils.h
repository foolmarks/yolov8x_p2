#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string> 
#include <vector>
#include <sstream>

namespace utils {
    class FileUtils {
        public: 
            static bool file_exists(const std::string& path);
            
    };
}
#endif // FILE_UTILS_H