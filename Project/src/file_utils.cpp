#include "file_utils.h"

namespace utils {

namespace fs = std::filesystem;

std::string Join(const std::vector<std::string>& paths) {
    std::string result;

    uint32_t index = 0;
    for (const auto& path: paths) {
        if (index > 0) {
            result += fs::path::preferred_separator;
        }

        result += path;
        index++;
    }

    return result;
}

std::string GetAbsolutePath(const std::string& path) {
    const fs::path filepath(path);

    if (filepath.is_relative()) {
        return fs::canonical(filepath);
    }

    return fs::absolute(filepath);
}

bool IsFile(const std::string& path) {
    const fs::path filepath(path);
    std::error_code error_code;
    if (fs::is_regular_file(filepath, error_code)) {
        return true;
    }
    return false;
}

std::string GetFileName(const std::string& path) {
    const fs::path filepath(path);
    return filepath.stem();
}

std::string GetFileExtension(const std::string& path) {
    const fs::path filepath(path);
    return filepath.extension();
}

std::string GetFileNameWithExtension(const std::string& path) {
    const fs::path filepath(path);
    return filepath.filename();
}

std::string ReplaceFilename(const std::string& path, const std::string new_name) {
    fs::path filepath(path);
    const auto& extension = filepath.extension();
    filepath.replace_filename(new_name);
    std::string new_filepath(filepath.c_str());
    new_filepath += extension;
    return new_filepath;
}

bool IsDirectory(const std::string& path) {
    const fs::path filepath(path);
    std::error_code error_code;
    if (fs::is_directory(filepath, error_code)) {
        return true;
    }
    return false;
}

void ListFiles(const std::string& dir, std::vector<std::string>& result) {
    for (const auto& entry: fs::directory_iterator(dir)) {
        const auto& path = entry.path();
        std::string raw_path(path.c_str());

        if (IsFile(raw_path)) {
            result.push_back(raw_path);
        } else if (IsDirectory(raw_path)) {
            ListFiles(raw_path, result);
        }
    }
}

} // namespace utils
