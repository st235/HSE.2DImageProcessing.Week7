#include "file_utils.h"

#include "strings.h"

namespace utils {

namespace fs = std::filesystem;

bool Exists(const std::string& path) {
    return fs::exists(path);
}

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

bool IsFile(const std::string& path) {
    const fs::path filepath(path);
    std::error_code error_code;
    if (fs::is_regular_file(filepath, error_code)) {
        return true;
    }
    return false;
}

bool IsDirectory(const std::string& path) {
    const fs::path filepath(path);
    std::error_code error_code;
    if (fs::is_directory(filepath, error_code)) {
        return true;
    }
    return false;
}

std::string GetAbsolutePath(const std::string& path) {
    const fs::path filepath(path);

    if (filepath.is_relative()) {
        return fs::canonical(filepath);
    }

    return fs::absolute(filepath);
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

std::string ReplaceFilenameWithExtension(const std::string& path, const std::string new_name) {
    fs::path filepath(path);
    filepath.replace_filename(new_name);
    std::string new_filepath(filepath.c_str());
    return new_filepath;
}

void ListFiles(const std::string& dir,
               std::vector<std::string>& out_files,
               const std::unordered_set<std::string>& filter_extensions) {
    for (const auto& entry: fs::directory_iterator(dir)) {
        const auto& path = entry.path();
        std::string raw_path(path.c_str());

        if (IsFile(raw_path) && (filter_extensions.empty() ||
            (filter_extensions.find(GetFileExtension(raw_path)) != filter_extensions.end()))) {
            out_files.push_back(raw_path);
        } else if (IsDirectory(raw_path)) {
            ListFiles(raw_path, out_files, filter_extensions);
        }
    }
}

void FlatListDirectories(const std::string& dir,
                         std::vector<std::string>& result) {
    for (const auto& entry: fs::directory_iterator(dir)) {
        const auto& path = entry.path();
        std::string raw_path(path.c_str());

        if (IsDirectory(raw_path)) {
            result.push_back(raw_path);
        }
    }
}

std::vector<std::string> SplitPath(const std::string& path) {
    return std::Split(path, fs::path::preferred_separator);
}

std::vector<std::string> ListAllFiles(const std::vector<std::string>& raw_files,
                                      const std::vector<std::string>& filter_extensions) {
    std::vector<std::string> flat_files;
    std::unordered_set<std::string> extensions(filter_extensions.begin(), filter_extensions.end());

    for (const auto& raw_file: raw_files) {
        if (utils::IsDirectory(raw_file)) {
            utils::ListFiles(raw_file, flat_files, extensions);
        } else {
            flat_files.push_back(raw_file);
        }
    }

    // always process images in the same order,
    // it would be easier to visually debug them
    std::sort(flat_files.begin(), flat_files.end());
    return flat_files;
}

} // namespace utils
