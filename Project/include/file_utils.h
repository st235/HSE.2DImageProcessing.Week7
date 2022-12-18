#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <filesystem>
#include <string>
#include <vector>

namespace utils {

std::string Join(const std::vector<std::string>& paths);

bool IsFile(const std::string& path);

std::string GetAbsolutePath(const std::string& path);

std::string GetFileName(const std::string& path);

std::string GetFileExtension(const std::string& path);

std::string GetFileNameWithExtension(const std::string& path);

std::string ReplaceFilename(const std::string& path, const std::string new_name);

bool IsDirectory(const std::string& path);

void ListFiles(const std::string& dir,
               std::vector<std::string>& result);

} // namespace utils

#endif // FILE_UTILS_H
