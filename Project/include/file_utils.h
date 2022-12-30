#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <filesystem>
#include <string>
#include <vector>
#include <unordered_set>

namespace utils {

bool Exists(const std::string& path);

bool IsFile(const std::string& path);

bool IsDirectory(const std::string& path);

std::string Join(const std::vector<std::string>& paths);

std::string GetAbsolutePath(const std::string& path);

std::string GetFileName(const std::string& path);

std::string GetFileExtension(const std::string& path);

std::string GetFileNameWithExtension(const std::string& path);

std::string ReplaceFilename(const std::string& path, const std::string new_name);

std::string ReplaceFilenameWithExtension(const std::string& path, const std::string new_name);

void ListFiles(const std::string& dir,
               std::vector<std::string>& out_files,
               const std::unordered_set<std::string>& filter_extensions = {});

void FlatListDirectories(const std::string& dir,
                         std::vector<std::string>& result);

std::vector<std::string> SplitPath(const std::string& path);

std::vector<std::string> ListAllFiles(const std::vector<std::string>& raw_files,
                                      const std::vector<std::string>& filter_extensions = {});

} // namespace utils

#endif // FILE_UTILS_H
