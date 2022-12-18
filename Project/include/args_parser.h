#ifndef ARGS_PARSER_H
#define ARGS_PARSER_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace args {

const std::string FLAG_TITLE_UNSPECIFIED = "";

typedef std::unordered_map<std::string, std::vector<std::string>> ArgsDict;

ArgsDict ParseArgs(int argc, char* argv[]);

bool DetectArgs(const ArgsDict& dict, 
                const std::unordered_set<std::string>& compulsory_flags,
                const std::unordered_set<std::string>& optional_flags = {});

bool HasFlag(const ArgsDict& args, const std::string& arg);

int GetInt(const ArgsDict& args, const std::string& arg, int default_val = -1);

std::string GetString(const ArgsDict& args,
                      const std::string& arg);

std::string GetString(const ArgsDict& args, 
                      const std::string& arg,
                      const std::string& default_value);

std::vector<std::string> GetStringList(const ArgsDict& args, const std::string& arg);

}

#endif // ARGS_PARSER_H
