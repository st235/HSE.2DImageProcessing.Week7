#include "args_parser.h"

namespace {

bool IsFlag(const std::string& flag) {
    if (flag.size() <= 1) {
        return false;
    }

    return flag[0] == '-';
}

} // namespace

namespace args {

ArgsDict ParseArgs(int argc, char* argv[]) {
    if (argc <= 1) {
        throw std::runtime_error("Arguments list is empty");
    }

    // skip program name
    int index = 1;

    ArgsDict result;

    std::string prev_flag = FLAG_TITLE_UNSPECIFIED;
    std::vector<std::string> values;

    while (index < argc) {
        std::string argument(argv[index]);

        if (IsFlag(argument)) {
            if (prev_flag != FLAG_TITLE_UNSPECIFIED || !values.empty()) {
                result[prev_flag] = values;
            }

            if (result.find(argument) != result.end()) {
                // found a duplicate argument
                throw std::runtime_error(argument + " was twice on the input");
            }

            prev_flag = argument;
            values.clear();
        } else {
            values.push_back(argument);
        }

        index++;
    }

    if (prev_flag != FLAG_TITLE_UNSPECIFIED || !values.empty()) {
        result[prev_flag] = values;
    }

    return result;
}

bool DetectArgs(const ArgsDict& dict, 
                const std::unordered_set<std::string>& compulsory_flags,
                const std::unordered_set<std::string>& optional_flags) {
  for (const auto& comp_flag: compulsory_flags) {
    if (dict.find(comp_flag) == dict.end()) {
        return false;
    }
  }

  for (const auto& item: dict) {
    const auto& flag = item.first;
    const auto& values = item.second;

    bool is_compulsory_key = (compulsory_flags.find(flag) != compulsory_flags.end());
    bool is_optional_key = (optional_flags.find(flag) != optional_flags.end());

    if (!is_compulsory_key && !is_optional_key) {
      return false;
    }
  }

  return true;
}

bool HasFlag(const ArgsDict& args, const std::string& arg) {
    return args.find(arg) != args.end();
}

int GetInt(const ArgsDict& args, const std::string& arg, int default_val) {
    if (!HasFlag(args, arg)) {
        return default_val;
    }

    const auto& values = args.at(arg);
    if (values.size() != 1) {
        throw std::runtime_error("Cannot extract one value for flag " + arg);
    }

    return std::stoi(values[0]);
}

std::string GetString(const ArgsDict& args, const std::string& arg) {
    const auto& values = args.at(arg);

    if (values.size() != 1) {
        throw std::runtime_error("Cannot extract one value for flag " + arg);
    }

    return values[0];
}

std::string GetString(const ArgsDict& args, 
                      const std::string& arg,
                      const std::string& default_value) {
    if (!HasFlag(args, arg)) {
        return default_value;
    }

    const auto& values = args.at(arg);

    if (values.size() != 1) {
        throw std::runtime_error("Cannot extract one value for flag " + arg);
    }

    return values[0];
}

std::vector<std::string> GetStringList(const ArgsDict& args, const std::string& arg) {
    return args.at(arg);
}

} // namespace args
