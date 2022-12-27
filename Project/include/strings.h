#ifndef STRINGS_H
#define STRINGS_H

#include <string>
#include <vector>
#include <sstream>

namespace std {

static std::vector<std::string> Split(const std::string& origin,
                                      char delimiter) {
    std::stringstream test(origin.c_str());
    std::string segment;
    std::vector<std::string> seglist;

    while(std::getline(test, segment, delimiter)) {
        seglist.push_back(segment);
    }

    return seglist;
}

template <typename T>
static std::string to_string(T value) {
    std::ostringstream os;
    os << value;
    return os.str();
}


} // namespace std

#endif // STRINGS_H
