#ifndef STRINGS_H
#define STRINGS_H

#include <string>
#include <vector>
#include <sstream>

namespace std {

std::vector<std::string> Split(const std::string& origin,
                               char delimiter) {
    std::stringstream test(origin.c_str());
    std::string segment;
    std::vector<std::string> seglist;

    while(std::getline(test, segment, delimiter)) {
        seglist.push_back(segment);
    }

    return seglist;
}

} // namespace std

#endif // STRINGS_H
