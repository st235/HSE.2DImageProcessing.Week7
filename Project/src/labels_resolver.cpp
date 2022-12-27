#include "labels_resolver.h"

#include <fstream>
#include <vector>

#include "strings.h"

namespace {

template<class K, class V>
void WriteMap(std::ofstream& stream, const std::unordered_map<K, V>& map) {
    for (auto const& entry: map) {
        stream << entry.first
               << ','
               << entry.second
               << std::endl;
    }

    stream << std::endl;
}

} // namespace

namespace detection {

LabelsResolver::LabelsResolver():
    _next_available_id(0),
    _label_to_id_lookup_table(),
    _id_to_label_lookup_table() {
    // empty on purpose
}

LabelsResolver::LabelsResolver(const LabelsResolver& that):
    _next_available_id(that._next_available_id),
    _label_to_id_lookup_table(that._label_to_id_lookup_table),
    _id_to_label_lookup_table(that._id_to_label_lookup_table) {
    // empty on purpose
}

LabelsResolver& LabelsResolver::operator=(const LabelsResolver& that) {
    if (this != &that) {
        this->_next_available_id = that._next_available_id;
        this->_label_to_id_lookup_table = that._label_to_id_lookup_table;
        this->_id_to_label_lookup_table = that._id_to_label_lookup_table;
    }

    return *this;
}

bool LabelsResolver::hasId(uint32_t id) const {
    return _id_to_label_lookup_table.find(id) != _id_to_label_lookup_table.end();
}

uint32_t LabelsResolver::obtainIdByLabel(const std::string& label) {
    if (_label_to_id_lookup_table.find(label) == _label_to_id_lookup_table.end()) {
        _label_to_id_lookup_table[label] = _next_available_id;
        _id_to_label_lookup_table[_next_available_id] = label;
        _next_available_id += 1;
    }

    return _label_to_id_lookup_table[label];
}

std::string LabelsResolver::obtainLabelById(uint32_t id) {
    if (_id_to_label_lookup_table.find(id) == _id_to_label_lookup_table.end()) {
        throw std::runtime_error("Cannot find the given id.");
    }

    return _id_to_label_lookup_table[id];
}

uint32_t LabelsResolver::operator[](const std::string& label) {
    return this->obtainIdByLabel(label);
}

void LabelsResolver::write(const std::string& file) const {
    std::ofstream file_storage;
    file_storage.open(file, std::ios::out | std::ios::trunc);

    file_storage << _next_available_id << std::endl;
    WriteMap(file_storage, _label_to_id_lookup_table);
    WriteMap(file_storage, _id_to_label_lookup_table);
}

void LabelsResolver::read(const std::string& file) {
    std::ifstream file_storage;
    file_storage.open(file, std::ios::in);

    std::string line;
    std::getline(file_storage, line);

    _next_available_id = static_cast<uint32_t>(stoi(line));

    while (std::getline(file_storage, line))
    {
        if (line.empty()) {
            break;
        }

        std::vector<std::string> tokens = std::Split(line, ',' /* delimiter */);

        std::string label = tokens[0];
        uint32_t id = static_cast<uint32_t>(stoi(tokens[1]));

        _label_to_id_lookup_table[label] = id;
    }

    while (std::getline(file_storage, line))
    {
        if (line.empty()) {
            break;
        }

        std::vector<std::string> tokens = std::Split(line, ',' /* delimiter */);

        uint32_t id = static_cast<uint32_t>(stoi(tokens[0]));
        std::string label = tokens[1];

        _id_to_label_lookup_table[id] = label;
    }
}

} // namespace detection
